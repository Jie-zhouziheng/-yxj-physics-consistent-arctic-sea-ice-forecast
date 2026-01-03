# src/datasets/sat_scalar.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import xarray as xr
import torch


def _ym_int_from_str(ym: str) -> int:
    # ym: "YYYYMM"
    ym = str(ym)
    return int(ym[:4]) * 100 + int(ym[4:6])


@dataclass
class SatScalarLookup:
    """
    Build a lookup table from ERA5 SAT anomaly file:
      sat_anom(time, latitude, longitude)  ->  sat_scalar(time)

    Default: area-weighted mean over lat/lon using cos(lat).
    """
    nc_path: Union[str, Path]
    var_name: str = "sat_anom"
    time_name: str = "time"
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    use_coslat_weight: bool = True

    _map: Dict[int, float] = None  # ym_int -> scalar

    def __post_init__(self):
        self.nc_path = Path(self.nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"SAT anomaly file not found: {self.nc_path}")

        ds = xr.open_dataset(self.nc_path)

        # Be robust to time coordinate naming (just in case)
        if self.time_name not in ds.coords and self.time_name not in ds.dims:
            # common alternative: valid_time
            if "valid_time" in ds.coords or "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
                self.time_name = "time"
            else:
                raise KeyError(f"Cannot find time coordinate in dataset. coords={list(ds.coords)} dims={list(ds.dims)}")

        if self.var_name not in ds.data_vars:
            raise KeyError(f"Variable '{self.var_name}' not found in {self.nc_path}. data_vars={list(ds.data_vars)}")

        da = ds[self.var_name]

        # Ensure lat/lon exist
        if self.lat_name not in da.coords:
            # sometimes it's "lat"
            if "lat" in da.coords:
                da = da.rename({"lat": self.lat_name})
            else:
                raise KeyError(f"Latitude coord not found. coords={list(da.coords)}")

        if self.lon_name not in da.coords:
            if "lon" in da.coords:
                da = da.rename({"lon": self.lon_name})
            else:
                raise KeyError(f"Longitude coord not found. coords={list(da.coords)}")

        # Add month coordinate if not present (optional convenience)
        if "month" not in ds.coords:
            month = da[self.time_name].dt.month
            da = da.assign_coords(month=("time", month.values))

        # Weighted mean over lat/lon -> scalar per time
        if self.use_coslat_weight:
            lat = da[self.lat_name]
            w = np.cos(np.deg2rad(lat.values)).astype(np.float32)  # (lat,)
            w = xr.DataArray(w, dims=(self.lat_name,), coords={self.lat_name: lat})
            # weighted mean over latitude then mean over longitude
            # (weights only in lat dim)
            scalar = (da * w).sum(self.lat_name) / w.sum(self.lat_name)
            scalar = scalar.mean(self.lon_name)
        else:
            scalar = da.mean([self.lat_name, self.lon_name])

        scalar = scalar.load()  # pull into memory (only 528 values)

        # Build map: YYYYMM -> float
        times = scalar[self.time_name].values
        vals = scalar.values.astype(np.float32)

        self._map = {}
        for t, v in zip(times, vals):
            # t is numpy datetime64 month-start
            dt = np.datetime64(t).astype("datetime64[M]").astype(object)
            ym_int = dt.year * 100 + dt.month
            self._map[ym_int] = float(v)

        ds.close()

    def get_scalar_list(self, t_out_list: List[str]) -> List[float]:
        out = []
        for ym in t_out_list:
            ym_int = _ym_int_from_str(ym)
            if ym_int not in self._map:
                raise KeyError(f"SAT scalar missing for ym={ym} (ym_int={ym_int}). Check SAT time coverage.")
            out.append(self._map[ym_int])
        return out

    def get_scalar_tensor(self, t_out_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Returns tensor shape (B,) float32.
        """
        vals = self.get_scalar_list(t_out_list)
        return torch.tensor(vals, dtype=torch.float32, device=device)
