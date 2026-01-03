# src/datasets/sic_sat_scalar_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import xarray as xr

from src.datasets import SICWindowDataset


def _ym_from_datetime64(dt64) -> str:
    # dt64: numpy datetime64
    s = np.datetime_as_string(dt64, unit="D")  # "YYYY-MM-DD"
    y = s[0:4]
    m = s[5:7]
    return f"{y}{m}"


def build_sat_scalar_lookup(
    anom_nc_path: Path,
    lat_min: float = 60.0,
) -> Dict[str, float]:
    """
    Build dict: {"YYYYMM": sat_scalar} from ERA5 sat_anom NetCDF.

    sat_scalar = area-weighted mean over lat>=lat_min using cos(lat) weights.
    """
    ds = xr.open_dataset(anom_nc_path)

    # variable name
    if "sat_anom" in ds.data_vars:
        v = ds["sat_anom"]
    else:
        raise KeyError(f"sat_anom not found in {anom_nc_path}. vars={list(ds.data_vars)}")

    # time coord name could be "time" or "valid_time"
    time_name = None
    for cand in ("time", "valid_time"):
        if cand in v.dims or cand in ds.coords:
            time_name = cand
            break
    if time_name is None:
        raise KeyError(f"time coord not found. coords={list(ds.coords)} dims={v.dims}")

    # lat/lon coord names (ERA5 usually latitude/longitude)
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"lat/lon coords not found. coords={list(ds.coords)}")

    # subset arctic band
    v = v.sel({lat_name: slice(90.0, lat_min)})

    lat = v[lat_name].values.astype(np.float64)  # (nlat,)
    w = np.cos(np.deg2rad(lat))                  # area weight
    w = w / w.mean()                             # normalize scale (optional)

    # weighted mean over lat/lon
    # v: (time, lat, lon)
    # -> first mean over lon, then weighted mean over lat
    v_lonmean = v.mean(dim=lon_name)             # (time, lat)
    num = (v_lonmean * xr.DataArray(w, dims=(lat_name,))).sum(dim=lat_name)
    den = xr.DataArray(w, dims=(lat_name,)).sum(dim=lat_name)
    sat_scalar = (num / den).astype("float32")   # (time,)

    lookup: Dict[str, float] = {}
    times = ds[time_name].values
    vals = sat_scalar.values
    for i in range(len(times)):
        ym = _ym_from_datetime64(times[i])
        lookup[ym] = float(vals[i])

    return lookup


class SICSatScalarDataset:
    """
    Wrapper on SICWindowDataset:
      returns (x, y, meta) where meta includes "sat_scalar".
    """

    def __init__(self, base: SICWindowDataset, sat_lookup: Dict[str, float]):
        self.base = base
        self.sat_lookup = sat_lookup

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        x, y, meta = self.base[idx]
        ym = str(meta["t_out"])  # "YYYYMM"
        if ym not in self.sat_lookup:
            raise KeyError(f"t_out {ym} not found in SAT lookup (size={len(self.sat_lookup)})")
        meta = dict(meta)  # copy
        meta["sat_scalar"] = float(self.sat_lookup[ym])
        return x, y, meta
