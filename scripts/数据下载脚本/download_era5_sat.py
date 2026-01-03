# scripts/数据下载脚本/download_era5_sat.py

from pathlib import Path
import numpy as np
import xarray as xr

# -------------------------
# robust project root (based on this file location)
# scripts/数据下载脚本/download_era5_sat.py
# -> project_root = .../physics-consistent-arctic-sea-ice-forecast
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ERA5_DIR = RAW_DIR / "era5_sat"
ERA5_DIR.mkdir(parents=True, exist_ok=True)

era5_t2m_path = ERA5_DIR / "era5_t2m_monthly_1979_2022_arctic.nc"
era5_sat_anom_path = ERA5_DIR / "era5_sat_anom_monthly_1979_2022_arctic.nc"

print("PROJECT_ROOT ->", PROJECT_ROOT)
print("ERA5_DIR      ->", ERA5_DIR)
print("ERA5 t2m      ->", era5_t2m_path)
print("SAT anomaly   ->", era5_sat_anom_path)

# -------------------------
# (1) download t2m if needed
# -------------------------
import cdsapi

if era5_t2m_path.exists():
    print("Already exists, skip download:", era5_t2m_path)
else:
    c = cdsapi.Client()
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": "2m_temperature",
        "year": [str(y) for y in range(1979, 2023)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        # [North, West, South, East]
        "area": [90, -180, 60, 180],
        "format": "netcdf",
    }
    print("Requesting ERA5 monthly t2m (this can take a while)...")
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        request,
        str(era5_t2m_path),
    )
    print("Downloaded:", era5_t2m_path)

# -------------------------
# (2) build SAT anomaly
# -------------------------
ds = xr.open_dataset(era5_t2m_path)

print("Dataset loaded.")
print("Sizes:", ds.sizes)
print("Data variables:", list(ds.data_vars))
print("Coords:", list(ds.coords))

# variable
t2m = ds["t2m"] if "t2m" in ds.data_vars else ds[list(ds.data_vars)[0]]
print("Using variable:", t2m.name)
print("t2m.dims:", t2m.dims)

# detect time dim name (ERA5 often uses valid_time)
time_dim = None
for cand in ["time", "valid_time"]:
    if cand in t2m.dims:
        time_dim = cand
        break

if time_dim is None:
    # fallback: find a datetime64 coord in dims
    for d in t2m.dims:
        if d in ds.coords and np.issubdtype(ds[d].dtype, np.datetime64):
            time_dim = d
            break

if time_dim is None:
    raise KeyError(
        f"Cannot find time dimension. t2m.dims={t2m.dims}, coords={list(ds.coords)}"
    )

print("Using time dimension:", time_dim)

# climatology by month along the detected time dim
clim = t2m.groupby(f"{time_dim}.month").mean(time_dim)
sat_anom = (t2m.groupby(f"{time_dim}.month") - clim).astype("float32").rename("sat_anom")

# normalize time dim name to "time" for downstream code simplicity
if time_dim != "time":
    sat_anom = sat_anom.rename({time_dim: "time"})

out = xr.Dataset({"sat_anom": sat_anom})
out.to_netcdf(era5_sat_anom_path)

print("Saved SAT anomaly to:", era5_sat_anom_path)
