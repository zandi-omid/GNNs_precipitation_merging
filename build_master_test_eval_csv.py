#!/usr/bin/env python3
# coding: utf-8

"""
Build one master evaluation CSV for test-period gauge-supported points.

Output columns:
    date, y, x, lon, lat,
    gauge,
    idw_loo_p0, idw_loo_p2,
    tgcn, era5, imerg, prism
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# CONFIG
# ============================================================
NC_TGCN = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_avg10_test2020_2024/pred_inputs_daily_maps.nc"
NC_IDW  = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/IDW_test2020_2024/pred_inputs_daily_maps_IDW.nc"
NC_PRISM = "/xdisk/behrangi/omidzandi/GNNs/data/PRISM_daily_5km/PRISM_on_ERA5_DAILY_2005_2024.nc"

OUT_CSV = "/xdisk/behrangi/omidzandi/GNNs/evaluation/master_test_gauge_comparison.csv"

# Choose which TGCN field to export
TGCN_VAR = "pred_det"   # or "pred_expected_mean" / "pred_median"


# ============================================================
# HELPERS
# ============================================================
def get_var(ds, preferred, fallback_list):
    if preferred in ds.data_vars:
        return ds[preferred]
    for v in fallback_list:
        if v in ds.data_vars:
            return ds[v]
    raise KeyError(
        f"Could not find variable '{preferred}' or any of {fallback_list} "
        f"in {list(ds.data_vars)}"
    )


# ============================================================
# LOAD
# ============================================================
print("Loading datasets...")
ds_tgcn = xr.open_dataset(NC_TGCN)
ds_idw  = xr.open_dataset(NC_IDW)
ds_prism = xr.open_dataset(NC_PRISM)

if TGCN_VAR not in ds_tgcn:
    raise ValueError(f"{TGCN_VAR} not found in TGCN dataset")

required_idw = ["idw_loo_p0", "idw_loo_p2"]
for v in required_idw:
    if v not in ds_idw:
        raise ValueError(f"{v} not found in IDW dataset")

prism_da = get_var(
    ds_prism,
    "prism_on_era5",
    ["precipitation", "prism_ppt", "ppt", "prism"]
)

# sanity checks between TGCN and IDW
for dim in ["time", "y", "x"]:
    if ds_tgcn.sizes[dim] != ds_idw.sizes[dim]:
        raise ValueError(
            f"Dimension mismatch for {dim}: "
            f"{ds_tgcn.sizes[dim]} vs {ds_idw.sizes[dim]}"
        )

gauge = ds_tgcn["gauge"].values
gauge_mask = ds_tgcn["gauge_mask"].values.astype(bool)

lon = ds_tgcn["lon"].values
lat = ds_tgcn["lat"].values

era5 = ds_tgcn["era5"].values
imerg = ds_tgcn["imerg"].values
tgcn = ds_tgcn[TGCN_VAR].values

idw0 = ds_idw["idw_loo_p0"].values
idw2 = ds_idw["idw_loo_p2"].values

times = pd.to_datetime(ds_tgcn["time"].values)

# ============================================================
# ALIGN PRISM TO TGCN GRID USING REAL LAT/LON
# ============================================================
print("Interpolating PRISM to TGCN lat/lon grid...")

# TGCN grid geolocation
lat_target = xr.DataArray(ds_tgcn["lat"].values, dims=("y", "x"))
lon_target = xr.DataArray(ds_tgcn["lon"].values, dims=("y", "x"))

# match longitude convention if needed
x_min = float(prism_da["x"].min())
x_max = float(prism_da["x"].max())

lon_vals = lon_target.values.astype("float64")
if x_min >= 0 and x_max > 180 and np.nanmin(lon_vals) < 0:
    lon_vals = (lon_vals + 360.0) % 360.0
elif x_min < 0 and x_max <= 180 and np.nanmax(lon_vals) > 180:
    lon_vals = ((lon_vals + 180.0) % 360.0) - 180.0

lon_target = xr.DataArray(lon_vals, dims=("y", "x"))

prism_on_tgcn_grid = prism_da.interp(
    time=ds_tgcn["time"],
    y=lat_target,
    x=lon_target,
    method="nearest"
)

prism = prism_on_tgcn_grid.values

print("Datasets loaded.")
print("Gauge-supported samples:", int(gauge_mask.sum()))
print("TGCN grid shape:", tgcn.shape)
print("PRISM aligned shape:", prism.shape)

print("PRISM aligned shape:", prism.shape)
print("PRISM NaN count:", np.isnan(prism).sum())
print("PRISM finite count:", np.isfinite(prism).sum())

if np.isfinite(prism).any():
    print("PRISM min/max:", np.nanmin(prism), np.nanmax(prism))

# ============================================================
# BUILD MASTER TABLE
# ============================================================
print("Building master table...")
tt, yy, xx = np.where(gauge_mask)

rows = []
for t_idx, i, j in zip(tt, yy, xx):
    rows.append({
        "date": str(times[t_idx].date()),
        "y": int(i),
        "x": int(j),
        "lon": float(lon[i, j]) if np.isfinite(lon[i, j]) else np.nan,
        "lat": float(lat[i, j]) if np.isfinite(lat[i, j]) else np.nan,
        "gauge": float(gauge[t_idx, i, j]) if np.isfinite(gauge[t_idx, i, j]) else np.nan,
        "idw_loo_p0": float(idw0[t_idx, i, j]) if np.isfinite(idw0[t_idx, i, j]) else np.nan,
        "idw_loo_p2": float(idw2[t_idx, i, j]) if np.isfinite(idw2[t_idx, i, j]) else np.nan,
        "tgcn": float(tgcn[t_idx, i, j]) if np.isfinite(tgcn[t_idx, i, j]) else np.nan,
        "era5": float(era5[t_idx, i, j]) if np.isfinite(era5[t_idx, i, j]) else np.nan,
        "imerg": float(imerg[t_idx, i, j]) if np.isfinite(imerg[t_idx, i, j]) else np.nan,
        "prism": float(prism[t_idx, i, j]) if np.isfinite(prism[t_idx, i, j]) else np.nan,
    })

df = pd.DataFrame(rows)

# optional: sort nicely
df = df.sort_values(["date", "y", "x"]).reset_index(drop=True)

# ============================================================
# QUICK SANITY CHECKS
# ============================================================
print("\nColumns:")
print(df.columns.tolist())

print(f"\nRows: {len(df):,}")

print("\nNaN counts:")
print(df[[
    "gauge", "idw_loo_p0", "idw_loo_p2",
    "tgcn", "era5", "imerg", "prism"
]].isna().sum())

print("\nSummary stats:")
print(df[[
    "gauge", "idw_loo_p2", "tgcn", "era5", "imerg", "prism"
]].describe())

# ============================================================
# SAVE
# ============================================================
out_path = Path(OUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\nSaved master CSV to:\n{out_path}")
print("\nHead:")
print(df.head())