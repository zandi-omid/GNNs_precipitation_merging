#!/usr/bin/env python3
# coding: utf-8

"""
Build one master evaluation CSV for test-period gauge-supported points.

Output columns:
    date, y, x, lon, lat,
    gauge,
    idw_loo_p0, idw_loo_p2,
    tgcn, era5, imerg
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

OUT_CSV = "/xdisk/behrangi/omidzandi/GNNs/evaluation/master_test_gauge_comparison.csv"

# Choose which TGCN field to export
TGCN_VAR = "pred_det"   # or "pred_expected_mean" / "pred_median"

# ============================================================
# LOAD
# ============================================================
print("Loading datasets...")
ds_tgcn = xr.open_dataset(NC_TGCN)
ds_idw  = xr.open_dataset(NC_IDW)

if TGCN_VAR not in ds_tgcn:
    raise ValueError(f"{TGCN_VAR} not found in TGCN dataset")

required_idw = ["idw_loo_p0", "idw_loo_p2"]
for v in required_idw:
    if v not in ds_idw:
        raise ValueError(f"{v} not found in IDW dataset")

# sanity checks
for dim in ["time", "y", "x"]:
    if ds_tgcn.sizes[dim] != ds_idw.sizes[dim]:
        raise ValueError(f"Dimension mismatch for {dim}: {ds_tgcn.sizes[dim]} vs {ds_idw.sizes[dim]}")

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

print("Datasets loaded.")
print("Gauge-supported samples:", int(gauge_mask.sum()))

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
    })

df = pd.DataFrame(rows)

# optional: sort nicely
df = df.sort_values(["date", "y", "x"]).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================
out_path = Path(OUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\nSaved master CSV to:\n{out_path}")
print("\nColumns:")
print(df.columns.tolist())
print(f"\nRows: {len(df):,}")
print("\nHead:")
print(df.head())