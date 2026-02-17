#%%
#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/reg_analysis/daily_cc_mlr.csv")
NC_PATH  = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_log_normal_test2020_2024/pred_inputs_daily_maps.nc")

OUT_DIR  = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/reg_analysis/plots_best_worst_1x3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# how many best/worst days to plot
K = 6

# pick which metric defines best/worst
# - "cc_gain_vs_best" (your earlier choice)
# - or "cc_era5", "cc_imerg", "cc_mlr"
# RANK_BY = "cc_gain_vs_best"
RANK_BY = "cc_mlr"

def _origin_from_lat(lat2d: np.ndarray) -> str:
    lat_top = np.nanmean(lat2d[0, :])
    lat_bot = np.nanmean(lat2d[-1, :])
    # if top row has larger latitude => top is north => origin='upper' puts north on top
    return "upper" if lat_top > lat_bot else "lower"

def _extent_from_latlon(lat2d: np.ndarray, lon2d: np.ndarray):
    lon_min = float(np.nanmin(lon2d)); lon_max = float(np.nanmax(lon2d))
    lat_min = float(np.nanmin(lat2d)); lat_max = float(np.nanmax(lat2d))
    return [lon_min, lon_max, lat_min, lat_max]

def _robust_limits(*arrays, frac=0.8):
    """
    vmin = 0
    vmax = frac * max value across all finite values in the input arrays
    """
    vals = []
    for a in arrays:
        if a is None:
            continue
        v = a[np.isfinite(a)]
        if v.size > 0:
            vals.append(v)

    if not vals:
        return 0.0, 1.0

    allv = np.concatenate(vals)
    vmax_raw = np.nanmax(allv)

    if (not np.isfinite(vmax_raw)) or vmax_raw <= 0:
        return 0.0, 1.0

    vmax = float(frac * vmax_raw)
    return 0.0, vmax

def make_gauge_map(gauge: np.ndarray, gauge_mask: np.ndarray) -> np.ndarray:
    gmap = np.full_like(gauge, np.nan, dtype=np.float32)
    ok = gauge_mask.astype(bool) & np.isfinite(gauge)
    gmap[ok] = gauge[ok].astype(np.float32)
    return gmap

def plot_day_1x3(ds: xr.Dataset, day: str, tag: str, out_png: Path):
    lat2d = ds["lat"].values
    lon2d = ds["lon"].values
    origin = _origin_from_lat(lat2d)
    extent = _extent_from_latlon(lat2d, lon2d)

    dsel = ds.sel(time=np.datetime64(day))

    era = dsel["era5"].values.astype(np.float32)
    im  = dsel["imerg"].values.astype(np.float32)

    if "gauge" not in dsel:
        raise KeyError("NetCDF is missing variable 'gauge'. Add it to the writer first.")
    if "gauge_mask" not in dsel:
        raise KeyError("NetCDF is missing variable 'gauge_mask'.")

    gauge = dsel["gauge"].values.astype(np.float32)
    gmask = dsel["gauge_mask"].values.astype(bool)

    gmap = make_gauge_map(gauge, gmask)

    # shared color limits for the day (same across ERA5/IMERG/GAUGE)
    vmin, vmax = _robust_limits(era, im, gmap, frac=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)

    im0 = axes[0].imshow(era, origin=origin, extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"ERA5 | {day}")
    axes[0].set_xlabel("Lon"); axes[0].set_ylabel("Lat")

    im1 = axes[1].imshow(im, origin=origin, extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"IMERG | {day}")
    axes[1].set_xlabel("Lon"); axes[1].set_ylabel("Lat")

    im2 = axes[2].imshow(gmap, origin=origin, extent=extent, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"GAUGE (masked) | {day}")
    axes[2].set_xlabel("Lon"); axes[2].set_ylabel("Lat")

    # One shared colorbar for the whole figure
    cbar = fig.colorbar(im0, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Precipitation (mm/day)")

    fig.suptitle(f"{day}  |  {tag}  |  shared vmin={vmin:.2f}, vmax={vmax:.2f}", y=1.03, fontsize=13)
    plt.show
    # fig.savefig(out_png, dpi=200)
    # plt.close(fig)

def main():

    df = pd.read_csv(CSV_PATH).replace([np.inf, -np.inf], np.nan)

    ds = xr.open_dataset(NC_PATH)

    # Only keep days present in this NetCDF (2020–2024)
    ds_days = pd.to_datetime(ds["time"].values).strftime("%Y-%m-%d")
    df = df[df["date"].isin(ds_days)].copy()

    # Only keep rows with cc_mlr available
    df = df.dropna(subset=["cc_mlr"])

    # Best/Worst by cc_mlr
    best = df.sort_values("cc_mlr", ascending=False).head(K)
    worst = df.sort_values("cc_mlr", ascending=True).head(K)

    print("NetCDF range:", ds_days.min(), "->", ds_days.max())
    print("Using days:", len(df))
    print("Best cc_mlr:", best[["date","n_gauges","cc_mlr"]].head(3).to_string(index=False))
    print("Worst cc_mlr:", worst[["date","n_gauges","cc_mlr"]].head(3).to_string(index=False))

    # Now you can safely rank/select:
    df = df.dropna(subset=[RANK_BY])
    best = df.sort_values(RANK_BY, ascending=False).head(K)
    worst = df.sort_values(RANK_BY, ascending=True).head(K)

    best = df.sort_values(RANK_BY, ascending=False).head(K)
    worst = df.sort_values(RANK_BY, ascending=True).head(K)

    ds = xr.open_dataset(NC_PATH)

    # quick orientation info
    lat2d = ds["lat"].values
    print("Lat top row:", float(np.nanmean(lat2d[0,:])),
          "Lat bottom row:", float(np.nanmean(lat2d[-1,:])),
          "=> origin:", _origin_from_lat(lat2d))

    # BEST
    for _, r in best.iterrows():
        day = str(r["date"])
        tag = f"BEST {RANK_BY}={r[RANK_BY]:.3f}, n={int(r['n_gauges'])}"
        out_png = OUT_DIR / f"BEST_{RANK_BY}_{day}.png"
        plot_day_1x3(ds, day, tag, out_png)
        print("Wrote", out_png)

    # WORST
    for _, r in worst.iterrows():
        day = str(r["date"])
        tag = f"WORST {RANK_BY}={r[RANK_BY]:.3f}, n={int(r['n_gauges'])}"
        out_png = OUT_DIR / f"WORST_{RANK_BY}_{day}.png"
        plot_day_1x3(ds, day, tag, out_png)
        print("Wrote", out_png)

    ds.close()
    print("Done. Output dir:", OUT_DIR)

#%%
if __name__ == "__main__":
    main()
# %%
