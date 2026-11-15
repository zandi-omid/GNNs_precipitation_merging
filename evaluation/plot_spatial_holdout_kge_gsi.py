#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ModuleNotFoundError as exc:
    raise SystemExit("cartopy is required for spatial KGE maps.") from exc

try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None


def load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot spatial distribution of holdout-station KGE from GSI per-node metrics CSVs."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--metric", default=None, help="Metric column to plot, default from config or KGE.")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--dpi", type=int, default=None)
    return p.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")


def read_train_gauges(path: Optional[Path]) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame(columns=["lon", "lat"])
    if not path.exists():
        print(f"Train gauge CSV not found; skipping train markers: {path}")
        return pd.DataFrame(columns=["lon", "lat"])

    df = pd.read_csv(path)
    colmap = {}
    if {"LONGITUDE", "LATITUDE"}.issubset(df.columns):
        colmap = {"LONGITUDE": "lon", "LATITUDE": "lat"}
    elif {"lon", "lat"}.issubset(df.columns):
        colmap = {"lon": "lon", "lat": "lat"}
    elif {"longitude", "latitude"}.issubset(df.columns):
        colmap = {"longitude": "lon", "latitude": "lat"}
    else:
        raise ValueError(
            f"Could not find lon/lat columns in {path}. Available columns: {df.columns.tolist()}"
        )
    return (
        df[list(colmap)]
        .rename(columns=colmap)
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def read_dem(path: Optional[Path]):
    if path is None or str(path).strip() == "":
        return None
    if rasterio is None:
        print("rasterio is unavailable; skipping DEM background.")
        return None
    if not path.exists():
        print(f"DEM not found; skipping DEM background: {path}")
        return None

    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)
        transform = src.transform
        ny, nx = dem.shape
        x_coords = np.arange(nx) * transform.a + transform.c
        y_coords = np.arange(ny) * transform.e + transform.f
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    return dem, extent


def select_products(per_node: pd.DataFrame, products_cfg: List[Dict[str, Any]]) -> pd.DataFrame:
    if not products_cfg:
        raise ValueError("Config must define [[products]] entries.")

    frames = []
    available = sorted(per_node["Product"].dropna().unique().tolist())
    for item in products_cfg:
        source = str(item["source"])
        label = str(item.get("label", source))
        tag = str(item.get("tag", safe_name(label)))
        sub = per_node.loc[per_node["Product"] == source].copy()
        if sub.empty:
            raise ValueError(
                f"Product {source!r} not found in per-node metrics CSV. "
                f"Available products: {available}"
            )
        sub["PlotLabel"] = label
        sub["PlotTag"] = tag
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def compute_extent(selected: pd.DataFrame, train: pd.DataFrame, pad_deg: float) -> List[float]:
    lon = selected["lon"].to_numpy(dtype=float)
    lat = selected["lat"].to_numpy(dtype=float)
    if len(train):
        lon = np.concatenate([lon, train["lon"].to_numpy(dtype=float)])
        lat = np.concatenate([lat, train["lat"].to_numpy(dtype=float)])
    return [
        float(np.nanmin(lon) - pad_deg),
        float(np.nanmax(lon) + pad_deg),
        float(np.nanmin(lat) - pad_deg),
        float(np.nanmax(lat) + pad_deg),
    ]


def add_base_map(
    ax,
    extent,
    dem_payload,
    train,
    cfg,
    projection,
    *,
    show_left_labels: bool = True,
    show_bottom_labels: bool = True,
):
    if dem_payload is not None and bool(cfg_get(cfg, "style", "show_dem", True)):
        dem, dem_extent = dem_payload
        ax.imshow(
            dem,
            extent=dem_extent,
            origin="upper",
            cmap=str(cfg_get(cfg, "style", "dem_cmap", "terrain")),
            alpha=float(cfg_get(cfg, "style", "dem_alpha", 0.45)),
            transform=projection,
            zorder=0,
        )

    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black", zorder=1)
    ax.add_feature(states, linewidth=0.7, edgecolor="white", zorder=2)
    ax.set_extent(extent, crs=projection)

    if len(train) and bool(cfg_get(cfg, "style", "show_train_gauges", True)):
        ax.scatter(
            train["lon"],
            train["lat"],
            s=float(cfg_get(cfg, "style", "train_marker_size", 18)),
            c=str(cfg_get(cfg, "style", "train_marker_color", "black")),
            marker=str(cfg_get(cfg, "style", "train_marker", "^")),
            edgecolors="white",
            linewidths=0.25,
            transform=projection,
            zorder=3,
            label="Train gauges",
        )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="gray",
        alpha=0.55,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.bottom_labels = show_bottom_labels
    grid_label_size = float(cfg_get(cfg, "style", "tick_label_size", 12))
    gl.xlabel_style = {"size": grid_label_size}
    gl.ylabel_style = {"size": grid_label_size}


def plot_one(
    df: pd.DataFrame,
    label: str,
    metric: str,
    out_png: Path,
    cfg: Dict[str, Any],
    train: pd.DataFrame,
    dem_payload,
    extent,
    cmap,
    norm,
    dpi: int,
) -> None:
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            float(cfg_get(cfg, "style", "single_width", 10.0)),
            float(cfg_get(cfg, "style", "single_height", 8.0)),
        ),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    add_base_map(ax, extent, dem_payload, train, cfg, projection)

    values = df[metric].astype(float)
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=values.clip(norm.vmin, norm.vmax),
        s=float(cfg_get(cfg, "style", "holdout_marker_size", 45)),
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.45,
        transform=projection,
        zorder=4,
        label="Holdout gauges",
    )
    ax.set_title(f"{label}: Holdout Station {metric}", weight="bold")
    ax.legend(loc=str(cfg_get(cfg, "style", "legend_loc", "lower left")), frameon=True)
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.78, pad=0.02)
    cbar.set_label(str(cfg_get(cfg, "metric", "label", metric)))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)


def plot_panel(
    selected: pd.DataFrame,
    metric: str,
    out_png: Path,
    cfg: Dict[str, Any],
    train: pd.DataFrame,
    dem_payload,
    extent,
    cmap,
    norm,
    dpi: int,
) -> None:
    projection = ccrs.PlateCarree()
    labels = selected["PlotLabel"].drop_duplicates().tolist()
    ncols = int(cfg_get(cfg, "style", "panel_ncols", min(2, len(labels))))
    ncols = max(1, min(ncols, len(labels)))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            float(cfg_get(cfg, "style", "panel_col_width", 7.2)) * ncols,
            float(cfg_get(cfg, "style", "panel_row_height", 6.2)) * nrows,
        ),
        subplot_kw={"projection": projection},
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    last_sc = None
    for panel_index, (ax, label) in enumerate(zip(axes_flat, labels)):
        row, col = divmod(panel_index, ncols)
        add_base_map(
            ax,
            extent,
            dem_payload,
            train,
            cfg,
            projection,
            show_left_labels=(col == 0),
            show_bottom_labels=(row == nrows - 1),
        )
        sub = selected.loc[selected["PlotLabel"] == label]
        last_sc = ax.scatter(
            sub["lon"],
            sub["lat"],
            c=sub[metric].astype(float).clip(norm.vmin, norm.vmax),
            s=float(cfg_get(cfg, "style", "holdout_marker_size", 45)),
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.45,
            transform=projection,
            zorder=4,
        )
        ax.set_title(label, weight="bold")

    for ax in axes_flat[len(labels):]:
        ax.set_visible(False)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes_flat[:len(labels)].tolist(), orientation="vertical", shrink=0.82, pad=0.02)
        cbar.set_label(
            str(cfg_get(cfg, "metric", "label", metric)),
            fontsize=float(cfg_get(cfg, "style", "axis_label_size", 14)),
        )
        cbar.ax.tick_params(labelsize=float(cfg_get(cfg, "style", "tick_label_size", 12)))
    if bool(cfg_get(cfg, "style", "show_title", True)):
        fig.suptitle(str(cfg_get(cfg, "meta", "title", f"Holdout Station {metric}")), weight="bold")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)


def plot_delta(
    selected: pd.DataFrame,
    metric: str,
    out_png: Path,
    cfg: Dict[str, Any],
    train: pd.DataFrame,
    dem_payload,
    extent,
    dpi: int,
) -> None:
    delta_cfg = cfg.get("delta", {})
    left = str(delta_cfg.get("left_product", ""))
    right = str(delta_cfg.get("right_product", ""))
    left_label = str(delta_cfg.get("left_label", left))
    right_label = str(delta_cfg.get("right_label", right))
    if not left or not right:
        raise ValueError("[delta] requires left_product and right_product.")

    left_df = selected.loc[selected["PlotLabel"] == left, ["node", "lon", "lat", metric]].copy()
    right_df = selected.loc[selected["PlotLabel"] == right, ["node", metric]].copy()
    if left_df.empty:
        raise ValueError(f"Delta left product {left!r} was not selected.")
    if right_df.empty:
        raise ValueError(f"Delta right product {right!r} was not selected.")

    merged = left_df.merge(right_df, on="node", suffixes=("_left", "_right"), how="inner")
    merged[f"delta_{metric}"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    delta_col = f"delta_{metric}"

    out_csv = out_png.with_suffix(".csv")
    merged.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(merged[delta_col].describe())

    vmax = float(delta_cfg.get("vmax", cfg_get(cfg, "metric", "delta_vmax", 0.3)))
    vmin = float(delta_cfg.get("vmin", -vmax))
    cmap = plt.get_cmap(str(delta_cfg.get("cmap", "RdBu_r")))
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    projection = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            float(cfg_get(cfg, "style", "single_width", 10.0)),
            float(cfg_get(cfg, "style", "single_height", 8.0)),
        ),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    add_base_map(ax, extent, dem_payload, train, cfg, projection)
    sc = ax.scatter(
        merged["lon"],
        merged["lat"],
        c=merged[delta_col].clip(vmin, vmax),
        s=float(cfg_get(cfg, "style", "holdout_marker_size", 58)),
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.45,
        transform=projection,
        zorder=4,
    )
    ax.set_title(f"{left_label} minus {right_label}: Holdout Station {metric}", weight="bold")
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.78, pad=0.02)
    cbar.set_label(f"Delta {metric} ({left_label} - {right_label})")
    cbar.set_ticks(np.linspace(vmin, vmax, 7))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    scenario = str(cfg_get(cfg, "meta", "scenario", "spatial_holdout_kge"))
    metric = str(args.metric or cfg_get(cfg, "metric", "name", "KGE"))
    per_node_csv = Path(cfg_get(cfg, "paths", "per_node_metrics_csv"))
    out_dir = Path(cfg_get(cfg, "paths", "out_dir"))
    train_csv_raw = cfg_get(cfg, "paths", "train_gauge_csv", "")
    dem_raw = cfg_get(cfg, "paths", "dem", "")
    train_csv = Path(train_csv_raw) if str(train_csv_raw).strip() else None
    dem = Path(dem_raw) if str(dem_raw).strip() else None

    vmin = float(args.vmin if args.vmin is not None else cfg_get(cfg, "metric", "vmin", -0.2))
    vmax = float(args.vmax if args.vmax is not None else cfg_get(cfg, "metric", "vmax", 0.8))
    dpi = int(args.dpi if args.dpi is not None else cfg_get(cfg, "style", "dpi", 400))

    plt.rcParams.update({
        "font.size": float(cfg_get(cfg, "style", "font_size", 13)),
        "axes.titlesize": float(cfg_get(cfg, "style", "title_size", 15)),
        "axes.labelsize": float(cfg_get(cfg, "style", "axis_label_size", 14)),
        "xtick.labelsize": float(cfg_get(cfg, "style", "tick_label_size", 12)),
        "ytick.labelsize": float(cfg_get(cfg, "style", "tick_label_size", 12)),
        "legend.fontsize": float(cfg_get(cfg, "style", "legend_size", 11)),
    })

    print("Scenario:", scenario)
    print("Per-node metrics CSV:", per_node_csv)
    print("Output dir:", out_dir)
    print("Metric:", metric)

    per_node = pd.read_csv(per_node_csv)
    if metric not in per_node.columns:
        raise ValueError(f"Metric {metric!r} not found. Available columns: {per_node.columns.tolist()}")
    required = {"node", "lon", "lat", "Product", metric}
    missing = sorted(required - set(per_node.columns))
    if missing:
        raise ValueError(f"Missing required columns in {per_node_csv}: {missing}")

    selected = select_products(per_node, cfg.get("products", []))
    selected = selected[np.isfinite(selected[metric].astype(float))].copy()
    train = read_train_gauges(train_csv)
    dem_payload = read_dem(dem)
    extent = cfg_get(cfg, "map", "extent", None)
    if extent is None:
        extent = compute_extent(
            selected,
            train,
            pad_deg=float(cfg_get(cfg, "map", "pad_deg", 0.4)),
        )
    extent = [float(x) for x in extent]

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_csv = out_dir / f"selected_spatial_holdout_{safe_name(metric)}_{scenario}.csv"
    selected.to_csv(selected_csv, index=False)
    print("Saved:", selected_csv)

    summary = (
        selected
        .groupby("PlotLabel", as_index=False)
        .agg(
            n_nodes=(metric, "count"),
            metric_mean=(metric, "mean"),
            metric_median=(metric, "median"),
            metric_min=(metric, "min"),
            metric_max=(metric, "max"),
        )
        .sort_values("metric_median", ascending=False)
    )
    summary_csv = out_dir / f"summary_spatial_holdout_{safe_name(metric)}_{scenario}.csv"
    summary.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)
    print(summary)

    cmap = plt.get_cmap(str(cfg_get(cfg, "metric", "cmap", "turbo")))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plot_panel(
        selected=selected,
        metric=metric,
        out_png=out_dir / f"spatial_holdout_{safe_name(metric)}_panel_{scenario}.png",
        cfg=cfg,
        train=train,
        dem_payload=dem_payload,
        extent=extent,
        cmap=cmap,
        norm=norm,
        dpi=dpi,
    )

    if bool(cfg_get(cfg, "delta", "enabled", False)):
        plot_delta(
            selected=selected,
            metric=metric,
            out_png=out_dir / f"spatial_holdout_delta_{safe_name(metric)}_{scenario}.png",
            cfg=cfg,
            train=train,
            dem_payload=dem_payload,
            extent=extent,
            dpi=dpi,
        )

    if bool(cfg_get(cfg, "style", "save_individual", True)):
        for label in selected["PlotLabel"].drop_duplicates():
            sub = selected.loc[selected["PlotLabel"] == label]
            tag = str(sub["PlotTag"].iloc[0])
            plot_one(
                df=sub,
                label=label,
                metric=metric,
                out_png=out_dir / f"spatial_holdout_{safe_name(metric)}_{safe_name(tag)}_{scenario}.png",
                cfg=cfg,
                train=train,
                dem_payload=dem_payload,
                extent=extent,
                cmap=cmap,
                norm=norm,
                dpi=dpi,
            )


if __name__ == "__main__":
    main()
