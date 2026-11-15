#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.colors import BoundaryNorm

from train_stack_meta_mlp_gsi import MetaMLP


SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall": (9, 10, 11),
    "autumn": (9, 10, 11),
}


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a ranked wet day from gauge observations within a season and "
            "plot six precipitation maps plus gauge scatter diagnostics."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--season",
        required=True,
        choices=sorted(SEASON_MONTHS),
        help="Meteorological season: winter=DJF, spring=MAM, summer=JJA, fall/autumn=SON.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="One-based rank by daily mean gauge precipitation within the selected season.",
    )
    parser.add_argument(
        "--date-start",
        default=None,
        help="Optional test-period start override (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-end",
        default=None,
        help="Optional test-period end override (YYYY-MM-DD).",
    )
    return parser.parse_args()


def nearest_quantile(da: xr.DataArray, quantiles: xr.DataArray, q: float) -> np.ndarray:
    index = int(np.argmin(np.abs(quantiles.values.astype(float) - float(q))))
    return np.asarray(da.isel(quantile=index).values, dtype=np.float32)


def stack_day(
    ds_gat: xr.Dataset,
    ds_mlp: xr.Dataset,
    time_index: int,
    model_path: Path,
    output_quantile: float,
) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_cols = list(checkpoint["feature_cols"])
    expected = {
        "gatgcn_q65",
        "gatgcn_q70",
        "gatgcn_q75",
        "mlp_q65",
        "mlp_q70",
        "mlp_q75",
    }
    if set(feature_cols) != expected:
        raise ValueError(
            "This event-map utility expects the six base-model quantiles only; "
            f"the stacker contains {feature_cols}."
        )

    gat_day = ds_gat["pred_quantile"].isel(time=time_index)
    mlp_day = ds_mlp["pred_quantile_mlp"].isel(time=time_index)
    arrays = {
        "gatgcn_q65": nearest_quantile(gat_day, ds_gat["quantile"], 0.65),
        "gatgcn_q70": nearest_quantile(gat_day, ds_gat["quantile"], 0.70),
        "gatgcn_q75": nearest_quantile(gat_day, ds_gat["quantile"], 0.75),
        "mlp_q65": nearest_quantile(mlp_day, ds_mlp["quantile"], 0.65),
        "mlp_q70": nearest_quantile(mlp_day, ds_mlp["quantile"], 0.70),
        "mlp_q75": nearest_quantile(mlp_day, ds_mlp["quantile"], 0.75),
    }
    features = np.column_stack([arrays[column] for column in feature_cols]).astype(np.float32)
    finite = np.all(np.isfinite(features), axis=1)

    model = MetaMLP(
        in_dim=len(feature_cols),
        hidden_dims=[int(value) for value in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
        out_dim=len(checkpoint["quantiles"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    standardized = (
        (features[finite] - np.asarray(checkpoint["x_mean"], dtype=np.float32))
        / np.asarray(checkpoint["x_std"], dtype=np.float32)
    ).astype(np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(standardized)).cpu().numpy()
    prediction = (
        prediction * float(checkpoint["y_std"]) + float(checkpoint["y_mean"])
    )
    meta_quantiles = np.asarray(checkpoint["quantiles"], dtype=float)
    output_index = int(np.argmin(np.abs(meta_quantiles - float(output_quantile))))
    result = np.full(features.shape[0], np.nan, dtype=np.float32)
    result[finite] = np.clip(prediction[:, output_index], 0.0, None)
    return result


def find_time_index(ds: xr.Dataset, date: pd.Timestamp) -> int:
    times = pd.to_datetime(ds["time"].values).normalize()
    matches = np.flatnonzero(times == date.normalize())
    if not len(matches):
        raise ValueError(f"{date.date()} is absent from {ds.encoding.get('source', 'dataset')}.")
    return int(matches[0])


def select_nodes_from_grid(
    path: Path,
    variable: str,
    date: pd.Timestamp,
    lon: np.ndarray,
    lat: np.ndarray,
    lon_name: str,
    lat_name: str,
    require_complete_day: bool = False,
) -> np.ndarray:
    with xr.open_dataset(path, cache=False) as dataset:
        time_index = find_time_index(dataset, date)
        if require_complete_day and "hours_present" in dataset:
            hours = int(dataset["hours_present"].isel(time=time_index).values)
            if hours != 24:
                raise ValueError(f"{date.date()} has only {hours} GMCP hours.")
        field = dataset[variable].isel(time=time_index)
        node_coord = xr.DataArray(np.arange(len(lon)), dims="node")
        selected = field.sel(
            {
                lon_name: xr.DataArray(lon, dims="node", coords={"node": node_coord}),
                lat_name: xr.DataArray(lat, dims="node", coords={"node": node_coord}),
            },
            method="nearest",
        )
        return np.asarray(selected.values, dtype=np.float32)


def select_prism_nodes(path: Path, date: pd.Timestamp, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    with xr.open_dataset(path, cache=False) as dataset:
        time_index = find_time_index(dataset, date)
        field = dataset["prism_on_era5"].isel(time=time_index)
        selected = field.interp(
            x=xr.DataArray(lon, dims="node"),
            y=xr.DataArray(lat, dims="node"),
            method="linear",
        )
        return np.asarray(selected.values, dtype=np.float32)


def nodes_to_grid(lon: np.ndarray, lat: np.ndarray, values: np.ndarray):
    rounded_lon = np.round(lon, 6)
    rounded_lat = np.round(lat, 6)
    unique_lon = np.sort(np.unique(rounded_lon))
    unique_lat = np.sort(np.unique(rounded_lat))
    grid = np.full((len(unique_lat), len(unique_lon)), np.nan, dtype=np.float32)
    lon_index = {value: index for index, value in enumerate(unique_lon)}
    lat_index = {value: index for index, value in enumerate(unique_lat)}
    for longitude, latitude, value in zip(rounded_lon, rounded_lat, values):
        grid[lat_index[latitude], lon_index[longitude]] = value
    dx = float(np.median(np.diff(unique_lon)))
    dy = float(np.median(np.diff(unique_lat)))
    extent = [
        float(unique_lon.min() - dx / 2),
        float(unique_lon.max() + dx / 2),
        float(unique_lat.min() - dy / 2),
        float(unique_lat.max() + dy / 2),
    ]
    return grid, extent


def select_ranked_day(
    evaluation: pd.DataFrame,
    season: str,
    rank: int,
    date_start: str,
    date_end: str,
    product_columns: list[str],
    require_all_products: bool,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    if rank < 1:
        raise ValueError("--rank must be at least 1.")
    frame = evaluation.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(date_start), pd.Timestamp(date_end))
        & frame["date"].dt.month.isin(SEASON_MONTHS[season])
    ]
    if require_all_products:
        availability = frame.groupby("date")[product_columns].count().gt(0).all(axis=1)
        frame = frame.loc[frame["date"].isin(availability.index[availability])]

    ranked = (
        frame.groupby("date", as_index=False)
        .agg(gauge_mean_mm_day=("gauge", "mean"), n_gauges=("gauge", "count"))
        .sort_values(["gauge_mean_mm_day", "date"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(ranked) < rank:
        raise ValueError(f"Only {len(ranked)} eligible {season} days are available; rank {rank} is invalid.")
    selected = pd.Timestamp(ranked.iloc[rank - 1]["date"])
    return selected, ranked


def plot_maps(
    products: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    date: pd.Timestamp,
    season: str,
    rank: int,
    output: Path,
    plot_cfg: dict[str, Any],
) -> None:
    finite = np.concatenate([values[np.isfinite(values)] for values in products.values()])
    step = float(plot_cfg.get("colorbar_step", 5.0))
    percentile = float(plot_cfg.get("colorbar_percentile", 99.5))
    vmax = max(step, float(np.ceil(np.nanpercentile(finite, percentile) / step) * step))
    levels = np.arange(0.0, vmax + step, step)
    cmap = plt.get_cmap(str(plot_cfg.get("cmap", "turbo")), len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N, clip=False)

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20, 12.5),
        subplot_kw={"projection": projection},
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    image = None
    for panel_index, (axis, (label, values)) in enumerate(zip(axes.flat, products.items())):
        row, column = divmod(panel_index, 3)
        grid, extent = nodes_to_grid(lon, lat, values)
        image = axis.imshow(
            grid,
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            transform=projection,
        )
        axis.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
        axis.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black")
        axis.add_feature(states, linewidth=0.65, edgecolor="white")
        axis.set_extent(extent, crs=projection)
        axis.set_title(label, fontsize=27, weight="bold")
        gridlines = axis.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.65,
            linestyle="--",
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.left_labels = column == 0
        gridlines.bottom_labels = row == 1
        gridlines.xlabel_style = {"size": 21}
        gridlines.ylabel_style = {"size": 21}

    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        shrink=0.84,
        pad=0.02,
        extend="max",
    )
    colorbar.set_label(r"Precipitation (mm day$^{-1}$)", fontsize=25)
    colorbar.ax.tick_params(labelsize=21)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(plot_cfg.get("dpi", 400)), bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    day: pd.DataFrame,
    products: list[dict[str, str]],
    date: pd.Timestamp,
    season: str,
    rank: int,
    output: Path,
    plot_cfg: dict[str, Any],
) -> pd.DataFrame:
    columns = ["gauge", *[product["column"] for product in products]]
    finite_values = day[columns].to_numpy(dtype=float)
    upper = float(np.nanpercentile(finite_values, float(plot_cfg.get("scatter_percentile", 99.5))))
    limit = max(5.0, float(np.ceil(upper / 5.0) * 5.0))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), sharex=True, sharey=True, constrained_layout=True)
    rows = []
    for axis, product in zip(axes.flat, products):
        paired = day[["gauge", product["column"]]].dropna()
        observed = paired["gauge"].to_numpy(float)
        predicted = paired[product["column"]].to_numpy(float)
        correlation = float(np.corrcoef(observed, predicted)[0, 1]) if len(paired) > 1 else np.nan
        observed_std = float(np.std(observed, ddof=1)) if len(paired) > 1 else np.nan
        observed_mean = float(np.mean(observed)) if len(paired) else np.nan
        alpha = (
            float(np.std(predicted, ddof=1)) / observed_std
            if np.isfinite(observed_std) and observed_std > 0
            else np.nan
        )
        beta = (
            float(np.mean(predicted)) / observed_mean
            if np.isfinite(observed_mean) and observed_mean != 0
            else np.nan
        )
        kge = (
            1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
            if np.all(np.isfinite([correlation, alpha, beta]))
            else np.nan
        )
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "season": season,
                "rank": rank,
                "product": product["label"],
                "n": len(paired),
                "KGE": kge,
                "CC": correlation,
                "Alpha": alpha,
                "Beta": beta,
            }
        )
        axis.scatter(
            observed,
            predicted,
            s=28,
            alpha=0.72,
            color=product.get("color", "#3274A1"),
            edgecolors="black",
            linewidths=0.35,
        )
        axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1.3)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(product["label"], fontsize=19, weight="bold")
        axis.text(
            0.04,
            0.96,
            f"KGE = {kge:.2f}\nCC = {correlation:.2f}\n"
            rf"$\alpha$ = {alpha:.2f}" + "\n" + rf"$\beta$ = {beta:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.85},
        )
        axis.grid(alpha=0.25, linestyle="--")
        axis.tick_params(labelsize=14)

    fig.supxlabel(r"Gauge precipitation (mm day$^{-1}$)", fontsize=19)
    fig.supylabel(r"Estimated precipitation (mm day$^{-1}$)", fontsize=19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(plot_cfg.get("dpi", 400)), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)
    paths = cfg["paths"]
    plot_cfg = cfg.get("plot", {})
    date_start = args.date_start or cfg.get("period", {}).get("date_start", "2020-01-01")
    date_end = args.date_end or cfg.get("period", {}).get("date_end", "2024-12-31")
    products_cfg = list(cfg["products"]["items"])
    product_columns = [product["column"] for product in products_cfg]

    evaluation = pd.read_csv(paths["evaluation_csv"], parse_dates=["date"])
    selected_date, ranking = select_ranked_day(
        evaluation=evaluation,
        season=args.season,
        rank=args.rank,
        date_start=date_start,
        date_end=date_end,
        product_columns=product_columns,
        require_all_products=bool(cfg.get("selection", {}).get("require_all_products", True)),
    )
    selected_row = ranking.iloc[args.rank - 1]
    print(
        f"Selected {args.season} rank {args.rank}: {selected_date:%Y-%m-%d}; "
        f"mean gauge precipitation={selected_row['gauge_mean_mm_day']:.3f} mm/day; "
        f"n={int(selected_row['n_gauges'])}"
    )

    output_dir = Path(paths["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / f"{args.season}_gauge_wet_day_ranking.csv", index=False)

    with xr.open_dataset(paths["nc_gatgcn"], cache=False) as ds_gat, xr.open_dataset(
        paths["nc_mlp"], cache=False
    ) as ds_mlp:
        if not np.array_equal(ds_gat["time"].values, ds_mlp["time"].values):
            raise ValueError("GATGCN and MLP time coordinates differ.")
        time_index = find_time_index(ds_gat, selected_date)
        lon = ds_gat["node_lon"].values.astype(float)
        lat = ds_gat["node_lat"].values.astype(float)
        maps = {
            "Stacked-q70": stack_day(
                ds_gat,
                ds_mlp,
                time_index,
                Path(paths["stacker_model"]),
                output_quantile=0.70,
            ),
            "IDW": np.asarray(ds_gat["idw_train"].isel(time=time_index).values, dtype=np.float32),
            "PRISM": select_prism_nodes(Path(paths["nc_prism"]), selected_date, lon, lat),
            "MSWEP": select_nodes_from_grid(
                Path(paths["nc_mswep"]), "precipitation", selected_date, lon, lat, "lon", "lat"
            ),
            "GMCP": select_nodes_from_grid(
                Path(paths["nc_gmcp"]),
                "precipitation",
                selected_date,
                lon,
                lat,
                "lon",
                "lat",
                require_complete_day=True,
            ),
            "MRMS": select_nodes_from_grid(
                Path(paths["nc_mrms"]),
                "precipitation",
                selected_date,
                lon,
                lat,
                "longitude",
                "latitude",
            ),
        }

    stem = f"{args.season}_rank{args.rank}_{selected_date:%Y%m%d}"
    map_path = output_dir / f"precipitation_maps_2x3_{stem}.png"
    plot_maps(maps, lon, lat, selected_date, args.season, args.rank, map_path, plot_cfg)
    print("Saved:", map_path)

    day = evaluation.loc[pd.to_datetime(evaluation["date"]).dt.normalize() == selected_date].copy()
    scatter_path = output_dir / f"gauge_scatter_2x3_{stem}.png"
    scatter_metrics = plot_scatter(
        day, products_cfg, selected_date, args.season, args.rank, scatter_path, plot_cfg
    )
    metrics_path = output_dir / f"gauge_scatter_metrics_{stem}.csv"
    scatter_metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    print("Saved:", scatter_path)
    print("Saved:", metrics_path)
    print(scatter_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
