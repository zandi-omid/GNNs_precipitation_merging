#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def paired(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    return obs[valid], sim[valid]


def metrics(obs, sim):
    obs, sim = paired(obs, sim)
    result = {"n_samples": len(obs)}
    if len(obs) < 2:
        return result | {k: np.nan for k in ("KGE", "CC", "RMSE", "Bias", "Alpha", "Beta")}
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    obs_mean = np.mean(obs)
    cc = np.corrcoef(obs, sim)[0, 1] if obs_std > 0 and sim_std > 0 else np.nan
    alpha = sim_std / obs_std if obs_std > 0 else np.nan
    beta = np.mean(sim) / obs_mean if obs_mean != 0 else np.nan
    kge = (
        1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        if np.all(np.isfinite([cc, alpha, beta]))
        else np.nan
    )
    return result | {
        "KGE": kge,
        "CC": cc,
        "RMSE": np.sqrt(np.mean((sim - obs) ** 2)),
        "Bias": np.mean(sim - obs),
        "Alpha": alpha,
        "Beta": beta,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate daily holdout-gauge predictions by generalized Koppen class."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "rb") as stream:
        cfg = tomllib.load(stream)

    eval_csv = Path(cfg["paths"]["eval_csv"])
    climate_nc = Path(cfg["paths"]["climate_nc"])
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    products = cfg["products"]["columns"]

    required = ["date", "node", "lon", "lat", "gauge"] + [p["column"] for p in products]
    daily = pd.read_csv(eval_csv, usecols=required, parse_dates=["date"])
    start = cfg.get("filters", {}).get("date_start")
    end = cfg.get("filters", {}).get("date_end")
    if start:
        daily = daily[daily["date"] >= start]
    if end:
        daily = daily[daily["date"] <= end]
    if cfg.get("filters", {}).get("common_samples", False):
        finite_columns = ["gauge"] + [p["column"] for p in products]
        finite = np.isfinite(daily[finite_columns].to_numpy(dtype=float)).all(axis=1)
        daily = daily.loc[finite].copy()
        if daily.empty:
            raise ValueError("Common finite-sample filtering removed every row.")

    nodes = daily[["node", "lon", "lat"]].drop_duplicates("node").sort_values("node")
    with xr.open_dataset(climate_nc) as climate_ds:
        climate = climate_ds["climate_class"]
        assigned = climate.sel(
            lon=xr.DataArray(nodes["lon"].to_numpy(), dims="station"),
            lat=xr.DataArray(nodes["lat"].to_numpy(), dims="station"),
            method="nearest",
        ).to_numpy()
        labels_text = climate.attrs.get("class_labels", "")

    labels = {
        int(item.split(":", 1)[0].strip()): item.split(":", 1)[1].strip()
        for item in labels_text.split(";")
        if ":" in item
    }
    nodes["climate_class"] = assigned
    if nodes["climate_class"].isna().any():
        raise ValueError("Some holdout nodes map to missing climate cells.")
    nodes["climate_class"] = nodes["climate_class"].astype(int)
    nodes["climate_label"] = nodes["climate_class"].map(labels)
    nodes.to_csv(out_dir / "holdout_node_climate_classes.csv", index=False)
    daily = daily.merge(nodes[["node", "climate_class", "climate_label"]], on="node", how="left")

    pooled_rows = []
    station_rows = []
    for (class_id, class_label), class_data in daily.groupby(
        ["climate_class", "climate_label"], sort=True
    ):
        n_nodes = class_data["node"].nunique()
        for product in products:
            column, label = product["column"], product["label"]
            pooled_rows.append({
                "climate_class": class_id,
                "climate_label": class_label,
                "n_nodes": n_nodes,
                "Product": label,
            } | metrics(class_data["gauge"], class_data[column]))
            for (node, lon, lat), station_data in class_data.groupby(["node", "lon", "lat"]):
                station_rows.append({
                    "climate_class": class_id,
                    "climate_label": class_label,
                    "node": node,
                    "lon": lon,
                    "lat": lat,
                    "Product": label,
                } | metrics(station_data["gauge"], station_data[column]))

    pooled = pd.DataFrame(pooled_rows)
    stations = pd.DataFrame(station_rows)
    metric_names = ["KGE", "CC", "RMSE", "Bias", "Alpha", "Beta"]
    summary = (
        stations.groupby(["climate_class", "climate_label", "Product"], sort=False)
        .agg(
            n_nodes=("node", "nunique"),
            **{f"{name}_mean": (name, "mean") for name in metric_names},
            **{f"{name}_median": (name, "median") for name in metric_names},
        )
        .reset_index()
    )

    pooled.to_csv(out_dir / "pooled_daily_metrics_by_climate.csv", index=False)
    stations.to_csv(out_dir / "per_node_metrics_by_climate.csv", index=False)
    summary.to_csv(out_dir / "station_summary_metrics_by_climate.csv", index=False)

    # Keep full precision above and provide compact, publication-ready tables.
    pooled.round(2).to_csv(
        out_dir / "pooled_daily_metrics_by_climate_rounded2.csv", index=False
    )
    summary.round(2).to_csv(
        out_dir / "station_summary_metrics_by_climate_rounded2.csv", index=False
    )

    counts = nodes.groupby(["climate_class", "climate_label"]).size().rename("n_nodes").reset_index()
    counts.to_csv(out_dir / "holdout_node_counts_by_climate.csv", index=False)
    print("Holdout nodes by climate class:")
    print(counts.to_string(index=False))
    print("\nPooled daily KGE by climate class:")
    print(pooled.pivot(index="climate_label", columns="Product", values="KGE").round(2).to_string())
    print(f"\nSaved climate-stratified results to: {out_dir}")


if __name__ == "__main__":
    main()
