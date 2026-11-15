#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def category_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # edges are the lower bounds; the final category has no upper bound.
    return np.searchsorted(edges[1:], values, side="right")


def plot_roebber(metrics, ax=None, title=None):
    """Plot products on a Roebber performance diagram."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    sr_grid, pod_grid = np.meshgrid(
        np.linspace(0.001, 0.999, 400),
        np.linspace(0.001, 0.999, 400),
    )
    csi = 1 / (1 / sr_grid + 1 / pod_grid - 1)
    contours = ax.contour(
        sr_grid,
        pod_grid,
        csi,
        levels=np.arange(0.1, 1, 0.1),
        colors="0.5",
        linewidths=0.8,
    )
    ax.clabel(contours, fmt="CSI %.1f", fontsize=19)

    for frequency_bias in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5):
        x = np.linspace(0, min(1, 1 / frequency_bias), 200)
        ax.plot(x, frequency_bias * x, "--", color="0.7", linewidth=0.8)
        ax.text(
            x[-1] * 0.92,
            frequency_bias * x[-1] * 0.92,
            f"B={frequency_bias:g}",
            fontsize=19,
        )

    for name, values in metrics.items():
        sr, pod = float(values["SR"]), float(values["POD"])
        if not (np.isfinite(sr) and np.isfinite(pod)):
            continue
        if not (0 <= sr <= 1 and 0 <= pod <= 1):
            raise ValueError(f"{name}: SR and POD must be in [0, 1]")
        style = {
            key: values[key]
            for key in ("color", "marker")
            if key in values
        }
        style.setdefault("marker", "o")
        ax.plot(
            sr,
            pod,
            linestyle="None",
            markersize=18,
            label=name,
            zorder=5,
            **style,
        )

    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Success ratio (1 - FAR)",
        ylabel="Probability of detection (POD)",
        title=title or "",
    )
    ax.set_xlabel("Success ratio (1 - FAR)", fontsize=22)
    ax.set_ylabel("Probability of detection (POD)", fontsize=22)
    ax.set_title(title or "", fontsize=24, pad=14)
    ax.tick_params(axis="both", labelsize=19)
    ax.set_aspect("equal", adjustable="box")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=19)
    return ax


def roebber_metrics_for_category(
    results: pd.DataFrame,
    products: list[dict],
    category: str,
) -> dict:
    markers = ["o", "s", "^", "D", "P", "X"]
    metrics = {}
    for index, product in enumerate(products):
        label = product["label"]
        row = results.loc[
            (results["product"] == label)
            & (results["category"] == category)
        ]
        if row.empty:
            continue
        metrics[label] = {
            "SR": 1.0 - float(row.iloc[0]["FAR"]),
            "POD": float(row.iloc[0]["POD"]),
            "color": product.get("color", f"C{index}"),
            "marker": markers[index % len(markers)],
        }
    return metrics


def plot_roebber_by_category(
    results: pd.DataFrame,
    products: list[dict],
    labels: list[str],
    out_dir: Path,
    scenario: str,
) -> None:
    for category in labels:
        fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)
        plot_roebber(
            roebber_metrics_for_category(results, products, category),
            ax=ax,
            title=f"Precipitation category: {category} mm/day",
        )
        category_tag = (
            category.replace(">=", "ge")
            .replace("<=", "le")
            .replace(">", "gt")
            .replace("<", "lt")
        )
        output = out_dir / f"roebber_{category_tag}_{scenario}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Roebber diagram: {output}")

    fig, axes = plt.subplots(2, 3, figsize=(26, 17), constrained_layout=True)
    handles = legend_labels = None
    for ax, category in zip(axes.flat, labels):
        plot_roebber(
            roebber_metrics_for_category(results, products, category),
            ax=ax,
            title=f"{category} mm/day",
        )
        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="outside upper center",
            ncol=3,
            frameon=False,
            fontsize=22,
        )
    output = out_dir / f"roebber_by_precipitation_category_2x3_{scenario}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined Roebber diagram: {output}")


def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    edges: np.ndarray,
    labels: list[str],
    product: str,
) -> list[dict]:
    obs_category = category_index(observed, edges)
    pred_category = category_index(predicted, edges)
    rows = []
    for category, label in enumerate(labels):
        observed_event = obs_category == category
        predicted_event = pred_category == category
        hits = int(np.sum(observed_event & predicted_event))
        misses = int(np.sum(observed_event & ~predicted_event))
        false_alarms = int(np.sum(~observed_event & predicted_event))
        correct_negatives = int(np.sum(~observed_event & ~predicted_event))
        rows.append(
            {
                "product": product,
                "category": label,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_negatives,
                "observed_count": hits + misses,
                "predicted_count": hits + false_alarms,
                "POD": safe_ratio(hits, hits + misses),
                "FAR": safe_ratio(false_alarms, hits + false_alarms),
                "CSI": safe_ratio(hits, hits + misses + false_alarms),
                "fBIAS": safe_ratio(hits + false_alarms, hits + misses),
            }
        )
    return rows


def plot_metrics(results: pd.DataFrame, products: list[dict], labels: list[str], output: Path):
    colors = [p.get("color", f"C{i}") for i, p in enumerate(products)]
    product_labels = [p["label"] for p in products]
    x = np.arange(len(labels), dtype=float)
    width = 0.78 / len(products)
    offsets = (np.arange(len(products)) - (len(products) - 1) / 2) * width

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    settings = [
        ("POD", "Probability of detection", (0.0, 1.0)),
        ("FAR", "False alarm ratio", (0.0, 1.0)),
        ("CSI", "Critical success index", (0.0, 1.0)),
        ("fBIAS", "Frequency bias", None),
    ]

    for ax, (metric, ylabel, ylim) in zip(axes.flat, settings):
        for i, (product, color) in enumerate(zip(product_labels, colors)):
            subset = results.loc[results["product"] == product].set_index("category")
            values = subset.loc[labels, metric].to_numpy(float)
            ax.bar(x + offsets[i], values, width=width, color=color, label=product)
        if metric == "fBIAS":
            ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--")
            ymax = max(1.15, float(np.nanmax(results[metric])) * 1.08)
            ax.set_ylim(0.0, ymax)
        else:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks(x, labels, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside upper center",
        ncol=len(products),
        frameon=False,
        fontsize=12,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_lines(
    results: pd.DataFrame,
    products: list[dict],
    labels: list[str],
    out_dir: Path,
    scenario: str,
):
    settings = [
        ("POD", "Probability of detection", (0.0, 1.0)),
        ("FAR", "False alarm ratio", (0.0, 1.0)),
        ("CSI", "Critical success index", (0.0, 1.0)),
        ("fBIAS", "Frequency bias", None),
    ]
    x = np.arange(len(labels), dtype=float)
    markers = ["o", "s", "^", "D", "P", "X"]

    for metric, ylabel, ylim in settings:
        fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
        for i, product in enumerate(products):
            label = product["label"]
            subset = results.loc[results["product"] == label].set_index("category")
            values = subset.loc[labels, metric].to_numpy(float)
            ax.plot(
                x,
                values,
                color=product.get("color", f"C{i}"),
                marker=markers[i % len(markers)],
                markersize=7,
                linewidth=2.3,
                linestyle=product.get("linestyle", "-"),
                label=label,
            )

        if metric == "fBIAS":
            ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--")
            ymax = max(1.15, float(np.nanmax(results[metric])) * 1.08)
            ax.set_ylim(0.0, ymax)
        else:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Daily precipitation category (mm/day)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks(x, labels, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center")
        output = out_dir / f"{metric}_by_precipitation_category_{scenario}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric} figure: {output}")


def plot_metric_lines_2x2(
    results: pd.DataFrame,
    products: list[dict],
    labels: list[str],
    output: Path,
):
    settings = [
        ("POD", "Probability of detection", (0.0, 1.0)),
        ("FAR", "False alarm ratio", (0.0, 1.0)),
        ("CSI", "Critical success index", (0.0, 1.0)),
        ("fBIAS", "Frequency bias", None),
    ]
    x = np.arange(len(labels), dtype=float)
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)

    for ax, (metric, ylabel, ylim) in zip(axes.flat, settings):
        for i, product in enumerate(products):
            label = product["label"]
            subset = results.loc[results["product"] == label].set_index("category")
            values = subset.loc[labels, metric].to_numpy(float)
            ax.plot(
                x,
                values,
                color=product.get("color", f"C{i}"),
                marker=markers[i % len(markers)],
                markersize=6.5,
                linewidth=2.2,
                linestyle=product.get("linestyle", "-"),
                label=label,
            )
        if metric == "fBIAS":
            ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--")
            ymax = max(1.15, float(np.nanmax(results[metric])) * 1.08)
            ax.set_ylim(0.0, ymax)
        else:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Daily precipitation category (mm/day)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x, labels, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate precipitation intensity categories.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    input_csv = Path(cfg["paths"]["eval_csv"])
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    observed_column = cfg["data"].get("observed_column", "gauge")
    products = cfg["products"]["columns"]
    edges = np.asarray(cfg["categories"]["lower_bounds_mm_day"], dtype=float)
    labels = list(cfg["categories"]["labels"])
    if len(edges) != len(labels):
        raise ValueError("One lower bound is required for each category label.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Category lower bounds must be strictly increasing.")

    filters = cfg.get("filters", {})
    common_samples = bool(filters.get("common_samples", True))
    date_column = filters.get("date_column", "date")
    columns = [observed_column] + [p["column"] for p in products]
    usecols = columns + ([date_column] if date_column not in columns else [])
    df = pd.read_csv(input_csv, usecols=usecols).replace([np.inf, -np.inf], np.nan)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        if filters.get("date_start"):
            df = df.loc[df[date_column] >= filters["date_start"]]
        if filters.get("date_end"):
            df = df.loc[df[date_column] <= filters["date_end"]]
    if common_samples:
        df = df.dropna(subset=columns)

    rows = []
    sample_counts = {}
    for product in products:
        if common_samples:
            paired = df
        else:
            paired = df.dropna(subset=[observed_column, product["column"]])
        sample_counts[product["label"]] = len(paired)
        rows.extend(
            compute_metrics(
                paired[observed_column].to_numpy(float),
                paired[product["column"]].to_numpy(float),
                edges,
                labels,
                product["label"],
            )
        )
    results = pd.DataFrame(rows)
    scenario = cfg["meta"]["scenario"]
    csv_path = out_dir / f"precipitation_category_metrics_{scenario}.csv"
    figure_path = out_dir / f"precipitation_category_metrics_2x2_{scenario}.png"
    results.to_csv(csv_path, index=False, float_format="%.6f")
    plot_metrics(results, products, labels, figure_path)
    plot_metric_lines(results, products, labels, out_dir, scenario)
    line_panel_path = out_dir / f"precipitation_category_metrics_lines_2x2_{scenario}.png"
    plot_metric_lines_2x2(results, products, labels, line_panel_path)
    print(f"Saved 2x2 line figure: {line_panel_path}")
    plot_roebber_by_category(results, products, labels, out_dir, scenario)

    if common_samples:
        print(f"Common station-days: {len(df):,}")
    else:
        print("Available station-days by product:")
        for product, count in sample_counts.items():
            print(f"  {product}: {count:,}")
    print(results.to_string(index=False))
    print(f"Saved metrics: {csv_path}")
    print(f"Saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
