#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
import argparse
from typing import Any, Dict, List, Tuple

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRODUCTS = [
    ("idw_train", "IDW"),
    ("pred_gsi", "GSI-Mean"),
    ("era5", "ERA5"),
    ("imerg", "IMERG"),
    ("prism", "PRISM"),
    ("mrms", "MRMS"),
    ("mswep", "MSWEP"),
    ("gmcp", "GMCP"),
]

DEFAULT_COLORS = {
    "IDW": "#4C78A8",
    "GSI-Mean": "#111111",
    "ERA5": "#72B7B2",
    "IMERG": "#F2CF5B",
    "PRISM": "#B279A2",
    "MRMS": "#59A14F",
    "MSWEP": "#FF9DA6",
    "GMCP": "#9C755F",
}

METRICS_TO_PLOT = ["KGE", "CC", "RMSE", "Bias", "Alpha", "Beta"]
METRIC_AXIS_LABELS = {
    "KGE": "Kling-Gupta efficiency (KGE)",
    "CC": "Pearson correlation (CC)",
    "RMSE": r"RMSE (mm day$^{-1}$)",
    "Bias": r"Bias (mm day$^{-1}$)",
    "Alpha": r"Variability ratio ($\alpha$)",
    "Beta": r"Bias ratio ($\beta$)",
}
METRIC_YLIM = {
    "KGE": (-0.5, 1.05),
    "CC": (0.0, 1.05),
    "RMSE": None,
    "Bias": None,
    "Alpha": (0.0, 1.5),
    "Beta": (0.0, 2.0),
}


def load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot holdout gauge metrics from a GSI master evaluation CSV."
    )
    p.add_argument("--config", default="configs/plot_holdout_metrics_gsi.toml")
    p.add_argument("--eval_csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--scenario", default=None)
    p.add_argument("--quantile", type=float, default=None)
    p.add_argument("--quantiles", default=None, help="Comma-separated quantiles, e.g. 0.70,0.75,0.80")
    p.add_argument("--include_point", type=str_to_bool, default=None)
    p.add_argument("--include_unavailable_days", type=str_to_bool, default=None)
    return p.parse_args()


def quantile_col_name(q: float) -> str:
    return f"pred_quantile_q{q:.2f}".replace(".", "p")


def quantile_label(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def parse_quantiles(args, cfg: Dict[str, Any]) -> List[float]:
    if args.quantiles is not None:
        return [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    if args.quantile is not None:
        return [float(args.quantile)]
    configured = cfg_get(cfg, "model", "quantiles", None)
    if configured is not None:
        return [float(q) for q in configured]
    return [float(cfg_get(cfg, "model", "quantile", 0.75))]


def _paired(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def cc(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2 or np.std(o) == 0 or np.std(s) == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def mse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean((s - o) ** 2)) if len(o) else np.nan


def rmse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else np.nan


def bias(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(s - o)) if len(o) else np.nan


def alpha_ratio(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    sig_o = np.std(o, ddof=0)
    if sig_o == 0:
        return np.nan
    return float(np.std(s, ddof=0) / sig_o)


def beta_ratio(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    mu_o = np.mean(o)
    if mu_o == 0:
        return np.nan
    return float(np.mean(s) / mu_o)


def kge(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    r = cc(o, s)
    alpha = alpha_ratio(o, s)
    beta = beta_ratio(o, s)
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def metric_row(node: int, lon: float, lat: float, product: str, obs, sim) -> Dict[str, Any]:
    n_paired = len(_paired(obs, sim)[0])
    return {
        "node": int(node),
        "lon": lon,
        "lat": lat,
        "n": n_paired,
        "Product": product,
        "KGE": kge(obs, sim),
        "CC": cc(obs, sim),
        "MSE": mse(obs, sim),
        "RMSE": rmse(obs, sim),
        "Bias": bias(obs, sim),
        "Alpha": alpha_ratio(obs, sim),
        "Beta": beta_ratio(obs, sim),
    }


def product_list_from_config(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    configured = cfg_get(cfg, "products", "columns", None)
    if configured is None:
        return list(DEFAULT_PRODUCTS)
    products = []
    for item in configured:
        if isinstance(item, dict):
            products.append((str(item["column"]), str(item.get("label", item["column"]))))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            products.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(
                "[products].columns entries must be {column=..., label=...} "
                f"or [column, label], got {item!r}"
            )
    return products


def resolve_products(cfg: Dict[str, Any], quantiles: List[float], include_point: bool) -> List[Tuple[str, str]]:
    products = []
    for col, label in product_list_from_config(cfg):
        if col == "pred_gsi" and not include_point:
            continue
        products.append((col, label))

    insert_after = str(cfg_get(cfg, "model", "insert_quantile_after", "idw_train"))
    configured_labels = cfg_get(cfg, "model", "quantile_labels", None)
    label_by_quantile = {}
    if isinstance(configured_labels, dict):
        label_by_quantile = {float(k): str(v) for k, v in configured_labels.items()}
    elif isinstance(configured_labels, list):
        for item in configured_labels:
            if isinstance(item, dict):
                label_by_quantile[float(item["quantile"])] = str(item["label"])

    q_items = [
        (
            quantile_col_name(q),
            label_by_quantile.get(q, f"GSI-{quantile_label(q)}"),
        )
        for q in quantiles
    ]

    out = []
    inserted = False
    for item in products:
        out.append(item)
        if item[0] == insert_after:
            out.extend(q_items)
            inserted = True
    if not inserted:
        out.extend(q_items)
    return out


def require_columns(df: pd.DataFrame, cols: List[str], csv_path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def main():
    args = parse_args()
    cfg = load_toml(args.config)

    eval_csv = Path(args.eval_csv or cfg_get(cfg, "paths", "eval_csv"))
    scenario = str(args.scenario or cfg_get(cfg, "meta", "scenario", eval_csv.stem))
    quantiles = parse_quantiles(args, cfg)
    include_point = (
        bool(cfg_get(cfg, "model", "include_point", True))
        if args.include_point is None
        else bool(args.include_point)
    )
    include_unavailable_days = (
        bool(cfg_get(cfg, "filters", "include_unavailable_days", True))
        if args.include_unavailable_days is None
        else bool(args.include_unavailable_days)
    )
    show_title = bool(cfg_get(cfg, "plot", "show_title", True))
    font_size = float(cfg_get(cfg, "plot", "font_size", 14))
    title_size = float(cfg_get(cfg, "plot", "title_size", 16))
    axis_label_size = float(cfg_get(cfg, "plot", "axis_label_size", 16))
    tick_label_size = float(cfg_get(cfg, "plot", "tick_label_size", 13))
    legend_size = float(cfg_get(cfg, "plot", "legend_size", 13))
    value_label_size = float(cfg_get(cfg, "plot", "value_label_size", 11))
    show_means = bool(cfg_get(cfg, "plot", "show_means", True))
    show_median_labels = bool(cfg_get(cfg, "plot", "show_median_labels", True))
    metrics_to_plot = [
        str(metric) for metric in cfg_get(cfg, "plot", "metrics", METRICS_TO_PLOT)
    ]
    unknown_metrics = sorted(set(metrics_to_plot) - set(METRICS_TO_PLOT))
    if unknown_metrics:
        raise ValueError(f"Unsupported plot metrics: {unknown_metrics}")
    common_samples = bool(cfg_get(cfg, "filters", "common_samples", False))
    date_start = cfg_get(cfg, "filters", "date_start", None)
    date_end = cfg_get(cfg, "filters", "date_end", None)

    q_tag = str(cfg_get(cfg, "model", "quantile_tag", "")).strip()
    if not q_tag:
        q_tag = "_".join(quantile_label(q) for q in quantiles) or "selected"
    out_dir = Path(
        args.out_dir
        or cfg_get(
            cfg,
            "paths",
            "out_dir",
            f"evaluation/plots/{scenario}/holdout_metrics_{q_tag}",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scenario:", scenario)
    print("Eval CSV:", eval_csv)
    print("Output dir:", out_dir)
    print("Quantiles:", quantiles)
    print("Quantile columns:", [quantile_col_name(q) for q in quantiles])
    print("Include point prediction:", include_point)
    print("Include unavailable days:", include_unavailable_days)

    df = pd.read_csv(eval_csv, parse_dates=["date"])
    if date_start is not None:
        df = df[df["date"] >= pd.Timestamp(date_start)].copy()
    if date_end is not None:
        df = df[df["date"] <= pd.Timestamp(date_end)].copy()
    if not include_unavailable_days and "model_available" in df.columns:
        before = len(df)
        df = df[df["model_available"].astype(bool)].copy()
        print(f"Filtered unavailable days: {before:,} -> {len(df):,} rows")

    products = resolve_products(cfg, quantiles=quantiles, include_point=include_point)
    require_columns(df, ["date", "node", "lon", "lat", "gauge", *[c for c, _ in products]], eval_csv)
    if common_samples:
        before = len(df)
        common_cols = ["gauge", *[col for col, _ in products]]
        finite = np.isfinite(df[common_cols].to_numpy(dtype=float)).all(axis=1)
        df = df.loc[finite].copy()
        print(f"Applied common finite-sample mask: {before:,} -> {len(df):,} rows")
        if df.empty:
            raise ValueError("The common finite-sample mask removed every row.")

    product_order = [label for _, label in products]
    colors_cfg = cfg.get("colors", {})
    product_colors = {
        label: str(colors_cfg.get(label, DEFAULT_COLORS.get(label, "#808080")))
        for label in product_order
    }

    print("Products:")
    for col, label in products:
        print(f"  {col} -> {label}")

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": axis_label_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "legend.fontsize": legend_size,
    })

    rows = []
    for (node, lon, lat), g in df.groupby(["node", "lon", "lat"], sort=False):
        obs = g["gauge"]
        for col, label in products:
            rows.append(metric_row(node, lon, lat, label, obs, g[col]))

    per_node = pd.DataFrame(rows)
    per_node["Product"] = pd.Categorical(per_node["Product"], categories=product_order, ordered=True)
    per_node = per_node.sort_values(["Product", "node"]).reset_index(drop=True)

    per_node_csv = out_dir / f"per_node_holdout_metrics_{q_tag}_{scenario}.csv"
    per_node.to_csv(per_node_csv, index=False)
    print("Saved:", per_node_csv)

    summary = (
        per_node
        .groupby("Product", observed=False, as_index=False)
        .agg(
            KGE_median=("KGE", "median"),
            KGE_mean=("KGE", "mean"),
            CC_median=("CC", "median"),
            CC_mean=("CC", "mean"),
            MSE_median=("MSE", "median"),
            MSE_mean=("MSE", "mean"),
            RMSE_median=("RMSE", "median"),
            RMSE_mean=("RMSE", "mean"),
            Bias_median=("Bias", "median"),
            Bias_mean=("Bias", "mean"),
            Alpha_median=("Alpha", "median"),
            Alpha_mean=("Alpha", "mean"),
            Beta_median=("Beta", "median"),
            Beta_mean=("Beta", "mean"),
            n_nodes=("KGE", "count"),
        )
    )
    summary["Product"] = pd.Categorical(summary["Product"], categories=product_order, ordered=True)
    summary = summary.sort_values("Product")

    summary_csv = out_dir / f"summary_holdout_metrics_{q_tag}_{scenario}.csv"
    summary.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)
    print("Summary:")
    print(summary)

    global_rows = []
    for col, label in products:
        global_rows.append({
            "Product": label,
            "KGE_global": kge(df["gauge"], df[col]),
            "CC_global": cc(df["gauge"], df[col]),
            "MSE_global": mse(df["gauge"], df[col]),
            "RMSE_global": rmse(df["gauge"], df[col]),
            "Bias_global": bias(df["gauge"], df[col]),
            "Alpha_global": alpha_ratio(df["gauge"], df[col]),
            "Beta_global": beta_ratio(df["gauge"], df[col]),
            "n_samples": len(_paired(df["gauge"], df[col])[0]),
        })
    global_metrics = pd.DataFrame(global_rows)
    global_metrics["Product"] = pd.Categorical(global_metrics["Product"], categories=product_order, ordered=True)
    global_metrics = global_metrics.sort_values("Product")

    global_csv = out_dir / f"global_holdout_metrics_{q_tag}_{scenario}.csv"
    global_metrics.to_csv(global_csv, index=False)
    print("Saved:", global_csv)
    print("Global metrics:")
    print(global_metrics)

    def annotate_boxplot_medians(ax, bp) -> None:
        if not show_median_labels:
            return
        for median in bp["medians"]:
            x = float(np.mean(median.get_xdata()))
            y = float(np.mean(median.get_ydata()))
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=value_label_size,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            )

    def plot_metric_boxplot(metric_name: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        data = [
            per_node.loc[per_node["Product"] == prod, metric_name].dropna().values
            for prod in product_order
        ]
        colors = [product_colors[prod] for prod in product_order]
        bp = ax.boxplot(
            data,
            tick_labels=product_order,
            patch_artist=True,
            showfliers=True,
            showmeans=show_means,
            whis=1.5,
            medianprops=dict(color="black", linewidth=2.0),
            meanprops=dict(
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=7,
            ),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(
                marker="o",
                markersize=3.5,
                markerfacecolor="none",
                markeredgecolor="gray",
                alpha=0.8,
            ),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.85)
        annotate_boxplot_medians(ax, bp)

        if show_title:
            ax.set_title(f"Node-Level {metric_name} at Held-Out Gauge Locations", weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_AXIS_LABELS.get(metric_name, metric_name))
        if METRIC_YLIM.get(metric_name) is not None:
            ax.set_ylim(*METRIC_YLIM[metric_name])
        if metric_name == "Bias":
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        if metric_name in {"Alpha", "Beta"}:
            ax.axhline(1, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        out = out_dir / f"{metric_name}_boxplot_holdout_nodes_{q_tag}_{scenario}.png"
        fig.savefig(out, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)

    def plot_global_barplot(metric_name: str) -> None:
        col = f"{metric_name}_global"
        fig, ax = plt.subplots(figsize=(10, 5.8))
        vals = [
            global_metrics.loc[global_metrics["Product"] == prod, col].iloc[0]
            for prod in product_order
        ]
        bars = ax.bar(
            product_order,
            vals,
            color=[product_colors[prod] for prod in product_order],
            edgecolor="black",
            linewidth=1.0,
        )
        finite_vals = [abs(v) for v in vals if np.isfinite(v)]
        offset = 0.02 * max(finite_vals + [1.0])
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_label_size,
                    fontweight="bold",
                )
        if show_title:
            ax.set_title(f"Global {metric_name} at Held-Out Gauge Locations", weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_AXIS_LABELS.get(metric_name, metric_name))
        if METRIC_YLIM.get(metric_name) is not None:
            ax.set_ylim(*METRIC_YLIM[metric_name])
        if metric_name == "Bias":
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        if metric_name in {"Alpha", "Beta"}:
            ax.axhline(1, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        out = out_dir / f"global_{metric_name}_barplot_holdout_{q_tag}_{scenario}.png"
        fig.savefig(out, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)

    def style_metric_axis(ax, metric_name: str) -> None:
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_AXIS_LABELS.get(metric_name, metric_name))
        if METRIC_YLIM.get(metric_name) is not None:
            ax.set_ylim(*METRIC_YLIM[metric_name])
        if metric_name in {"Alpha", "Beta"}:
            ax.axhline(1, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=25)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    component_metrics = ["KGE", "CC", "Alpha", "Beta"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    for ax, metric_name in zip(axes.flat, component_metrics):
        data = [
            per_node.loc[per_node["Product"] == prod, metric_name].dropna().values
            for prod in product_order
        ]
        bp = ax.boxplot(
            data,
            tick_labels=product_order,
            patch_artist=True,
            showfliers=True,
            showmeans=show_means,
            whis=1.5,
            medianprops=dict(color="black", linewidth=2.0),
            meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black",
                           markeredgewidth=1.2, markersize=7),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(marker="o", markersize=3.5, markerfacecolor="none",
                            markeredgecolor="gray", alpha=0.8),
        )
        for box, prod in zip(bp["boxes"], product_order):
            box.set_facecolor(product_colors[prod])
            box.set_edgecolor("black")
            box.set_alpha(0.85)
        annotate_boxplot_medians(ax, bp)
        style_metric_axis(ax, metric_name)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    out = out_dir / f"KGE_components_2x2_boxplots_holdout_nodes_{q_tag}_{scenario}.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, metric_name in zip(axes.flat, component_metrics):
        col = f"{metric_name}_global"
        vals = [
            global_metrics.loc[global_metrics["Product"] == prod, col].iloc[0]
            for prod in product_order
        ]
        bars = ax.bar(
            product_order,
            vals,
            color=[product_colors[prod] for prod in product_order],
            edgecolor="black",
            linewidth=1.0,
        )
        finite_vals = [abs(v) for v in vals if np.isfinite(v)]
        offset = 0.02 * max(finite_vals + [1.0])
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=value_label_size,
                    fontweight="bold",
                )
        style_metric_axis(ax, metric_name)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    out = out_dir / f"global_KGE_components_2x2_barplots_holdout_{q_tag}_{scenario}.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

    for metric in metrics_to_plot:
        plot_metric_boxplot(metric)
        plot_global_barplot(metric)

    print("All outputs saved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
