#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

PRODUCTS = {
    "Stacked-q70": "stack_temporal_oof_meta_mlp_q70",
    "IDW": "idw_train",
    "PRISM": "prism",
    "MSWEP": "mswep",
    "GMCP": "gmcp",
    "MRMS": "mrms",
}

COLORS = {
    "Stacked-q70": "#E9C46A",
    "IDW": "#A98467",
    "PRISM": "#8A5A9E",
    "MSWEP": "#36A195",
    "GMCP": "#929292",
    "MRMS": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot pooled global KGE by meteorological season."
    )
    parser.add_argument("--evaluation-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--date-start", default="2020-01-01")
    parser.add_argument("--date-end", default="2024-12-31")
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def kge_components(observed: np.ndarray, estimated: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(observed) & np.isfinite(estimated)
    observed = observed[finite]
    estimated = estimated[finite]
    if len(observed) < 2:
        return {"KGE": np.nan, "CC": np.nan, "Alpha": np.nan, "Beta": np.nan, "n": len(observed)}

    cc = float(np.corrcoef(observed, estimated)[0, 1])
    alpha = float(np.std(estimated, ddof=1) / np.std(observed, ddof=1))
    beta = float(np.mean(estimated) / np.mean(observed))
    kge = float(1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {"KGE": kge, "CC": cc, "Alpha": alpha, "Beta": beta, "n": len(observed)}


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.evaluation_csv, parse_dates=["date"])
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(args.date_start), pd.Timestamp(args.date_end))
    ].copy()

    missing = {"gauge", *PRODUCTS.values()} - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")

    rows = []
    for season, months in SEASONS.items():
        seasonal = frame.loc[frame["date"].dt.month.isin(months)]
        for product, column in PRODUCTS.items():
            metrics = kge_components(
                seasonal["gauge"].to_numpy(float), seasonal[column].to_numpy(float)
            )
            rows.append({"Season": season, "Product": product, **metrics})

    results = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "seasonal_global_KGE_components_available_coverage.csv"
    results.to_csv(csv_path, index=False, float_format="%.8f")

    fig, axis = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
    x = np.arange(len(SEASONS))
    for product in PRODUCTS:
        values = (
            results.loc[results["Product"] == product]
            .set_index("Season")
            .reindex(SEASONS)["KGE"]
            .to_numpy()
        )
        axis.plot(
            x,
            values,
            label=product,
            color=COLORS[product],
            marker="o",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=2.6,
        )

    axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
    axis.set_xticks(x, SEASONS)
    axis.set_xlim(-0.15, len(SEASONS) - 0.85)
    axis.set_ylim(-0.05, 0.85)
    axis.set_xlabel("Meteorological season", fontsize=17)
    axis.set_ylabel("Global Kling-Gupta efficiency (KGE)", fontsize=17)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    axis.legend(ncol=3, frameon=False, fontsize=13, loc="lower center")

    figure_path = args.out_dir / "seasonal_global_KGE_lines_available_coverage.png"
    fig.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(results.to_string(index=False))
    print("Saved:", csv_path)
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
