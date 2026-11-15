#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "evaluation" / "plots" / "all_gauges_temporal_ablation_training"

RUNS = {
    "GATGCN": {
        "versions": [
            ROOT / "logs" / "ALLGAUGE_crossfit10_GATGCN_currentDay_encodedLateFusion_4head_Q32_train2005_2019" / "version_0",
        ],
        "parameters": 61_837,
        "training_seconds": 1 * 3600 + 3 * 60 + 11,
        "color": "#0072B2",
    },
    "TGATGCN": {
        "versions": [
            ROOT / "logs" / "ALLGAUGE_crossfit10_TGATGCN_T7_encodedLateFusion_4head_Q32_train2005_2019" / "version_0",
            ROOT / "logs" / "ALLGAUGE_crossfit10_TGATGCN_T7_encodedLateFusion_4head_Q32_train2005_2019" / "version_1",
        ],
        "parameters": 456_589,
        "training_seconds": 4 * 3600 + 18 + 2 * 3600 + 52 * 60 + 7,
        "color": "#D55E00",
    },
    "A3TGATGCN": {
        "versions": [
            ROOT / "logs" / "ALLGAUGE_crossfit10_TGATGCN_T7_temporalAttention_encodedLateFusion_4head_Q32_train2005_2019" / "version_0",
            ROOT / "logs" / "ALLGAUGE_crossfit10_TGATGCN_T7_temporalAttention_encodedLateFusion_4head_Q32_train2005_2019" / "version_1",
        ],
        "parameters": 464_846,
        "training_seconds": 4 * 3600 + 18 + 3 * 3600 + 2 * 60 + 33,
        "color": "#009E73",
    },
}


def read_version(version_dir: Path) -> pd.DataFrame:
    accumulator = EventAccumulator(str(version_dir))
    accumulator.Reload()
    rows = {}
    for tag, column in (("train/loss", "train_loss"), ("val/loss", "val_loss")):
        values = accumulator.Scalars(tag)
        epochs = accumulator.Scalars("epoch")
        epoch_by_step = {item.step: int(round(item.value)) for item in epochs}
        for item in values:
            epoch = epoch_by_step[item.step]
            rows.setdefault(epoch, {"epoch": epoch})[column] = float(item.value)
    return pd.DataFrame(rows.values())


def elapsed_text(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_frames = []
    summary_rows = []

    for model, settings in RUNS.items():
        frames = [read_version(path) for path in settings["versions"]]
        curve = (
            pd.concat(frames, ignore_index=True)
            .sort_values("epoch")
            .drop_duplicates("epoch", keep="last")
            .reset_index(drop=True)
        )
        if curve["epoch"].tolist() != list(range(120)):
            raise ValueError(f"{model} does not contain exactly epochs 0-119")
        curve.insert(0, "model", model)
        curve_frames.append(curve)

        best_idx = curve["val_loss"].idxmin()
        summary_rows.append(
            {
                "model": model,
                "trainable_parameters": settings["parameters"],
                "parameter_ratio_vs_gatgcn": settings["parameters"] / RUNS["GATGCN"]["parameters"],
                "best_epoch": int(curve.loc[best_idx, "epoch"]) + 1,
                "best_val_loss": curve.loc[best_idx, "val_loss"],
                "final_train_loss": curve.iloc[-1]["train_loss"],
                "final_val_loss": curve.iloc[-1]["val_loss"],
                "active_training_seconds": settings["training_seconds"],
                "active_training_time_hms": elapsed_text(settings["training_seconds"]),
                "training_time_ratio_vs_gatgcn": settings["training_seconds"] / RUNS["GATGCN"]["training_seconds"],
            }
        )

    curves = pd.concat(curve_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    curves.to_csv(OUT_DIR / "temporal_ablation_epoch_losses.csv", index=False)
    summary.to_csv(OUT_DIR / "temporal_ablation_model_summary.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    panels = (("train_loss", "(a) Training"), ("val_loss", "(b) Validation"))
    for ax, (column, panel_label) in zip(axes, panels):
        for model, settings in RUNS.items():
            frame = curves[curves["model"] == model]
            ax.plot(
                frame["epoch"] + 1,
                frame[column],
                color=settings["color"],
                linewidth=1.5,
                label=model,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Quantile loss")
        ax.set_xlim(1, 120)
        ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.text(0.03, 0.95, panel_label, transform=ax.transAxes, va="top", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "temporal_ablation_training_validation_loss.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT_DIR / "temporal_ablation_training_validation_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
