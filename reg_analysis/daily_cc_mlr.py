#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from pathlib import Path

SEQ_DIR = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T030_all_years")
OUT_CSV = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/reg_analysis/daily_cc_mlr.csv")

def pearson_cc(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    aa = a[ok] - np.mean(a[ok])
    bb = b[ok] - np.mean(b[ok])
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2)) + 1e-12
    return float(np.sum(aa * bb) / denom)

def fit_mlr(x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
    """
    Fit y = a*x1 + b*x2 + c using least squares on finite pairs.
    Returns a,b,c and yhat
    """
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan, np.full_like(y, np.nan, dtype=np.float64)

    X = np.column_stack([x1[ok], x2[ok], np.ones(ok.sum(), dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(X, y[ok].astype(np.float64), rcond=None)
    a, b, c = beta
    yhat = a * x1 + b * x2 + c
    return float(a), float(b), float(c), yhat

def main():
    files = sorted(SEQ_DIR.glob("seq_*.pt"))
    if not files:
        raise FileNotFoundError(f"No seq_*.pt found in {SEQ_DIR}")

    rows = []
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=True)
        date = str(d["date"])

        x = d["x"].numpy().astype(np.float64)   # [T,N,2]
        y = d["y"].numpy().astype(np.float64)   # [N] with NaNs
        m = d["y_mask"].numpy().astype(bool)

        # Use only gauge nodes for THIS day
        if m.sum() < 10:
            continue

        era = x[-1, :, 0][m]
        im  = x[-1, :, 1][m]
        obs = y[m]

        cc_era = pearson_cc(era, obs)
        cc_im  = pearson_cc(im, obs)

        a, b, c, yhat = fit_mlr(era, im, obs)
        cc_mlr = pearson_cc(yhat, obs)

        rows.append({
            "date": date,
            "n_gauges": int(m.sum()),
            "cc_era5": cc_era,
            "cc_imerg": cc_im,
            "cc_mlr": cc_mlr,
            "a": a, "b": b, "c": c,
            "cc_gain_vs_era5": cc_mlr - cc_era,
            "cc_gain_vs_best": cc_mlr - np.nanmax([cc_era, cc_im]),
        })

    df = pd.DataFrame(rows).sort_values("date")

    # Round numeric columns to 1 decimal
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # quick summary
    print("Saved:", OUT_CSV)
    print("Days:", len(df))
    print("Mean CC ERA5 :", df["cc_era5"].mean())
    print("Mean CC IMERG:", df["cc_imerg"].mean())
    print("Mean CC MLR  :", df["cc_mlr"].mean())
    print("Frac MLR > ERA5 :", np.mean(df["cc_mlr"] > df["cc_era5"]))
    print("Frac MLR > best :", np.mean(df["cc_mlr"] > df[["cc_era5","cc_imerg"]].max(axis=1)))
    print("Median gain vs best:", df["cc_gain_vs_best"].median())

if __name__ == "__main__":
    main()