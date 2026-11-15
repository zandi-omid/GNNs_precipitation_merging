#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import pickle
import random

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stack_gsi_predictions import (
    _check_no_overlap,
    _date_years,
    _merge_auxiliary_features,
    _paired_frame,
    _read_meta_train_csvs,
    cfg_get,
    metric_row,
    per_node_metrics,
)


def load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train/apply a PyTorch level-2 MLP stacker from leakage-free GSI OOF CSVs."
    )
    p.add_argument("--config", required=True)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_quantiles(n: int, device=None, dtype=None) -> torch.Tensor:
    if n < 1:
        raise ValueError(f"n_quantiles must be >= 1, got {n}")
    return torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)


class MetaMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RowQuantileLoss(nn.Module):
    def __init__(
        self,
        n_quantiles: int = 32,
        quantiles: List[float] | None = None,
        crossing_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if quantiles is None:
            q = make_quantiles(int(n_quantiles), dtype=torch.float32)
        else:
            q = torch.tensor(list(quantiles), dtype=torch.float32)
            if torch.any(q <= 0.0) or torch.any(q >= 1.0):
                raise ValueError(f"All quantiles must be in (0,1), got {quantiles}")
            q, _ = torch.sort(q)
        self.register_buffer("quantiles", q)
        self.crossing_weight = float(crossing_weight)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.ndim != 2:
            raise ValueError(f"Quantile meta loss expects preds [B,Q], got {tuple(preds.shape)}")
        if target.ndim != 1:
            raise ValueError(f"Quantile meta loss expects target [B], got {tuple(target.shape)}")
        qs = self.quantiles.to(device=preds.device, dtype=preds.dtype).view(1, -1)
        err = target.unsqueeze(-1) - preds
        loss = torch.maximum(qs * err, (qs - 1.0) * err).mean()
        if self.crossing_weight > 0.0 and preds.shape[1] > 1:
            crossing = torch.relu(preds[:, :-1] - preds[:, 1:]).mean()
            loss = loss + self.crossing_weight * crossing
        return loss


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        mean = np.zeros(x_train.shape[1], dtype=np.float32)
        std = np.ones(x_train.shape[1], dtype=np.float32)
        return x_train, x_test, mean, std
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def _standardize_target(
    y_train: np.ndarray,
    enabled: bool,
) -> Tuple[np.ndarray, float, float]:
    if not enabled:
        return y_train.astype(np.float32), 0.0, 1.0
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std < 1e-6:
        std = 1.0
    return ((y_train - mean) / std).astype(np.float32), mean, std


def _predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred


def _q_col(prefix: str, q: float) -> str:
    return f"{prefix}_q{int(round(float(q) * 100)):02d}"


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    seed = int(cfg_get(cfg, "meta_model", "random_state", 42))
    seed_everything(seed)

    final_eval_csv = Path(cfg_get(cfg, "paths", "final_eval_csv"))
    out_csv = Path(cfg_get(cfg, "paths", "out_csv"))
    model_out = Path(cfg_get(cfg, "paths", "model_out", out_csv.with_suffix(".pt")))
    metrics_out = Path(
        cfg_get(cfg, "paths", "metrics_out", out_csv.with_name(out_csv.stem + "_global_metrics.csv"))
    )
    per_node_out = Path(
        cfg_get(cfg, "paths", "per_node_metrics_out", out_csv.with_name(out_csv.stem + "_per_node_metrics.csv"))
    )

    target_col = str(cfg_get(cfg, "stacking", "target_col", "gauge"))
    feature_cols = [str(c) for c in cfg_get(cfg, "stacking", "feature_cols", [])]
    output_col = str(cfg_get(cfg, "stacking", "output_col", "stack_meta_mlp"))
    output_label = str(cfg_get(cfg, "stacking", "output_label", "StackCV-MetaMLP"))
    clip_min = cfg_get(cfg, "stacking", "clip_min", 0.0)
    clip_max = cfg_get(cfg, "stacking", "clip_max", None)
    passthrough_cols = [str(c) for c in cfg_get(cfg, "stacking", "comparison_cols", [])]

    if not feature_cols:
        raise ValueError("[stacking].feature_cols must list base-model prediction columns.")

    print("Final-evaluation CSV:", final_eval_csv)
    print("Base feature columns:", feature_cols)
    print("Target column:", target_col)

    train_df_raw = _read_meta_train_csvs(cfg)
    test_df_raw = pd.read_csv(final_eval_csv)
    _check_no_overlap(train_df_raw, test_df_raw)

    train_df_raw, feature_cols = _merge_auxiliary_features(train_df_raw, cfg, feature_cols)
    test_df_raw, feature_cols = _merge_auxiliary_features(test_df_raw, cfg, feature_cols)
    print("Final feature columns:", feature_cols)

    train_years = _date_years(train_df_raw)
    test_years = _date_years(test_df_raw)
    print(f"Meta-training years: {train_years[0]}-{train_years[1]}")
    print(f"Final-evaluation years: {test_years[0]}-{test_years[1]}")

    train_df = _paired_frame(train_df_raw, [target_col, *feature_cols])
    test_df = _paired_frame(test_df_raw, [target_col, *feature_cols])
    if len(train_df) == 0:
        raise RuntimeError("No finite meta-training rows after filtering.")
    if len(test_df) == 0:
        raise RuntimeError("No finite final-evaluation rows after filtering.")

    x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_raw = train_df[target_col].to_numpy(dtype=np.float32)
    x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)

    standardize_x = bool(cfg_get(cfg, "meta_model", "standardize_features", True))
    standardize_y = bool(cfg_get(cfg, "meta_model", "standardize_target", True))
    x_train, x_test, x_mean, x_std = _standardize_train_test(
        x_train_raw,
        x_test_raw,
        enabled=standardize_x,
    )
    y_train, y_mean, y_std = _standardize_target(y_train_raw, enabled=standardize_y)

    hidden_dims = [int(v) for v in cfg_get(cfg, "meta_model", "hidden_dims", [32, 16])]
    dropout = float(cfg_get(cfg, "meta_model", "dropout", 0.0))
    lr = float(cfg_get(cfg, "meta_model", "lr", 3e-4))
    weight_decay = float(cfg_get(cfg, "meta_model", "weight_decay", 1e-4))
    max_epochs = int(cfg_get(cfg, "meta_model", "max_epochs", 200))
    batch_size = int(cfg_get(cfg, "meta_model", "batch_size", 8192))
    num_workers = int(cfg_get(cfg, "meta_model", "num_workers", 0))
    log_every_n_epochs = int(cfg_get(cfg, "meta_model", "log_every_n_epochs", 10))
    loss_name = str(cfg_get(cfg, "meta_model", "loss_name", "mse")).lower()
    loss_n_quantiles = int(cfg_get(cfg, "meta_model", "loss_n_quantiles", 32))
    loss_quantiles = cfg_get(cfg, "meta_model", "loss_quantiles", None)
    loss_crossing_weight = float(cfg_get(cfg, "meta_model", "loss_crossing_weight", 0.0))
    quantile_output_values = [
        float(v) for v in cfg_get(cfg, "quantile_output", "quantiles", [0.65, 0.70, 0.75])
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Rows used for meta training:", len(train_df))
    print("Rows used for final stack prediction:", len(test_df))
    print("Hidden dims:", hidden_dims)

    model = MetaMLP(
        in_dim=len(feature_cols),
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_dim=loss_n_quantiles if loss_name == "quantile" else 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_name == "quantile":
        loss_fn = RowQuantileLoss(
            n_quantiles=loss_n_quantiles,
            quantiles=loss_quantiles,
            crossing_weight=loss_crossing_weight,
        )
        quantiles = loss_fn.quantiles.detach().cpu().numpy().astype(np.float32)
        point_idx = int(np.argmin(np.abs(quantiles - 0.5)))
        print("Loss: quantile")
        print("Quantiles:", quantiles.tolist())
        print("Point output uses nearest median index:", point_idx, float(quantiles[point_idx]))
    elif loss_name == "mse":
        loss_fn = nn.MSELoss()
        quantiles = None
        point_idx = None
        print("Loss: mse")
    else:
        raise ValueError(f"Unsupported [meta_model].loss_name={loss_name!r}; use 'mse' or 'quantile'.")

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if loss_name == "mse":
                pred = pred.squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_n += len(xb)
        if epoch == 1 or epoch == max_epochs or epoch % log_every_n_epochs == 0:
            print(f"Epoch {epoch:04d}/{max_epochs} train/mse={total_loss / max(total_n, 1):.6f}")

    pred_scaled = _predict(model, x_test, device=device, batch_size=batch_size)
    pred_all = (pred_scaled * y_std + y_mean).astype(np.float32)
    if loss_name == "quantile":
        pred = pred_all[:, point_idx].astype(np.float32)
    else:
        pred = pred_all.astype(np.float32)

    if clip_min is not None or clip_max is not None:
        lo = None if clip_min is None else float(clip_min)
        hi = None if clip_max is None else float(clip_max)
        pred = np.clip(pred, lo, hi)
        if loss_name == "quantile":
            pred_all = np.clip(pred_all, lo, hi)

    out = test_df_raw.copy()
    out[output_col] = np.nan
    out.loc[test_df.index, output_col] = pred

    quantile_product_cols: List[Tuple[str, str]] = []
    if loss_name == "quantile":
        for q in quantile_output_values:
            idx = int(np.argmin(np.abs(quantiles - q)))
            actual_q = float(quantiles[idx])
            col = _q_col(output_col, q)
            out[col] = np.nan
            out.loc[test_df.index, col] = pred_all[:, idx]
            label = _q_col(output_label, q)
            quantile_product_cols.append((col, label))
            print(
                f"Quantile output requested {q:.3f} -> nearest {actual_q:.3f} -> column {col}"
            )

    scenario = str(cfg_get(cfg, "meta", "scenario", "stacked_gsi_meta_mlp"))
    out["scenario"] = scenario

    products = []
    for col in passthrough_cols:
        if col in out.columns:
            products.append((col, col))
    products.append((output_col, output_label))
    products.extend(quantile_product_cols)

    global_metrics = pd.DataFrame([
        metric_row(label, out[target_col], out[col])
        for col, label in products
    ])
    node_metrics = per_node_metrics(out, products)
    summary = (
        node_metrics
        .groupby("Product", as_index=False)
        .agg(
            KGE_median=("KGE", "median"),
            KGE_mean=("KGE", "mean"),
            CC_median=("CC", "median"),
            CC_mean=("CC", "mean"),
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    per_node_out.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv, index=False)
    global_metrics.to_csv(metrics_out, index=False)
    node_metrics.to_csv(per_node_out, index=False)
    summary.to_csv(per_node_out.with_name(per_node_out.stem + "_summary.csv"), index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "target_col": target_col,
            "output_col": output_col,
            "scenario": scenario,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "loss_name": loss_name,
            "quantiles": quantiles,
            "point_idx": point_idx,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "standardize_features": standardize_x,
            "standardize_target": standardize_y,
        },
        model_out,
    )
    with open(model_out.with_suffix(".pkl"), "wb") as f:
        pickle.dump(
            {
                "feature_cols": feature_cols,
                "target_col": target_col,
                "output_col": output_col,
                "scenario": scenario,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "loss_name": loss_name,
                "quantiles": quantiles,
                "point_idx": point_idx,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "standardize_features": standardize_x,
                "standardize_target": standardize_y,
            },
            f,
        )

    print("Saved stacked CSV:", out_csv)
    print("Saved meta-MLP checkpoint:", model_out)
    print("Saved global metrics:", metrics_out)
    print(global_metrics.sort_values("KGE_global", ascending=False))
    print("Saved per-node metrics:", per_node_out)
    print("Station summary:")
    print(summary.sort_values("KGE_median", ascending=False))


if __name__ == "__main__":
    main()
