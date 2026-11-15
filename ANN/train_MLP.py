#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import date as date_cls, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import tomli as tomllib


RUN_NAME = "MLP_100pct_train2005_2018_val2019"
BASE_DIR = str(Path(__file__).resolve().parent)
GNN_DIR = Path(__file__).resolve().parents[1] / "GSI"
sys.path.insert(0, str(GNN_DIR))

from losses.quantile import QuantileLoss


# ============================================================
# TOML helpers
# ============================================================
def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for section, kv in cfg.items():
        if not isinstance(kv, dict):
            out[section] = kv
            continue

        if section == "meta":
            if "run_name" in kv:
                out["run_name"] = kv["run_name"]
            continue

        for k, v in kv.items():
            out[f"{section}_{k}"] = v
    return out


def _as_year_tuple(v, default: Tuple[int, int]) -> Tuple[int, int]:
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"Expected [y0, y1] for years, got: {v}")


def _as_year_ranges(v) -> Tuple[Tuple[int, int], ...]:
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return ()
        if len(v) == 2 and not isinstance(v[0], (list, tuple)):
            return (_as_year_tuple(v, (0, 0)),)
        return tuple(_as_year_tuple(item, (0, 0)) for item in v)
    raise ValueError(f"Expected [[y0, y1], ...], got: {v}")


def _cli_overrides(argv) -> set[str]:
    out = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].replace("-", "_")
        out.add(key)
    return out


def _parse_iso_date(value: str) -> date_cls:
    try:
        return date_cls.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Expected date in YYYY-MM-DD format, got: {value!r}") from exc


def _date_range(start: str, end: str) -> list[str]:
    d0 = _parse_iso_date(start)
    d1 = _parse_iso_date(end)
    if d1 < d0:
        raise ValueError(f"Date range end {end} is before start {start}")
    out = []
    d = d0
    while d <= d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _compact_date(day: str) -> str:
    return str(day).replace("-", "")


def _worker_slice(items: list[str], worker_index: int, num_workers: int) -> list[str]:
    num_workers = int(num_workers)
    worker_index = int(worker_index)
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(
            f"worker_index must be in [0, {num_workers - 1}], got {worker_index}"
        )
    start = len(items) * worker_index // num_workers
    end = len(items) * (worker_index + 1) // num_workers
    return items[start:end]


def _slurm_procid() -> int:
    return int(os.environ.get("SLURM_PROCID", "0"))


def _is_progress_worker(progress_worker_index: int) -> bool:
    return _slurm_procid() == int(progress_worker_index)


def _per_day_run_name(base_run_name: str, day: str, multi_day: bool) -> str:
    if not multi_day:
        return base_run_name
    return f"{base_run_name}/{_compact_date(day)}"


def _save_best_checkpoint_alias(trainer: Trainer, ckpt_root: Path) -> None:
    best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    if not best_path:
        return
    best_src = Path(best_path)
    if not best_src.exists():
        return
    best_dst = ckpt_root / "best.ckpt"
    if best_src.resolve() != best_dst.resolve():
        shutil.copy2(best_src, best_dst)
    rank_zero_info(f"Saved best checkpoint alias: {best_dst}")


# ============================================================
# Dataset
# ============================================================
class TabularPrecipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str):
        self.x = df[feature_cols].to_numpy(dtype=np.float32)
        self.y = df[target_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


# ============================================================
# DataModule
# ============================================================
class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        feature_cols,
        target_col: str = "gauge",
        batch_size: int = 4096,
        num_workers: int = 4,
        train_years: Tuple[int, int] = (2005, 2018),
        val_years: Tuple[int, int] = (2019, 2019),
        test_years: Tuple[int, int] = (2020, 2024),
        exclude_train_years: Optional[Tuple[Tuple[int, int], ...]] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.exclude_train_years = tuple(exclude_train_years or ())

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.feature_means = None
        self.feature_stds = None

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        df["year"] = df["date"].dt.year

        cols_needed = list(self.feature_cols) + [self.target_col]
        df = df.dropna(subset=cols_needed).copy()

        train_mask = df["year"].between(*self.train_years)
        for lo, hi in self.exclude_train_years:
            train_mask &= ~df["year"].between(lo, hi)
        val_mask = df["year"].between(*self.val_years)
        test_mask = df["year"].between(*self.test_years)

        df_train = df.loc[train_mask].copy()
        df_val = df.loc[val_mask].copy()
        df_test = df.loc[test_mask].copy()

        if len(df_train) == 0:
            raise RuntimeError("No training rows found.")
        if len(df_val) == 0:
            raise RuntimeError("No validation rows found.")

        self.feature_means = df_train[self.feature_cols].mean()
        self.feature_stds = df_train[self.feature_cols].std().replace(0, 1.0)

        for dfx in [df_train, df_val, df_test]:
            dfx.loc[:, self.feature_cols] = (
                dfx[self.feature_cols] - self.feature_means
            ) / self.feature_stds

        self.train_ds = TabularPrecipDataset(df_train, self.feature_cols, self.target_col)
        self.val_ds = TabularPrecipDataset(df_val, self.feature_cols, self.target_col)
        self.test_ds = TabularPrecipDataset(df_test, self.feature_cols, self.target_col)

        rank_zero_info(f"Train rows: {len(df_train):,}")
        rank_zero_info(f"Val rows:   {len(df_val):,}")
        rank_zero_info(f"Test rows:  {len(df_test):,}")
        if self.exclude_train_years:
            rank_zero_info(f"Excluded train year ranges: {self.exclude_train_years}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


def _normalize_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1.0).fillna(1.0)

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_train.loc[:, feature_cols] = (df_train[feature_cols] - means) / stds
    df_val.loc[:, feature_cols] = (df_val[feature_cols] - means) / stds
    return df_train, df_val, means, stds


def _make_loaders_from_dataframes(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TabularPrecipDataset(df_train, feature_cols, target_col)
    val_ds = TabularPrecipDataset(df_val, feature_cols, target_col)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def _split_day_dataframe(
    df_day: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_day) < 2:
        raise ValueError("Need at least two rows for a per-day train/val split.")
    val_fraction = float(val_fraction)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(df_day))
    n_val = int(round(val_fraction * len(df_day)))
    n_val = min(max(1, n_val), len(df_day) - 1)
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return df_day.iloc[train_idx].copy(), df_day.iloc[val_idx].copy()


def _resolve_per_day_dates(args, available_dates: list[str]) -> list[str]:
    available = set(available_dates)
    if args.training_task_dates:
        raw_dates = [str(d) for d in args.training_task_dates]
        if len(raw_dates) == 2:
            requested = _date_range(raw_dates[0], raw_dates[1])
            dates = [d for d in requested if d in available]
            missing = len(requested) - len(dates)
            if missing:
                rank_zero_info(
                    "[MLP_PER_DAY] "
                    f"{missing} requested dates have no dataframe rows and will be skipped. "
                    f"Available date range is {available_dates[0]} to {available_dates[-1]}."
                )
        else:
            dates = raw_dates
    elif args.training_task_date:
        dates = [str(args.training_task_date)]
    else:
        raise ValueError(
            "For per_day mode, provide [training_task].date or "
            "[training_task].dates = [start, end]."
        )
    missing = [d for d in dates if d not in available]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} requested per-day dates have no dataframe rows. "
            f"First missing dates: {preview}"
        )
    if not dates:
        raise FileNotFoundError("No requested per-day dates have dataframe rows.")
    return dates


def _load_sequence_train_precip_means(seq_dir: str, requested_dates: list[str]) -> dict[str, float]:
    seq_path = Path(seq_dir)
    if not seq_path.exists():
        raise FileNotFoundError(f"training_task.precip_seq_dir does not exist: {seq_path}")
    requested = set(str(d) for d in requested_dates)
    out: dict[str, float] = {}
    for fp in sorted(seq_path.glob("seq_*.pt")):
        sample = torch.load(fp, map_location="cpu", weights_only=True)
        day = str(sample.get("date", ""))
        if day not in requested:
            continue
        y = sample["y"].float()
        mask = sample["train_available_mask"].bool() & torch.isfinite(y)
        if int(mask.sum()) == 0:
            out[day] = 0.0
        else:
            out[day] = float(y[mask].mean().item())
        if len(out) == len(requested):
            break
    missing = sorted(requested.difference(out))
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Could not compute train precipitation mean for {len(missing)} dates. "
            f"First missing dates: {preview}"
        )
    return out


# ============================================================
# Model
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dims=(128, 128, 64),
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out


# ============================================================
# Lightning module
# ============================================================
class MLPLightning(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dims=(128, 128, 64),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_name: str = "CosineAnnealingLR",
        scheduler_t_max: int = 50,
        scheduler_eta_min: float = 0.0,
        loss_name: str = "mse",
        loss_n_quantiles: int = 32,
        loss_quantiles: Optional[list[float]] = None,
        loss_quantile_weights: Optional[list[float]] = None,
        loss_higher_q_weight: Optional[float] = None,
        loss_crossing_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_name = (loss_name or "mse").lower()
        self.n_quantiles = (
            len(loss_quantiles)
            if self.loss_name == "quantile" and loss_quantiles is not None
            else int(loss_n_quantiles)
        )
        out_dim = self.n_quantiles if self.loss_name == "quantile" else 1

        self.model = MLPRegressor(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            out_dim=out_dim,
        )

        if self.loss_name == "quantile":
            self.loss_fn = QuantileLoss(
                n_quantiles=self.n_quantiles,
                quantiles=loss_quantiles,
                quantile_weights=loss_quantile_weights,
                higher_q_weight=loss_higher_q_weight,
                crossing_weight=float(loss_crossing_weight),
            )
            self.register_buffer("quantiles", self.loss_fn.quantiles.detach().clone())
            self.q50_idx = int(torch.argmin(torch.abs(self.quantiles - 0.5)).item())
            rank_zero_info(
                f"[QUANTILE SETUP] quantiles={self.quantiles.detach().cpu().tolist()} "
                f"q50_idx={self.q50_idx}"
            )
        elif self.loss_name == "mse":
            self.loss_fn = nn.MSELoss()
            self.quantiles = None
            self.q50_idx = None
        else:
            raise ValueError(f"Unknown loss_name: {loss_name}")

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.scheduler_name = scheduler_name
        self.scheduler_t_max = int(scheduler_t_max)
        self.scheduler_eta_min = float(scheduler_eta_min)

    def forward(self, x):
        return self.model(x)

    def _loss(self, y_hat, y):
        if self.loss_name == "quantile":
            return self.loss_fn(
                y_hat.unsqueeze(1),
                y.unsqueeze(1),
                torch.isfinite(y).unsqueeze(1),
            )
        return self.loss_fn(y_hat, y)

    def _point_prediction(self, y_hat):
        if self.loss_name == "quantile":
            return y_hat[:, self.q50_idx]
        return y_hat

    @staticmethod
    def _metrics(y_hat, y):
        diff = y_hat - y
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        bias = torch.mean(diff)

        y_hat_c = y_hat - y_hat.mean()
        y_c = y - y.mean()
        denom = torch.sqrt(torch.sum(y_hat_c ** 2) * torch.sum(y_c ** 2))
        cc = torch.sum(y_hat_c * y_c) / (denom + 1e-12)

        alpha = torch.std(y_hat, unbiased=False) / (torch.std(y, unbiased=False) + 1e-12)
        beta = torch.mean(y_hat) / (torch.mean(y) + 1e-12)
        kge = 1.0 - torch.sqrt(
            (cc - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )

        return mse, rmse, bias, cc, kge

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]

        y_hat = self(x)
        loss = self._loss(y_hat, y)
        y_point = self._point_prediction(y_hat)

        mse, rmse, bias, cc, kge = self._metrics(y_point, y)

        B = x.shape[0]
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=B, prog_bar=True)
        self.log("train/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("train/rmse", rmse, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("train/bias", bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("train/cc", cc, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("train/kge", kge, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]

        y_hat = self(x)
        loss = self._loss(y_hat, y)
        y_point = self._point_prediction(y_hat)

        mse, rmse, bias, cc, kge = self._metrics(y_point, y)

        B = x.shape[0]
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=B, prog_bar=True)
        self.log("val/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/rmse", rmse, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/bias", bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/cc", cc, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/kge", kge, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        sched_name = (self.scheduler_name or "none").lower()

        if sched_name in ("none", "", "null"):
            return optimizer

        if sched_name == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_t_max,
                eta_min=self.scheduler_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        raise ValueError(f"Unknown scheduler_name: {self.scheduler_name}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=RUN_NAME)

    parser.add_argument("--training_task_mode", type=str, default="global", choices=["global", "per_day"])
    parser.add_argument("--training_task_date", type=str, default="")
    parser.add_argument("--training_task_dates", type=str, nargs="+", default=None)
    parser.add_argument("--training_task_worker_index", type=int, default=0)
    parser.add_argument("--training_task_num_workers", type=int, default=1)
    parser.add_argument("--training_task_progress_worker_index", type=int, default=0)
    parser.add_argument("--training_task_min_train_precip_mean", type=float, default=None)
    parser.add_argument("--training_task_val_fraction", type=float, default=0.20)
    parser.add_argument("--training_task_min_rows", type=int, default=20)
    parser.add_argument("--training_task_seed", type=int, default=42)
    parser.add_argument("--training_task_precip_seq_dir", type=str, default="")

    parser.add_argument("--data_csv_path", type=str, required=False)
    parser.add_argument("--data_batch_size", type=int, default=4096)
    parser.add_argument("--data_num_workers", type=int, default=4)
    parser.add_argument("--data_train_years", type=int, nargs=2, default=(2005, 2018))
    parser.add_argument("--data_val_years", type=int, nargs=2, default=(2019, 2019))
    parser.add_argument("--data_test_years", type=int, nargs=2, default=(2020, 2024))
    parser.add_argument("--data_exclude_train_years", type=int, nargs="+", action="append", default=None)
    parser.add_argument(
        "--data_feature_cols",
        type=str,
        nargs="+",
        default=["ERA5", "IMERG", "idw_loo"],
    )

    parser.add_argument("--model_hidden_dims", type=int, nargs="+", default=[128, 128, 64])
    parser.add_argument("--model_dropout", type=float, default=0.1)

    parser.add_argument("--optim_lr", type=float, default=1e-3)
    parser.add_argument("--optim_weight_decay", type=float, default=0.0)

    parser.add_argument("--loss_name", type=str, default="mse", choices=["mse", "quantile"])
    parser.add_argument("--loss_n_quantiles", type=int, default=32)
    parser.add_argument("--loss_quantiles", type=float, nargs="+", default=None)
    parser.add_argument("--loss_quantile_weights", type=float, nargs="+", default=None)
    parser.add_argument("--loss_higher_q_weight", type=float, default=None)
    parser.add_argument("--loss_crossing_weight", type=float, default=0.0)

    parser.add_argument("--scheduler_name", type=str, default="CosineAnnealingLR")
    parser.add_argument("--scheduler_t_max", type=int, default=50)
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0)

    parser.add_argument("--trainer_max_epochs", type=int, default=50)
    parser.add_argument("--trainer_devices", type=int, default=1)
    parser.add_argument("--trainer_num_nodes", type=int, default=1)
    parser.add_argument("--trainer_strategy", type=str, default="ddp")
    parser.add_argument("--trainer_accelerator", type=str, default=None)
    parser.add_argument("--trainer_precision", type=str, default="16-mixed")
    parser.add_argument("--trainer_log_every_n_steps", type=int, default=20)
    parser.add_argument("--trainer_early_stopping", action="store_true")
    parser.add_argument("--trainer_early_stopping_patience", type=int, default=5)
    parser.add_argument("--trainer_early_stopping_min_delta", type=float, default=0.0)

    args = parser.parse_args()
    cli_overrides = _cli_overrides(sys.argv[1:])

    if args.config is not None:
        cfg = flatten_config(load_toml_config(args.config))
        for k, v in cfg.items():
            if hasattr(args, k) and k not in cli_overrides:
                setattr(args, k, v)

        if "data_train_years" in cfg:
            args.data_train_years = _as_year_tuple(cfg["data_train_years"], tuple(args.data_train_years))
        if "data_val_years" in cfg:
            args.data_val_years = _as_year_tuple(cfg["data_val_years"], tuple(args.data_val_years))
        if "data_test_years" in cfg:
            args.data_test_years = _as_year_tuple(cfg["data_test_years"], tuple(args.data_test_years))
        if "data_exclude_train_years" in cfg:
            args.data_exclude_train_years = _as_year_ranges(cfg["data_exclude_train_years"])
        else:
            args.data_exclude_train_years = _as_year_ranges(args.data_exclude_train_years)

    if not args.data_csv_path:
        raise ValueError("csv path must be provided via --data_csv_path or [data].csv_path in TOML")

    if args.training_task_mode == "per_day" and "SLURM_NTASKS" in os.environ:
        args.training_task_worker_index = _slurm_procid()
        args.training_task_num_workers = int(os.environ.get("SLURM_NTASKS", "1"))

    seed_everything(42)
    rank_zero_info(vars(args))

    base_dir = Path(BASE_DIR)
    run_name = getattr(args, "run_name", None) or RUN_NAME

    ckpt_root = base_dir / "checkpoints" / run_name
    log_root = base_dir / "logs"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    feature_cols = list(args.data_feature_cols)
    rank_zero_info(f"Using feature columns: {feature_cols}")

    def make_model() -> MLPLightning:
        return MLPLightning(
            in_dim=len(feature_cols),
            hidden_dims=tuple(args.model_hidden_dims),
            dropout=args.model_dropout,
            lr=args.optim_lr,
            weight_decay=args.optim_weight_decay,
            scheduler_name=args.scheduler_name,
            scheduler_t_max=args.scheduler_t_max,
            scheduler_eta_min=args.scheduler_eta_min,
            loss_name=args.loss_name,
            loss_n_quantiles=args.loss_n_quantiles,
            loss_quantiles=args.loss_quantiles,
            loss_quantile_weights=args.loss_quantile_weights,
            loss_higher_q_weight=args.loss_higher_q_weight,
            loss_crossing_weight=args.loss_crossing_weight,
        )

    def make_trainer(day_run_name: str, day_ckpt_root: Path, show_progress: bool) -> Trainer:
        day_ckpt_root.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=str(log_root),
            name=day_run_name,
            default_hp_metric=False,
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(day_ckpt_root),
            filename="epoch{epoch:03d}-step{step}",
            save_last=True,
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            auto_insert_metric_name=False,
        )
        callbacks = [checkpoint_cb]
        if args.trainer_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    mode="min",
                    patience=args.trainer_early_stopping_patience,
                    min_delta=args.trainer_early_stopping_min_delta,
                    check_on_train_epoch_end=False,
                )
            )
        return Trainer(
            max_epochs=args.trainer_max_epochs,
            accelerator=args.trainer_accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
            devices=args.trainer_devices,
            num_nodes=args.trainer_num_nodes,
            strategy=args.trainer_strategy,
            precision=args.trainer_precision,
            log_every_n_steps=args.trainer_log_every_n_steps,
            enable_model_summary=True,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_progress_bar=show_progress,
        )

    if args.training_task_mode == "per_day":
        df = pd.read_csv(args.data_csv_path, parse_dates=["date"])
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        cols_needed = list(feature_cols) + ["gauge"]
        df = df.dropna(subset=cols_needed).copy()

        available_dates = sorted(df["date_str"].unique().tolist())
        all_dates = _resolve_per_day_dates(args, available_dates)
        before_filter = len(all_dates)

        if args.training_task_min_train_precip_mean is not None:
            if args.training_task_precip_seq_dir:
                day_mean_lookup = _load_sequence_train_precip_means(
                    args.training_task_precip_seq_dir,
                    all_dates,
                )
            else:
                day_mean_lookup = df.groupby("date_str")["gauge"].mean().to_dict()
            kept = []
            skipped = []
            for day in all_dates:
                mean_precip = float(day_mean_lookup[day])
                if mean_precip > float(args.training_task_min_train_precip_mean):
                    kept.append(day)
                else:
                    skipped.append((day, mean_precip))
            all_dates = kept
            rank_zero_info(
                "[MLP_PER_DAY] "
                f"train precipitation mean threshold: > {args.training_task_min_train_precip_mean}. "
                f"Keeping {len(all_dates)} of {before_filter} requested dates; "
                f"skipping {len(skipped)}."
            )
            if skipped:
                preview = ", ".join(f"{day}:{val:.4g}" for day, val in skipped[:10])
                rank_zero_info(f"[MLP_PER_DAY] First skipped dates mean precip: {preview}")

        if not all_dates:
            raise RuntimeError("No per-day dates remain after filtering.")

        worker_dates = _worker_slice(
            all_dates,
            worker_index=args.training_task_worker_index,
            num_workers=args.training_task_num_workers,
        )
        rank_zero_info(
            "[MLP_PER_DAY] "
            f"worker {args.training_task_worker_index}/{args.training_task_num_workers} "
            f"will train {len(worker_dates)} of {len(all_dates)} requested dates."
        )
        progress = _is_progress_worker(args.training_task_progress_worker_index)
        if progress:
            print(
                "[MLP_PER_DAY_PROGRESS] "
                f"watching worker {args.training_task_worker_index}/{args.training_task_num_workers}; "
                f"assigned_dates={len(worker_dates)}; total_filtered_dates={len(all_dates)}",
                flush=True,
            )

        for day_idx, day in enumerate(worker_dates, start=1):
            day_run_name = _per_day_run_name(run_name, day, multi_day=len(all_dates) > 1)
            day_ckpt_root = base_dir / "checkpoints" / day_run_name
            remaining_before = len(worker_dates) - day_idx
            df_day = df.loc[df["date_str"] == day].copy()

            rank_zero_info(
                f"[MLP_PER_DAY] worker {args.training_task_worker_index}/{args.training_task_num_workers} "
                f"date {day_idx}/{len(worker_dates)}: training date={day} rows={len(df_day)} "
                f"run_name={day_run_name}"
            )
            if progress:
                print(
                    "[MLP_PER_DAY_PROGRESS] "
                    f"starting date {day_idx}/{len(worker_dates)} ({day}); "
                    f"remaining_after_this={remaining_before}",
                    flush=True,
                )

            if len(df_day) < int(args.training_task_min_rows):
                day_ckpt_root.mkdir(parents=True, exist_ok=True)
                (day_ckpt_root / "skipped.txt").write_text(
                    f"Skipped {day}: only {len(df_day)} rows; "
                    f"minimum is {args.training_task_min_rows}.\n"
                )
                rank_zero_info(f"[MLP_PER_DAY] skipped {day}: too few rows.")
                continue

            split_seed = int(args.training_task_seed) + int(_compact_date(day))
            df_train, df_val = _split_day_dataframe(
                df_day,
                val_fraction=args.training_task_val_fraction,
                seed=split_seed,
            )
            train_loader, val_loader = _make_loaders_from_dataframes(
                df_train=df_train,
                df_val=df_val,
                feature_cols=feature_cols,
                target_col="gauge",
                batch_size=args.data_batch_size,
                num_workers=args.data_num_workers,
            )
            trainer = make_trainer(day_run_name, day_ckpt_root, progress)
            trainer.fit(
                make_model(),
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=None,
            )
            _save_best_checkpoint_alias(trainer, day_ckpt_root)

            remaining = len(worker_dates) - day_idx
            rank_zero_info(
                f"[MLP_PER_DAY] worker {args.training_task_worker_index}/{args.training_task_num_workers} "
                f"completed {day_idx}/{len(worker_dates)} dates; remaining={remaining}; "
                f"last_date={day}"
            )
            if progress:
                print(
                    "[MLP_PER_DAY_PROGRESS] "
                    f"completed date {day_idx}/{len(worker_dates)} ({day}); remaining={remaining}",
                    flush=True,
                )
        return

    dm = TabularDataModule(
        csv_path=args.data_csv_path,
        feature_cols=feature_cols,
        target_col="gauge",
        batch_size=args.data_batch_size,
        num_workers=args.data_num_workers,
        train_years=tuple(args.data_train_years),
        val_years=tuple(args.data_val_years),
        test_years=tuple(args.data_test_years),
        exclude_train_years=_as_year_ranges(args.data_exclude_train_years),
    )
    dm.setup()

    model = make_model()

    logger = TensorBoardLogger(
        save_dir=str(log_root),
        name=run_name,
        default_hp_metric=False,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_root),
        filename="epoch{epoch:03d}-step{step}",
        save_last=True,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_cb]
    if args.trainer_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=args.trainer_early_stopping_patience,
                min_delta=args.trainer_early_stopping_min_delta,
                check_on_train_epoch_end=False,
            )
        )

    trainer = Trainer(
        max_epochs=args.trainer_max_epochs,
        accelerator=args.trainer_accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=args.trainer_devices,
        num_nodes=args.trainer_num_nodes,
        strategy=args.trainer_strategy,
        precision=args.trainer_precision,
        log_every_n_steps=args.trainer_log_every_n_steps,
        enable_model_summary=True,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)
    _save_best_checkpoint_alias(trainer, ckpt_root)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)

    main()
