#!/usr/bin/env python3
# coding: utf-8
"""
Train TGCN (multi-feature) for retrospective precipitation estimation
--------------------------------------------------------------------
Input: 14-day window of (ERA5, IMERG) at every node -> estimate gauge precip at day t.

Data: seq_*.pt files, each sample dict contains:
  x: [T, N, 2], y: [N], y_mask: [N], edge_index: [2,E], edge_weight: [E], date: str

Splits (year-based):
  - Train: 2005–2018
  - Val:   2019
  - Test:  2020–2024 (NOT used in training; run later independently)

Logging:
  - train/* and val/* metrics per epoch
  - TensorBoard: logs/<run_name>/version_*/events...
  - Checkpoints: checkpoints/<run_name>/
"""

import os
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import tomli as tomllib

from utils.data.seq_datamodule import SpatioTemporalPTDataModule
from models.tgcn_multifeat import TGCNRegressor


RUN_NAME = "TGCN_T14_train2005_2018_val2019"
BASE_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval"


def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten TOML sections into argparse keys.
    Special-case: [meta].run_name -> run_name
    """
    out: Dict[str, Any] = {}
    for section, kv in cfg.items():
        if isinstance(kv, dict):
            if section == "meta" and "run_name" in kv:
                out["run_name"] = kv["run_name"]
            else:
                out.update(kv)
        else:
            out[section] = kv
    return out


def _as_year_tuple(v, default: Tuple[int, int]) -> Tuple[int, int]:
    """
    TOML lists come as python list. We accept list/tuple of length 2.
    """
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"Expected [y0, y1] for years, got: {v}")


class TGCNLightning(pl.LightningModule):
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        add_self_loops: bool = True,
        improved: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["edge_index", "edge_weight"])

        self.model = TGCNRegressor(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            add_self_loops=add_self_loops,
            improved=improved,
        )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    # ---------------- Metrics ----------------
    @staticmethod
    def _masked_vectors(y_hat, y, y_mask):
        mask = y_mask & torch.isfinite(y)
        if mask.sum() == 0:
            return None, None
        return y_hat[mask].float(), y[mask].float()

    def _mse_rmse_bias_cc(self, y_hat, y, y_mask):
        pred, obs = self._masked_vectors(y_hat, y, y_mask)
        if pred is None:
            z = torch.tensor(0.0, device=self.device)
            return z, z, z, z

        diff = pred - obs
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        bias = torch.mean(diff)

        # Pearson CC
        pred_c = pred - pred.mean()
        obs_c = obs - obs.mean()
        denom = torch.sqrt(torch.sum(pred_c**2) * torch.sum(obs_c**2))
        cc = torch.sum(pred_c * obs_c) / (denom + 1e-12)

        return mse, rmse, bias, cc

    def _log_epoch_metrics(self, stage: str, y_hat, y, y_mask, batch_size: int):
        mse, rmse, bias, cc = self._mse_rmse_bias_cc(y_hat, y, y_mask)

        # epoch-level (so tensorboard has one point per epoch)
        self.log(f"{stage}/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/rmse", rmse, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/bias", bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/cc", cc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return mse

    # ---------------- Lightning steps ----------------
    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_mask = batch["y_mask"]

        y_hat = self(x)
        B = x.shape[0]

        mse = self._log_epoch_metrics("train", y_hat, y, y_mask, batch_size=B)

        # optional step-level for progress bar smoothness
        self.log("train/loss_step", mse, prog_bar=True, on_step=True, on_epoch=False,
                 sync_dist=True, batch_size=B)
        return mse

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_mask = batch["y_mask"]

        y_hat = self(x)
        B = x.shape[0]
        self._log_epoch_metrics("val", y_hat, y, y_mask, batch_size=B)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config")

    # Data
    parser.add_argument("--seq_dir", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Year splits
    parser.add_argument("--train_years", type=int, nargs=2, default=(2005, 2018))
    parser.add_argument("--val_years", type=int, nargs=2, default=(2019, 2019))
    parser.add_argument("--test_years", type=int, nargs=2, default=(2020, 2024))

    # Model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--add_self_loops", action="store_true", default=True)
    parser.add_argument("--improved", action="store_true", default=False)

    # Optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Trainer
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--log_every_n_steps", type=int, default=20)

    args = parser.parse_args()

    # Load TOML and override args
    if args.config is not None:
        cfg = flatten_config(load_toml_config(args.config))
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

        # handle year ranges if present in TOML as lists
        if "train_years" in cfg:
            args.train_years = _as_year_tuple(cfg["train_years"], tuple(args.train_years))
        if "val_years" in cfg:
            args.val_years = _as_year_tuple(cfg["val_years"], tuple(args.val_years))
        if "test_years" in cfg:
            args.test_years = _as_year_tuple(cfg["test_years"], tuple(args.test_years))

    if not args.seq_dir:
        raise ValueError("seq_dir must be provided via --seq_dir or [data].seq_dir in TOML")

    rank_zero_info(vars(args))
    seed_everything(42)

    base_dir = Path(BASE_DIR)
    ckpt_root = base_dir / "checkpoints" / args.run_name
    log_root = base_dir / "logs"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    dm = SpatioTemporalPTDataModule(
        seq_dir=args.seq_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_years=tuple(args.train_years),
        val_years=tuple(args.val_years),
        test_years=tuple(args.test_years),
    )
    dm.setup()

    # Get graph + shapes from first file
    first_file = sorted(Path(args.seq_dir).glob("seq_*.pt"))[0]
    sample = torch.load(first_file, map_location="cpu", weights_only=True)
    _, N, _ = sample["x"].shape
    edge_index = sample["edge_index"].long()
    edge_weight = sample.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    model = TGCNLightning(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=N,
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        add_self_loops=args.add_self_loops,
        improved=args.improved,
    )

    logger = TensorBoardLogger(
        save_dir=str(log_root),
        name=args.run_name,              # logs/<run_name>/version_*/
        default_hp_metric=False,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_root),
        filename="epoch{epoch:03d}-step{step}",
        save_last=True,
        save_top_k=1,
        monitor="val/mse",
        mode="min",
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        enable_model_summary=False,
        logger=logger,
        callbacks=[checkpoint_cb],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)

    main()