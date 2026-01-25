#!/usr/bin/env python3
# coding: utf-8
"""
Train TGCN (multi-feature) for retrospective precipitation estimation
--------------------------------------------------------------------
Input: 14-day window of (ERA5, IMERG) at every node -> estimate gauge precip at day t.

Data: seq_*.pt files, each sample dict contains:
  x: [T, N, 2], y: [N], y_mask: [N], edge_index: [2,E], edge_weight: [E], date: str

Training:
  - chronological split over DAYS: 70% train / 30% test
  - shuffle within train only
  - loss uses y_mask (and ignores NaNs)
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info

from utils.data.seq_datamodule import SpatioTemporalPTDataModule  # your datamodule file
from models.tgcn_multifeat import TGCNRegressor

import tomli as tomllib  # Python 3.11+ ; for 3.10 use: import tomli as tomllib
from typing import Any, Dict

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

RUN_NAME = "TGCN_T14_70train_30test"
BASE_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval"



def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for section, kv in cfg.items():
        if isinstance(kv, dict):
            # special-case meta.run_name -> run_name
            if section == "meta" and "run_name" in kv:
                out["run_name"] = kv["run_name"]
            else:
                out.update(kv)
        else:
            out[section] = kv
    return out


class TGCNLightning(pl.LightningModule):
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
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

    def _masked_mse(self, y_hat, y, y_mask):
        # y_hat, y: [B, N]   y_mask: [B, N]
        mask = y_mask & ~torch.isnan(y)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return F.mse_loss(y_hat[mask], y[mask])

    def training_step(self, batch, batch_idx):
        x = batch["x"]              # [B, T, N, F]
        y = batch["y"]              # [B, N]
        y_mask = batch["y_mask"]    # [B, N]

        y_hat = self(x)             # [B, N]
        loss = self._masked_mse(y_hat, y, y_mask)

        B = batch["x"].shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_mask = batch["y_mask"]

        y_hat = self(x)
        loss = self._masked_mse(y_hat, y, y_mask)

        B = x.shape[0]
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=B)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _masked_vectors(self, y_hat, y, y_mask):
        """
        Return flattened vectors (pred, obs) after applying mask and removing NaNs.
        y_hat, y: [B, N]; y_mask: [B, N] bool
        """
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

        # Pearson correlation coefficient
        pred_c = pred - pred.mean()
        obs_c  = obs - obs.mean()
        denom = torch.sqrt(torch.sum(pred_c**2) * torch.sum(obs_c**2))
        cc = torch.sum(pred_c * obs_c) / (denom + 1e-12)

        return mse, rmse, bias, cc

    def _log_metrics(self, stage, y_hat, y, y_mask, batch_size):
        mse, rmse, bias, cc = self._mse_rmse_bias_cc(y_hat, y, y_mask)

        self.log(f"{stage}/mse",  mse,  prog_bar=(stage!="train"), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/rmse", rmse, prog_bar=(stage!="train"), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/bias", bias, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/cc",   cc,   prog_bar=(stage!="train"), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)

        return mse  # if you want a scalar returned

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_mask = batch["y_mask"]

        y_hat = self(x)
        B = x.shape[0]

        # Use MSE as your optimization loss
        mse, _, _, _ = self._mse_rmse_bias_cc(y_hat, y, y_mask)

        # log epoch-level train metrics
        self._log_metrics("train", y_hat, y, y_mask, batch_size=B)

        # optional: step-level loss for progress bar smoothness
        self.log("train/loss_step", mse, prog_bar=True, on_step=True, on_epoch=False,
                 sync_dist=True, batch_size=B)

        return mse

    def validation_step(self, batch, batch_idx):
        # This is your "holdout/test" split (last 30%)
        x = batch["x"]
        y = batch["y"]
        y_mask = batch["y_mask"]

        y_hat = self(x)
        B = x.shape[0]
        self._log_metrics("holdout", y_hat, y, y_mask, batch_size=B)


def main():
    parser = argparse.ArgumentParser()

    # NEW
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config")

    # Data
    parser.add_argument("--seq_dir", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=4)

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

    # -----------------------
    # Load TOML + override CLI defaults
    # -----------------------
    if args.config is not None:
        cfg = flatten_config(load_toml_config(args.config))
        # overwrite argparse Namespace
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Allow run_name from TOML/CLI to control output folders
    run_name = args.run_name

    base_dir = Path(BASE_DIR)
    ckpt_root = base_dir / "checkpoints" / run_name
    log_root  = base_dir / "logs"

    if not args.seq_dir:
        raise ValueError("seq_dir must be provided via --seq_dir or [data].seq_dir in TOML")

    rank_zero_info(vars(args))
    seed_everything(42)

    dm = SpatioTemporalPTDataModule(
        seq_dir=args.seq_dir,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        num_workers=args.num_workers,
    )
    dm.setup()

    first_file = sorted(Path(args.seq_dir).glob("seq_*.pt"))[0]
    sample = torch.load(first_file, map_location="cpu", weights_only=True)
    T, N, F_in = sample["x"].shape
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        add_self_loops=args.add_self_loops,
        improved=args.improved,
        num_layers=args.num_layers,
    )

    # --- Logger (TensorBoard) ---
    logger = TensorBoardLogger(
        save_dir=str(log_root),
        name=run_name,          # logs/<run_name>/version_*/events...
        default_hp_metric=False
    )

    # --- Checkpoints ---
    ckpt_root.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_root),             # checkpoints/<run_name>/
        filename="epoch{epoch:03d}-step{step}",
        save_last=True,
        save_top_k=1,
        monitor="holdout/mse",
        mode="min",
        auto_insert_metric_name=False
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
        limit_val_batches=0,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    # Only strip SLURM vars when NOT running inside a SLURM job
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)

    main()