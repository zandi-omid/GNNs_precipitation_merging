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
from losses import build_loss


RUN_NAME = "TGCN_T14_train2005_2018_val2019"
BASE_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval"


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
            if section == "loss" and k == "name":
                out["loss_name"] = v
            elif section == "loss" and k == "n_quantiles":
                out["loss_n_quantiles"] = v
            elif section == "loss" and k == "crossing_weight":
                out["loss_crossing_weight"] = v
            elif section == "loss" and k == "delta":
                out["loss_delta"] = v
            elif section == "scheduler" and k == "name":
                out["scheduler_scheduler"] = v
            else:
                out[f"{section}_{k}"] = v

    return out


def _as_year_tuple(v, default: Tuple[int, int]) -> Tuple[int, int]:
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
        rnn_layers: int = 2,
        gcn_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        add_self_loops: bool = True,
        improved: bool = False,
        # ---- transforms / clipping ----
        x_transform: str = "none",     # "none" | "log1p"
        y_transform: str = "none",     # "none" | "log1p"
        x_clip_min: Optional[float] = None,
        y_clip_min: Optional[float] = None,
        # ---- scheduler ----
        scheduler: str = "none",       # "none" | "CosineAnnealingLR"
        t_max: int = 20,
        eta_min: float = 0.0,
        # ---- loss ----
        loss_name: str = "mse",
        loss_delta: float = 1.0,
        loss_n_quantiles: int = 32,
        loss_crossing_weight: float = 0.0,
        # ---- debug ----
        debug_every_n_steps: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["edge_index", "edge_weight"])

        self.loss_name = (loss_name or "mse").lower()
        self.n_quantiles = int(loss_n_quantiles)

        # decide output channels Q
        out_channels = self.n_quantiles if self.loss_name == "quantile" else 1

        self.model = TGCNRegressor(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            gcn_layers=gcn_layers,
            add_self_loops=add_self_loops,
            improved=improved,
            out_channels=out_channels,
        )

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.x_transform = (x_transform or "none").lower()
        self.y_transform = (y_transform or "none").lower()
        self.x_clip_min = x_clip_min
        self.y_clip_min = y_clip_min

        self.scheduler_name = (scheduler or "none").lower()
        self.t_max = int(t_max)
        self.eta_min = float(eta_min)

        # build loss
        if self.loss_name == "quantile":
            self.loss_fn = build_loss(
                "quantile",
                n_quantiles=self.n_quantiles,
                crossing_weight=float(loss_crossing_weight),
            )
        elif self.loss_name == "huber":
            self.loss_fn = build_loss("huber", delta=float(loss_delta))
        else:
            self.loss_fn = build_loss("mse")

        # ---- debug helpers ----
        self.debug_every_n_steps = int(debug_every_n_steps)
        self._debug_prev_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_x_transform(x)
        y_hat = self.model(x)  # [B,N,Q] or [B,N,1]
        return y_hat

    # ---------------- Transforms ----------------
    def _apply_x_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_clip_min is not None:
            x = torch.clamp(x, min=float(self.x_clip_min))
        if self.x_transform == "log1p":
            x = torch.log1p(x)
        elif self.x_transform == "none":
            pass
        else:
            raise ValueError(f"Unknown x_transform: {self.x_transform}")
        return x

    def _apply_y_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.y_clip_min is not None:
            y = torch.clamp(y, min=float(self.y_clip_min))
        if self.y_transform == "log1p":
            y = torch.log1p(y)
        elif self.y_transform == "none":
            pass
        else:
            raise ValueError(f"Unknown y_transform: {self.y_transform}")
        return y

    # ---------------- Metrics ----------------
    @staticmethod
    def _masked_vectors(y_hat: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor):
        mask = y_mask & torch.isfinite(y)
        if mask.sum() == 0:
            return None, None
        return y_hat[mask].float(), y[mask].float()

    def _mse_rmse_bias_cc(self, y_hat: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor):
        pred, obs = self._masked_vectors(y_hat, y, y_mask)
        if pred is None:
            z = torch.tensor(0.0, device=self.device)
            return z, z, z, z

        diff = pred - obs
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        bias = torch.mean(diff)

        pred_c = pred - pred.mean()
        obs_c = obs - obs.mean()
        denom = torch.sqrt(torch.sum(pred_c**2) * torch.sum(obs_c**2))
        cc = torch.sum(pred_c * obs_c) / (denom + 1e-12)

        return mse, rmse, bias, cc

    def _log_epoch_metrics(self, stage: str, y_hat: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor, batch_size: int):
        mse, rmse, bias, cc = self._mse_rmse_bias_cc(y_hat, y, y_mask)

        self.log(f"{stage}/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/rmse", rmse, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/bias", bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}/cc", cc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return mse

    # ---------------- Debug hook ----------------
    def on_after_backward(self):
        # Lightning calls this after loss.backward()
        if (self.global_rank == 0) and (self.global_step % self.debug_every_n_steps == 0):
            head_w = self.model.head.weight
            g = head_w.grad
            if g is None:
                print("[DEBUG] AFTER backward: head grad is None -> something is wrong.")
            else:
                print("[DEBUG] AFTER backward: head grad norm =", float(g.norm().item()))

    # ---------------- Lightning steps ----------------
    def training_step(self, batch, batch_idx):
        x = batch["x"]                       # [B,T,N,F]
        y = batch["y"]                       # [B,N]
        y_mask = batch["y_mask"].bool()      # [B,N]

        # Make mask consistent with finite labels
        y_mask = y_mask & torch.isfinite(y)

        # Replace NaNs where mask is False (so math never sees NaN)
        y_safe = torch.where(y_mask, y, torch.zeros_like(y))

        # Safe transform (currently "none" in your config)
        y_t = self._apply_y_transform(y_safe)

        # Forward
        y_hat = self(x)  # [B,N,Q] or [B,N,1]
        B = x.shape[0]

        # ---- DEBUG prints (rank0 only) ----
        debug_every = getattr(self, "debug_every_n_steps", 0)
        do_dbg = (self.global_rank == 0) and (
            batch_idx == 0 or (debug_every and (self.global_step % debug_every == 0))
        )

        if do_dbg:
            with torch.no_grad():
                print("\n[DEBUG] epoch", int(self.current_epoch),
                    "step", int(self.global_step),
                    "batch_idx", int(batch_idx))
                print("x:", tuple(x.shape),
                    "y:", tuple(y.shape),
                    "y_mask:", tuple(y_mask.shape),
                    "y_hat:", tuple(y_hat.shape))

                valid = y_mask & torch.isfinite(y_t)
                n_valid = int(valid.sum().item())
                print("valid count:", n_valid, "/", valid.numel())

                if n_valid > 0:
                    yv = y_t[valid]
                    print("y(valid): min/mean/max =",
                        float(yv.min()), float(yv.mean()), float(yv.max()))

                    # helpful: how many of the VALID targets are exactly 0?
                    frac_zero = float((yv == 0).float().mean().item())
                    print("y(valid) frac==0:", frac_zero)
                else:
                    print("No valid targets in this batch -> loss should be 0")

                if self.loss_name == "quantile":
                    Q = y_hat.shape[-1]
                    q0 = y_hat[:, :, 0]
                    qmid = y_hat[:, :, Q // 2]
                    qlast = y_hat[:, :, -1]

                    if n_valid > 0:
                        print("pred q0(valid):    min/mean/max =",
                            float(q0[valid].min()), float(q0[valid].mean()), float(q0[valid].max()))
                        print("pred qmid(valid):  min/mean/max =",
                            float(qmid[valid].min()), float(qmid[valid].mean()), float(qmid[valid].max()))
                        print("pred qlast(valid): min/mean/max =",
                            float(qlast[valid].min()), float(qlast[valid].mean()), float(qlast[valid].max()))

                        # Crossing check: count decreases between adjacent quantiles,
                        # and require the node to be valid (same valid mask applies to all Q for that node).
                        diffs = y_hat[:, :, 1:] - y_hat[:, :, :-1]   # [B,N,Q-1]
                        crossings = (diffs < 0) & valid.unsqueeze(-1)  # [B,N,Q-1]
                        denom = valid.sum().float() * float(Q - 1) + 1e-6
                        cross_frac = float(crossings.float().sum().item() / denom.item())
                        print("crossing fraction:", cross_frac)

                        # Spread between lowest and highest quantile (should generally grow)
                        spread = (qlast - q0)
                        print("spread(valid): min/mean/max =",
                            float(spread[valid].min()), float(spread[valid].mean()), float(spread[valid].max()))
                else:
                    # MSE / Huber: y_hat expected [B,N,1]
                    y1 = y_hat.squeeze(-1)
                    if n_valid > 0:
                        print("pred(valid): min/mean/max =",
                            float(y1[valid].min()), float(y1[valid].mean()), float(y1[valid].max()))

                # Head weight change check (safe if attribute not set)
                head_w = self.model.head.weight
                head_norm = float(head_w.data.norm().item())
                prev = getattr(self, "_debug_prev_head", None)

                if prev is None:
                    self._debug_prev_head = head_w.data.detach().clone()
                    print("head weight norm:", head_norm, "(init)")
                else:
                    delta = float((head_w.data - prev).norm().item())
                    print("head weight norm:", head_norm, "delta since last dbg:", delta)
                    self._debug_prev_head = head_w.data.detach().clone()

        # Loss + metrics channel selection
        if self.loss_name == "quantile":
            # y_hat [B,N,Q], y_t [B,N], y_mask [B,N]
            loss = self.loss_fn(y_hat, y_t, y_mask)
            q_mid = y_hat.shape[-1] // 2
            y_for_metrics = y_hat[:, :, q_mid]  # [B,N]
        else:
            y_hat_1 = y_hat.squeeze(-1)         # [B,N]
            loss = self.loss_fn(y_hat_1, y_t, y_mask)
            y_for_metrics = y_hat_1

        # Extra debug scalar logs (cheap)
        with torch.no_grad():
            valid = y_mask & torch.isfinite(y_t)
            self.log("debug/n_valid", valid.sum().float(), on_step=True, on_epoch=False, sync_dist=True, batch_size=B)
            if self.loss_name == "quantile" and valid.sum() > 0:
                y_ev = y_hat.mean(dim=-1)  # simple proxy for expected value
                self.log("debug/pred_ev_mean", y_ev[valid].mean(), on_step=True, on_epoch=False, sync_dist=True, batch_size=B)
                self.log("debug/pred_ev_std", y_ev[valid].std(), on_step=True, on_epoch=False, sync_dist=True, batch_size=B)
                self.log("debug/y_mean", y_t[valid].mean(), on_step=True, on_epoch=False, sync_dist=True, batch_size=B)

        # Epoch metrics (still based on one quantile for quantile case)
        self._log_epoch_metrics("train", y_for_metrics, y_t, y_mask, batch_size=B)

        self.log(
            "train/loss_step",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=B,
        )

        # LR logging
        opt = self.optimizers()
        if opt is not None and len(opt.param_groups) > 0:
            self.log("lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_mask = batch["y_mask"].bool()

        y_mask = y_mask & torch.isfinite(y)
        y_safe = torch.where(y_mask, y, torch.zeros_like(y))
        y_t = self._apply_y_transform(y_safe)

        y_hat = self(x)
        B = x.shape[0]

        if self.loss_name == "quantile":
            loss = self.loss_fn(y_hat, y_t, y_mask)
            q_mid = y_hat.shape[-1] // 2
            y_for_metrics = y_hat[:, :, q_mid]
        else:
            y_hat_1 = y_hat.squeeze(-1)
            loss = self.loss_fn(y_hat_1, y_t, y_mask)
            y_for_metrics = y_hat_1

        # One concise val debug line (rank0, first batch)
        if self.global_rank == 0 and batch_idx == 0:
            valid = y_mask & torch.isfinite(y_t)
            print(
                "[VAL DEBUG] any NaN y_safe:", bool(torch.isnan(y_safe).any().item()),
                "any NaN y_t:", bool(torch.isnan(y_t).any().item()),
                "valid count:", int(valid.sum().item())
            )

        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self._log_epoch_metrics("val", y_for_metrics, y_t, y_mask, batch_size=B)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_name in ("none", "", None):
            return optimizer

        if self.scheduler_name == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.t_max,
                eta_min=self.eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        raise ValueError(f"Unknown scheduler: {self.scheduler_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)

    # Data
    parser.add_argument("--data_seq_dir", type=str, required=False)
    parser.add_argument("--data_batch_size", type=int, default=32)
    parser.add_argument("--data_num_workers", type=int, default=4)
    parser.add_argument("--data_train_years", type=int, nargs=2, default=(2005, 2018))
    parser.add_argument("--data_val_years", type=int, nargs=2, default=(2019, 2019))
    parser.add_argument("--data_test_years", type=int, nargs=2, default=(2020, 2024))

    # Model
    parser.add_argument("--model_hidden_dim", type=int, default=64)
    parser.add_argument("--model_in_channels", type=int, default=2)
    parser.add_argument("--model_add_self_loops", action="store_true", default=True)
    parser.add_argument("--model_improved", action="store_true", default=False)
    parser.add_argument("--model_rnn_layers", type=int, default=1)
    parser.add_argument("--model_gcn_layers", type=int, default=1)

    # Optim
    parser.add_argument("--optim_lr", type=float, default=1e-3)
    parser.add_argument("--optim_weight_decay", type=float, default=0.0)

    # Loss
    parser.add_argument("--loss_name", type=str, default="mse", choices=["mse", "huber", "quantile"])
    parser.add_argument("--loss_delta", type=float, default=1.0)
    parser.add_argument("--loss_n_quantiles", type=int, default=32)
    parser.add_argument("--loss_crossing_weight", type=float, default=0.0)

    # LR Scheduler
    parser.add_argument("--scheduler_scheduler", type=str, default="CosineAnnealingLR", choices=["none", "CosineAnnealingLR"])
    parser.add_argument("--scheduler_t_max", type=int, default=20)
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0)

    # Transform
    parser.add_argument("--transform_x_transform", type=str, default="none", choices=["none", "log1p"])
    parser.add_argument("--transform_y_transform", type=str, default="none", choices=["none", "log1p"])
    parser.add_argument("--transform_x_clip_min", type=float, default=0.0)
    parser.add_argument("--transform_y_clip_min", type=float, default=0.0)

    # Trainer
    parser.add_argument("--trainer_max_epochs", type=int, default=10)
    parser.add_argument("--trainer_devices", type=int, default=1)
    parser.add_argument("--trainer_num_nodes", type=int, default=1)
    parser.add_argument("--trainer_strategy", type=str, default="ddp")
    parser.add_argument("--trainer_accelerator", type=str, default=None)
    parser.add_argument("--trainer_precision", type=str, default="16-mixed")
    parser.add_argument("--trainer_log_every_n_steps", type=int, default=20)

    # Debug
    parser.add_argument("--debug_every_n_steps", type=int, default=200)

    args = parser.parse_args()

    # Load TOML and override args
    if args.config is not None:
        cfg = flatten_config(load_toml_config(args.config))
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

        if "data_train_years" in cfg:
            args.data_train_years = _as_year_tuple(cfg["data_train_years"], tuple(args.data_train_years))
        if "data_val_years" in cfg:
            args.data_val_years = _as_year_tuple(cfg["data_val_years"], tuple(args.data_val_years))
        if "data_test_years" in cfg:
            args.data_test_years = _as_year_tuple(cfg["data_test_years"], tuple(args.data_test_years))

    if not args.data_seq_dir:
        raise ValueError("seq_dir must be provided via --data_seq_dir or [data].seq_dir in TOML")

    rank_zero_info(vars(args))
    seed_everything(42)

    base_dir = Path(BASE_DIR)

    run_name = (
        getattr(args, "run_name", None)
        or getattr(args, "meta_run_name", None)
        or getattr(args, "meta__run_name", None)
        or RUN_NAME
    )

    ckpt_root = base_dir / "checkpoints" / run_name
    log_root = base_dir / "logs"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    dm = SpatioTemporalPTDataModule(
        seq_dir=args.data_seq_dir,
        batch_size=args.data_batch_size,
        num_workers=args.data_num_workers,
        train_years=tuple(args.data_train_years),
        val_years=tuple(args.data_val_years),
        test_years=tuple(args.data_test_years),
    )
    dm.setup()

    # Get graph + shapes from first file
    first_file = sorted(Path(args.data_seq_dir).glob("seq_*.pt"))[0]
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
        in_channels=args.model_in_channels,
        hidden_dim=args.model_hidden_dim,
        rnn_layers=args.model_rnn_layers,
        gcn_layers=args.model_gcn_layers,
        lr=args.optim_lr,
        weight_decay=args.optim_weight_decay,
        add_self_loops=args.model_add_self_loops,
        improved=args.model_improved,
        scheduler=args.scheduler_scheduler,
        t_max=args.scheduler_t_max,
        eta_min=args.scheduler_eta_min,
        x_transform=args.transform_x_transform,
        y_transform=args.transform_y_transform,
        x_clip_min=args.transform_x_clip_min,
        y_clip_min=args.transform_y_clip_min,
        loss_name=args.loss_name,
        loss_delta=args.loss_delta,
        loss_n_quantiles=args.loss_n_quantiles,
        loss_crossing_weight=args.loss_crossing_weight,
        debug_every_n_steps=args.debug_every_n_steps,
    )

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
        monitor="val/mse",
        mode="min",
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=args.trainer_max_epochs,
        accelerator=args.trainer_accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=args.trainer_devices,
        num_nodes=args.trainer_num_nodes,
        strategy=args.trainer_strategy,
        precision=args.trainer_precision,
        log_every_n_steps=args.trainer_log_every_n_steps,
        enable_model_summary=False,
        logger=logger,
        callbacks=[checkpoint_cb],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)

    main()