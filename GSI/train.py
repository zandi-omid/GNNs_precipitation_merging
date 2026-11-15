#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import os
import sys
import argparse
import pickle
import shutil
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import tomli as tomllib

GNN_ROOT = Path(__file__).resolve().parents[1]
GSI_DIR = GNN_ROOT / "GSI"
if str(GSI_DIR) not in sys.path:
    sys.path.insert(0, str(GSI_DIR))

from utils.data.tgsi_seq_datamodule import (
    TGSIPTDataModule,
    TGSIRandomMaskedPTDataset,
    TGSIHoldoutPTDataset,
    collate_tgsi,
    seed_tgsi_worker,
)

from models.gsi_gcn import GSI_GCNRegressor
from models.gsi_gat import GSI_GATRegressor
from models.gsi_tgcn import GSI_TGCNRegressor
from models.gsi_tgat import GSI_TGATRegressor
from models.gsi_gatgcn import GSI_GATGCNRegressor
from models.gsi_tgatgcn import GSI_TGATGCNRegressor
from models.gsi_a1gat import GSI_A1GATRegressor
from models.gsi_a1_edgegat import (
    GSI_A1DistanceGATRegressor,
    GSI_A1EdgeAwareGATRegressor,
)

from losses import build_loss

BASE_DIR = str(GSI_DIR)


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
            elif section == "loss" and k == "quantiles":
                out["loss_quantiles"] = v
            elif section == "loss" and k == "quantile_weights":
                out["loss_quantile_weights"] = v
            elif section == "loss" and k == "higher_q_weight":
                out["loss_higher_q_weight"] = v
            elif section == "scheduler" and k == "name":
                out["scheduler_scheduler"] = v
            else:
                out[f"{section}_{k}"] = v

    return out


def _as_year_tuple(v, default: Tuple[int, int]) -> Tuple[int, int]:
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return int(v[0]), int(v[1])
    raise ValueError(f"Expected [y0, y1], got: {v}")


def _as_optional_year_tuple(v) -> Optional[Tuple[int, int]]:
    if v is None:
        return None
    return _as_year_tuple(v, (0, 0))


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


def _approximate_lonlat_to_km(coords_lonlat: np.ndarray) -> np.ndarray:
    mean_lat = np.deg2rad(np.nanmean(coords_lonlat[:, 1]))
    coords_km = np.empty_like(coords_lonlat, dtype=np.float64)
    coords_km[:, 0] = coords_lonlat[:, 0] * 111.0 * np.cos(mean_lat)
    coords_km[:, 1] = coords_lonlat[:, 1] * 111.0
    return coords_km


def _edge_distance_a1_from_graph(
    graph_path: str,
    edge_index_a1: torch.Tensor,
) -> torch.Tensor:
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)

    if "edge_distance_a1_km" in payload:
        edge_distance = payload["edge_distance_a1_km"]
        return torch.as_tensor(edge_distance, dtype=torch.float32)

    graph = payload["graph"]
    common_nodes = payload["common_nodes"]
    coords_lonlat = np.array(
        [(graph.nodes[n]["lon"], graph.nodes[n]["lat"]) for n in common_nodes],
        dtype=np.float64,
    )
    coords_km = _approximate_lonlat_to_km(coords_lonlat)
    edge_np = edge_index_a1.cpu().numpy()
    recv = edge_np[0].astype(np.int64)
    send = edge_np[1].astype(np.int64)
    distance = np.linalg.norm(coords_km[send] - coords_km[recv], axis=1)
    return torch.from_numpy(distance.astype(np.float32))


def _find_file_by_date(seq_dir: Path, date: str) -> Path:
    files = sorted(seq_dir.glob("seq_*.pt"))
    if not files:
        raise FileNotFoundError(f"No seq_*.pt files found in {seq_dir}")

    for fp in files:
        s = torch.load(fp, map_location="cpu", weights_only=True)
        if str(s.get("date", "")) == str(date):
            return fp

    raise FileNotFoundError(f"No sequence file with date={date} found in {seq_dir}")


def _parse_iso_date(value: str) -> date_cls:
    try:
        return date_cls.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Expected date in YYYY-MM-DD format, got: {value!r}") from exc


def _date_range(start: str, end: str) -> List[str]:
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


def _date_compact(date: str) -> str:
    return str(date).replace("-", "")


def _index_files_by_date(seq_files: List[Path]) -> Dict[str, Path]:
    out = {}
    for fp in seq_files:
        s = torch.load(fp, map_location="cpu", weights_only=True)
        if "date" not in s:
            raise KeyError(f"Missing 'date' key in {fp}")
        out[str(s["date"])] = fp
    return out


def _resolve_per_day_dates(args, date_to_file: Dict[str, Path]) -> List[str]:
    if args.training_task_dates:
        raw_dates = [str(d) for d in args.training_task_dates]
        if len(raw_dates) == 2:
            requested = _date_range(raw_dates[0], raw_dates[1])
            dates = [d for d in requested if d in date_to_file]
            if len(dates) < len(requested):
                available_dates = sorted(date_to_file)
                missing = len(requested) - len(dates)
                rank_zero_info(
                    "[PER_DAY] "
                    f"{missing} dates in requested range {raw_dates[0]} to {raw_dates[1]} "
                    "have no sequence file and will be skipped. "
                    f"Available sequence date range is {available_dates[0]} to {available_dates[-1]}."
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

    if not dates:
        available_dates = sorted(date_to_file)
        raise FileNotFoundError(
            "No requested per-day dates have sequence files. "
            f"Available sequence date range is {available_dates[0]} to {available_dates[-1]}."
        )

    missing = [d for d in dates if d not in date_to_file]
    if missing:
        examples = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} requested per-day dates have no sequence file. "
            f"First missing dates: {examples}"
        )

    return dates


def _worker_slice(items: List[str], worker_index: int, num_workers: int) -> List[str]:
    if num_workers < 1:
        raise ValueError(f"training_task.num_workers must be >= 1, got {num_workers}")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(
            f"training_task.worker_index must be in [0, {num_workers - 1}], "
            f"got {worker_index}"
        )
    start = len(items) * worker_index // num_workers
    end = len(items) * (worker_index + 1) // num_workers
    return items[start:end]


def _train_precip_mean(seq_file: Path) -> float:
    s = torch.load(seq_file, map_location="cpu", weights_only=True)
    y = s["y"].float()
    mask = s["train_available_mask"].bool() & torch.isfinite(y)
    if int(mask.sum()) == 0:
        return 0.0
    return float(y[mask].mean().item())


def _filter_dates_by_train_precip(
    dates: List[str],
    date_to_file: Dict[str, Path],
    threshold: Optional[float],
) -> Tuple[List[str], List[Tuple[str, float]]]:
    if threshold is None:
        return dates, []

    keep = []
    skipped = []
    for day in dates:
        mean_precip = _train_precip_mean(date_to_file[day])
        if mean_precip > float(threshold):
            keep.append(day)
        else:
            skipped.append((day, mean_precip))
    return keep, skipped


def _slurm_procid() -> int:
    return int(os.environ.get("SLURM_PROCID", "0"))


def _is_progress_worker(progress_worker_index: int) -> bool:
    return _slurm_procid() == int(progress_worker_index)


def _per_day_run_name(base_run_name: str, day: str, multi_day: bool) -> str:
    compact = _date_compact(day)
    if "{date}" in base_run_name or "{date_compact}" in base_run_name:
        return (
            base_run_name
            .replace("{date}", day)
            .replace("{date_compact}", compact)
        )
    if multi_day:
        return f"{base_run_name}/{compact}"
    return base_run_name


def _print_rendezvous_env() -> None:
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return
    print(
        "[RENDEZVOUS]",
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}",
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
        f"SLURM_NODELIST={os.environ.get('SLURM_NODELIST')}",
        f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')}",
    )

class SparseGaugeLightning(pl.LightningModule):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        edge_index_a1: Optional[torch.Tensor] = None,
        edge_weight_a1: Optional[torch.Tensor] = None,
        edge_index_a2: Optional[torch.Tensor] = None,
        edge_weight_a2: Optional[torch.Tensor] = None,
        input_feature_indices: Optional[list[int]] = None,
        model_type: str = "tgcn",
        hidden_dim: int = 64,
        gsi_hidden: Optional[int] = None,
        rnn_layers: int = 1,
        gcn_layers: int = 2,
        gat_layers: int = 1,
        gat_heads: int = 4,
        gat_dropout: float = 0.0,
        nheads: int = 4,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        improved: bool = False,
        use_temporal_attention: bool = False,
        attn_hidden: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        x_transform: str = "none",
        y_transform: str = "none",
        x_clip_min: Optional[float] = None,
        y_clip_min: Optional[float] = None,
        scheduler: str = "none",
        t_max: int = 20,
        t_mult: int = 1,
        eta_min: float = 0.0,
        loss_name: str = "mse",
        loss_delta: float = 1.0,
        loss_n_quantiles: int = 32,
        loss_crossing_weight: float = 0.0,
        loss_quantiles: Optional[list[float]] = None,
        loss_quantile_weights: Optional[list[float]] = None,
        loss_higher_q_weight: Optional[float] = None,
        t_in_override: Optional[int] = None,
        adaptive_adj_enabled: bool = False,
        adaptive_adj_mode: str = "edge",
        gauge_mask_channel: int = 1,
        edge_distance_a1_km: Optional[torch.Tensor] = None,
        edge_aware_enabled: bool = False,
        edge_aware_mode: str = "learnable_log_distance",
        edge_aware_epsilon_km: float = 1e-3,
        edge_aware_lambda_init: float = 1.0,
        edge_aware_per_head: bool = False,
        hide_receiver_gauge_features: bool = False,
        receiver_gauge_feature_channels: Optional[list[int]] = None,
        distance_attention: str = "none",
        distance_epsilon_km: float = 1e-3,
        distance_lambda_init: float = 1.0,
        distance_per_head: bool = False,
        distance_lambda_mode: Optional[str] = None,
        fusion_mode: str = "legacy",
        a1_hidden_dim: Optional[int] = None,
        a1_feature_channels: Optional[list[int]] = None,
        aux_feature_channels: Optional[list[int]] = None,
        aux_hidden_dim: int = 32,
        gcn_intermediate_dim: int = 128,
        graph_source: str = "a1a2",
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "edge_index",
                "edge_weight",
                "edge_index_a1",
                "edge_weight_a1",
                "edge_distance_a1_km",
                "edge_index_a2",
                "edge_weight_a2",
            ]
        )

        self.model_type = model_type.lower()
        self.input_feature_indices = input_feature_indices
        self.loss_name = loss_name.lower()
        self.n_quantiles = int(loss_n_quantiles)
        self.loss_quantiles = loss_quantiles

        if self.loss_name == "quantile":
            if loss_quantiles is not None:
                self.quantiles = torch.tensor(loss_quantiles, dtype=torch.float32)
                self.n_quantiles = len(loss_quantiles)
            else:
                self.quantiles = (
                    torch.arange(1, self.n_quantiles + 1, dtype=torch.float32)
                    / (self.n_quantiles + 1)
                )
            self.q50_idx = int(torch.argmin(torch.abs(self.quantiles - 0.5)).item())
        else:
            self.quantiles = None
            self.q50_idx = None

        out_channels = self.n_quantiles if self.loss_name == "quantile" else 1
        if hide_receiver_gauge_features and receiver_gauge_feature_channels is None:
            receiver_gauge_feature_channels = [0, gauge_mask_channel]

        valid_model_types = {
            "a1gat",
            "a1distancegat",
            "a1edgegat",
            "gcn",
            "gat",
            "gatgcn",
            "tgcn",
            "tgat",
            "tgatgcn",
        }
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"Unknown model_type={self.model_type!r}. "
                f"Choose from: {sorted(valid_model_types)}"
            )

        if self.model_type == "a1gat":
            if edge_index_a1 is None:
                raise ValueError(
                    "edge_index_a1 is required for model_type='a1gat'"
                )
            self.model = GSI_A1GATRegressor(
                edge_index_a1=edge_index_a1,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                heads=gat_heads,
                dropout=gat_dropout,
                out_channels=out_channels,
                gauge_mask_channel=gauge_mask_channel,
                adaptive_edges=adaptive_adj_enabled,
            )

        elif self.model_type in {"a1distancegat", "a1edgegat"}:
            if edge_index_a1 is None:
                raise ValueError(
                    f"edge_index_a1 is required for model_type={self.model_type!r}"
                )
            if not edge_aware_enabled:
                edge_aware_mode = "none"
            self.model = GSI_A1DistanceGATRegressor(
                edge_index_a1=edge_index_a1,
                edge_distance_a1_km=edge_distance_a1_km,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                heads=gat_heads,
                dropout=gat_dropout,
                out_channels=out_channels,
                gauge_mask_channel=gauge_mask_channel,
                edge_mode=edge_aware_mode,
                epsilon_km=edge_aware_epsilon_km,
                lambda_init=edge_aware_lambda_init,
                per_head=edge_aware_per_head,
                adaptive_edges=adaptive_adj_enabled,
            )

        elif self.model_type == "gcn":
            self.model = GSI_GCNRegressor(
                edge_index_a1=edge_index_a1,
                edge_weight_a1=edge_weight_a1,
                edge_index_a2=edge_index_a2,
                edge_weight_a2=edge_weight_a2,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                dropout=dropout,
                out_channels=out_channels,
                adaptive_edges=adaptive_adj_enabled,
                graph_source=graph_source,
            )

        elif self.model_type == "gat":
            self.model = GSI_GATRegressor(
                edge_index_a1=edge_index_a1,
                edge_weight_a1=edge_weight_a1,
                edge_index_a2=edge_index_a2,
                edge_weight_a2=edge_weight_a2,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                heads=gat_heads,
                dropout=gat_dropout,
                out_channels=out_channels,
                adaptive_edges=adaptive_adj_enabled,
            )

        elif self.model_type == "gatgcn":
            self.model = GSI_GATGCNRegressor(
                edge_index_a1=edge_index_a1,
                edge_weight_a1=edge_weight_a1,
                edge_index_a2=edge_index_a2,
                edge_weight_a2=edge_weight_a2,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                heads=gat_heads,
                dropout=gat_dropout,
                out_channels=out_channels,
                adaptive_edges=adaptive_adj_enabled,
                gauge_mask_channel=gauge_mask_channel,
                receiver_gauge_feature_channels=receiver_gauge_feature_channels,
                edge_distance_a1_km=edge_distance_a1_km,
                distance_attention=distance_attention,
                distance_epsilon_km=distance_epsilon_km,
                distance_lambda_init=distance_lambda_init,
                distance_per_head=distance_per_head,
                distance_lambda_mode=distance_lambda_mode,
                fusion_mode=fusion_mode,
                a1_hidden_dim=a1_hidden_dim,
                a1_feature_channels=a1_feature_channels,
                aux_feature_channels=aux_feature_channels,
                aux_hidden_dim=aux_hidden_dim,
                gcn_intermediate_dim=gcn_intermediate_dim,
            )

        elif self.model_type == "tgcn":
            self.model = GSI_TGCNRegressor(
                edge_index_a1=edge_index_a1,
                edge_weight_a1=edge_weight_a1,
                edge_index_a2=edge_index_a2,
                edge_weight_a2=edge_weight_a2,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                rnn_layers=rnn_layers,
                dropout=dropout,
                out_channels=out_channels,
                adaptive_edges=adaptive_adj_enabled,
                use_temporal_attention=use_temporal_attention,
                attn_hidden=attn_hidden,
                edge_distance_a1_km=edge_distance_a1_km,
                distance_attention=distance_attention,
                distance_epsilon_km=distance_epsilon_km,
                distance_lambda_init=distance_lambda_init,
                distance_per_head=distance_per_head,
                distance_lambda_mode=distance_lambda_mode,
            )

        elif self.model_type == "tgat":
            self.model = GSI_TGATRegressor(
                edge_index_a1=edge_index_a1,
                edge_weight_a1=edge_weight_a1,
                edge_index_a2=edge_index_a2,
                edge_weight_a2=edge_weight_a2,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                rnn_layers=rnn_layers,
                heads=gat_heads,
                dropout=gat_dropout,
                out_channels=out_channels,
                adaptive_edges=adaptive_adj_enabled,
                use_temporal_attention=use_temporal_attention,
                attn_hidden=attn_hidden,
            )

        elif self.model_type == "tgatgcn":
            self.model = GSI_TGATGCNRegressor(
                edge_index_a1=edge_index_a1,
                edge_weight_a1=edge_weight_a1,
                edge_index_a2=edge_index_a2,
                edge_weight_a2=edge_weight_a2,
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                rnn_layers=rnn_layers,
                heads=gat_heads,
                dropout=gat_dropout,
                out_channels=out_channels,
                adaptive_edges=adaptive_adj_enabled,
                use_temporal_attention=use_temporal_attention,
                attn_hidden=attn_hidden,
                gauge_mask_channel=gauge_mask_channel,
                receiver_gauge_feature_channels=receiver_gauge_feature_channels,
                edge_distance_a1_km=edge_distance_a1_km,
                distance_attention=distance_attention,
                distance_epsilon_km=distance_epsilon_km,
                distance_lambda_init=distance_lambda_init,
                distance_per_head=distance_per_head,
                distance_lambda_mode=distance_lambda_mode,
                fusion_mode=fusion_mode,
                a1_hidden_dim=(64 if a1_hidden_dim is None else a1_hidden_dim),
                a1_feature_channels=a1_feature_channels,
                aux_feature_channels=aux_feature_channels,
                aux_hidden_dim=aux_hidden_dim,
                gcn_intermediate_dim=gcn_intermediate_dim,
            )
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.x_transform = x_transform.lower()
        self.y_transform = y_transform.lower()
        self.x_clip_min = x_clip_min
        self.y_clip_min = y_clip_min
        self.scheduler_name = scheduler.lower()
        self.t_max = int(t_max)
        self.t_mult = int(t_mult)
        self.eta_min = float(eta_min)
        self.t_in_override = t_in_override

        if self.loss_name == "quantile":
            self.loss_fn = build_loss(
                "quantile",
                n_quantiles=self.n_quantiles,
                quantiles=loss_quantiles,
                quantile_weights=loss_quantile_weights,
                higher_q_weight=loss_higher_q_weight,
                crossing_weight=float(loss_crossing_weight),
            )
        elif self.loss_name == "huber":
            self.loss_fn = build_loss("huber", delta=float(loss_delta))
        else:
            self.loss_fn = build_loss("mse")

    def _slice_time_window(self, x):
        if self.t_in_override is None:
            return x
        k = int(self.t_in_override)
        return x[:, -k:, :, :]

    def _apply_x_transform(self, x):
        if self.x_clip_min is not None:
            x = torch.clamp(x, min=float(self.x_clip_min))
        if self.x_transform == "log1p":
            x = torch.log1p(x)
        elif self.x_transform != "none":
            raise ValueError(f"Unknown x_transform: {self.x_transform}")
        return x

    def _apply_y_transform(self, y):
        if self.y_clip_min is not None:
            y = torch.clamp(y, min=float(self.y_clip_min))
        if self.y_transform == "log1p":
            y = torch.log1p(y)
        elif self.y_transform != "none":
            raise ValueError(f"Unknown y_transform: {self.y_transform}")
        return y

    def forward(self, x):
        x = self._slice_time_window(x)

        if self.input_feature_indices is not None:
            x = x[..., self.input_feature_indices]

        x = self._apply_x_transform(x)
        return self.model(x)

    def _lambda_distance_values(self):
        value = getattr(self.model, "lambda_distance", None)
        if value is None:
            return None
        if callable(value):
            value = value()
        if value is None:
            return None
        return value.float()

    @staticmethod
    def _masked_vectors(y_hat, y, mask):
        mask = mask.bool() & torch.isfinite(y)
        if mask.sum() == 0:
            return None, None
        return y_hat[mask].float(), y[mask].float()

    def _metrics(self, y_hat, y, mask):
        pred, obs = self._masked_vectors(y_hat, y, mask)
        if pred is None:
            z = torch.tensor(0.0, device=self.device)
            return z, z, z, z, z

        diff = pred - obs
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)
        bias = torch.mean(diff)

        pred_c = pred - pred.mean()
        obs_c = obs - obs.mean()
        cc = torch.sum(pred_c * obs_c) / (
            torch.sqrt(torch.sum(pred_c ** 2) * torch.sum(obs_c ** 2)) + 1e-12
        )

        alpha = torch.std(pred, unbiased=False) / (torch.std(obs, unbiased=False) + 1e-12)
        beta = torch.mean(pred) / (torch.mean(obs) + 1e-12)

        kge = 1.0 - torch.sqrt(
            (cc - 1.0) ** 2 +
            (alpha - 1.0) ** 2 +
            (beta - 1.0) ** 2
        )

        return mse, rmse, bias, cc, kge

    def _shared_step(self, batch, stage: str):
        x = batch["x"]
        y = batch["y"]

        if stage == "train":
            mask = batch["train_mask"].bool()
        else:
            mask = batch["loss_mask"].bool()

        mask = mask & torch.isfinite(y)

        if int(mask.sum().item()) == 0:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log(
                f"{stage}/loss",
                zero,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
            return zero

        y_safe = torch.where(mask, y, torch.zeros_like(y))
        y_t = self._apply_y_transform(y_safe)

        y_hat = self(x)
        B = x.shape[0]

        if self.loss_name == "quantile":
            loss = self.loss_fn(y_hat, y_t, mask)
            y_metric = y_hat[:, :, self.q50_idx]
        else:
            y_hat_1 = y_hat.squeeze(-1)
            loss = self.loss_fn(y_hat_1, y_t, mask)
            y_metric = y_hat_1

        mse, rmse, bias, cc, kge = self._metrics(y_metric, y_t, mask)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(f"{stage}/mse", mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(f"{stage}/rmse", rmse, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(f"{stage}/bias", bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(f"{stage}/cc", cc, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(f"{stage}/kge", kge, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        lambda_dist = self._lambda_distance_values()
        if lambda_dist is not None:
            lambda_mean = lambda_dist.mean()
            lambda_mode = getattr(self.model, "distance_lambda_mode", "global")
            self.log(
                f"{stage}/lambda_dist",
                lambda_mean,
                prog_bar=(stage == "train"),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=B,
            )
            if lambda_mode == "per_head" and lambda_dist.ndim > 0 and lambda_dist.numel() > 1:
                for head_idx, value in enumerate(lambda_dist.flatten()):
                    self.log(
                        f"{stage}/lambda_head_{head_idx}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=B,
                    )
            elif lambda_mode == "per_receiver_node" and lambda_dist.ndim > 0 and lambda_dist.numel() > 1:
                for name, value in {
                    "min": lambda_dist.min(),
                    "max": lambda_dist.max(),
                    "std": lambda_dist.std(unbiased=False),
                }.items():
                    self.log(
                        f"{stage}/lambda_node_{name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=B,
                    )

        if (
            self.global_rank == 0
            and os.environ.get("GSI_PROGRESS_WORKER", "0") == os.environ.get("SLURM_PROCID", "0")
            and stage in {"train", "val"}
            and self.global_step == 0
        ):
            print(
                f"[{stage.upper()} DEBUG]",
                "x:", tuple(x.shape),
                "y:", tuple(y.shape),
                "mask nodes:", int(mask.sum().item()),
                "context nodes:", int(batch.get("context_mask", torch.zeros_like(mask)).sum().item()),
                "loss:", float(loss.detach().cpu()),
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def on_train_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_end(self):
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        if self.global_rank == 0:
            gib = 1024 ** 3
            print(
                "[CUDA PEAK MEMORY] "
                f"allocated={max_allocated / gib:.2f} GiB ({max_allocated} bytes), "
                f"reserved={max_reserved / gib:.2f} GiB ({max_reserved} bytes)"
            )

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_name in {"none", "", None}:
            return opt

        if self.scheduler_name == "cosineannealinglr":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.t_max,
                eta_min=self.eta_min,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        if self.scheduler_name == "cosineannealingwarmrestarts":
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=self.t_max,
                T_mult=self.t_mult,
                eta_min=self.eta_min,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

def str_to_bool(value):

    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a Boolean value, got {value!r}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--training_task_mode", type=str, default="global", choices=["global", "per_day"])
    parser.add_argument("--training_task_date", type=str, default=None)
    parser.add_argument("--training_task_dates", type=str, nargs="+", default=None)
    parser.add_argument("--training_task_worker_index", type=int, default=0)
    parser.add_argument("--training_task_num_workers", type=int, default=1)
    parser.add_argument("--training_task_min_train_precip_mean", type=float, default=None)
    parser.add_argument("--training_task_progress_worker_index", type=int, default=0)

    parser.add_argument("--data_seq_dir", type=str, required=False)
    parser.add_argument("--data_graph_path", type=str, default=None)
    parser.add_argument("--data_batch_size", type=int, default=8)
    parser.add_argument("--data_num_workers", type=int, default=4)
    parser.add_argument("--data_train_years", type=int, nargs=2, default=(2005, 2018))
    parser.add_argument("--data_val_years", type=int, nargs=2, default=None)
    parser.add_argument("--data_test_years", type=int, nargs=2, default=(2020, 2024))
    parser.add_argument("--data_exclude_train_years", type=int, nargs="+", action="append", default=None)
    parser.add_argument("--data_val_fraction", type=float, default=0.15)
    parser.add_argument("--data_t_in_override", type=int, default=None)

    parser.add_argument("--graph_source", type=str, default="a1a2", choices=["a1", "a2", "a1a2"])

    parser.add_argument("--masking_samples_per_epoch", type=int, default=10000)
    parser.add_argument("--masking_context_fraction", type=float, default=0.50)
    parser.add_argument("--masking_context_fraction_min", type=float, default=None)
    parser.add_argument("--masking_context_fraction_max", type=float, default=None)
    parser.add_argument("--masking_gauge_value_channel", type=int, default=0)
    parser.add_argument("--masking_gauge_mask_channel", type=int, default=1)
    parser.add_argument("--masking_min_context_gauges", type=int, default=5)
    parser.add_argument("--masking_min_loss_gauges", type=int, default=5)
    parser.add_argument("--masking_seed", type=int, default=42)

    parser.add_argument("--model_model_type", type=str, default="tgcn", choices=["a1gat", "a1distancegat", "a1edgegat", "gcn", "gat", "gatgcn", "tgcn", "tgat", "tgatgcn"],)
    parser.add_argument("--model_hidden_dim", type=int, default=128)
    parser.add_argument("--model_gsi_hidden", type=int, default=None)
    parser.add_argument("--model_input_feature_indices", type=int, nargs="+", default=None)
    parser.add_argument("--model_nheads", type=int, default=4)
    parser.add_argument("--model_dropout", type=float, default=0.0)
    parser.add_argument("--model_add_self_loops", type=str_to_bool, default=True)
    parser.add_argument("--model_improved", type=str_to_bool, default=False)
    parser.add_argument("--model_rnn_layers", type=int, default=1)
    parser.add_argument("--model_gcn_layers", type=int, default=2)
    parser.add_argument("--model_gat_layers", type=int, default=1)
    parser.add_argument("--model_gat_heads", type=int, default=4)
    parser.add_argument("--model_gat_dropout", type=float, default=0.0)
    parser.add_argument("--model_use_temporal_attention", type=str_to_bool, default=False)
    parser.add_argument("--model_attn_hidden", type=int, default=None)
    parser.add_argument("--model_gauge_mask_channel", type=int, default=1)
    parser.add_argument("--model_hide_receiver_gauge_features", type=str_to_bool, default=False)
    parser.add_argument("--model_receiver_gauge_feature_channels", type=int, nargs="+", default=None)
    parser.add_argument("--model_distance_attention", type=str, default="none", choices=["none", "learnable_log", "learnable_log_distance"])
    parser.add_argument("--model_distance_epsilon_km", type=float, default=1e-3)
    parser.add_argument("--model_distance_lambda_init", type=float, default=1.0)
    parser.add_argument("--model_distance_lambda_mode", type=str, default="global", choices=["global", "per_head", "per_receiver_node"])
    parser.add_argument("--model_distance_per_head", type=str_to_bool, default=False)
    parser.add_argument("--model_fusion_mode", type=str, default="legacy", choices=["legacy", "encoded_late"])
    parser.add_argument("--model_a1_hidden_dim", type=int, default=None)
    parser.add_argument("--model_a1_feature_channels", type=int, nargs="+", default=None)
    parser.add_argument("--model_aux_feature_channels", type=int, nargs="+", default=None)
    parser.add_argument("--model_aux_hidden_dim", type=int, default=32)
    parser.add_argument("--model_gcn_intermediate_dim", type=int, default=128)

    parser.add_argument("--optim_lr", type=float, default=5e-4)
    parser.add_argument("--optim_weight_decay", type=float, default=0.0)

    parser.add_argument("--loss_name", type=str, default="mse", choices=["mse", "huber", "quantile"])
    parser.add_argument("--loss_delta", type=float, default=1.0)
    parser.add_argument("--loss_n_quantiles", type=int, default=32)
    parser.add_argument("--loss_crossing_weight", type=float, default=0.0)
    parser.add_argument("--loss_quantiles", type=float, nargs="+", default=None)
    parser.add_argument("--loss_quantile_weights", type=float, nargs="+", default=None)
    parser.add_argument("--loss_higher_q_weight", type=float, default=None)

    parser.add_argument(
        "--scheduler_scheduler",
        type=str,
        default="CosineAnnealingLR",
        choices=["none", "CosineAnnealingLR", "CosineAnnealingWarmRestarts"],
    )
    parser.add_argument("--scheduler_t_max", type=int, default=200)
    parser.add_argument("--scheduler_t_mult", type=int, default=1)
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0)

    parser.add_argument("--transform_x_transform", type=str, default="none", choices=["none", "log1p"])
    parser.add_argument("--transform_y_transform", type=str, default="none", choices=["none", "log1p"])
    parser.add_argument("--transform_x_clip_min", type=float, default=0.0)
    parser.add_argument("--transform_y_clip_min", type=float, default=0.0)

    parser.add_argument("--trainer_max_epochs", type=int, default=200)
    parser.add_argument("--trainer_precision", type=str, default="16-mixed")
    parser.add_argument("--trainer_log_every_n_steps", type=int, default=20)
    parser.add_argument("--trainer_devices", type=int, default=1)
    parser.add_argument("--trainer_num_nodes", type=int, default=1)
    parser.add_argument("--trainer_accelerator", type=str, default=None)
    parser.add_argument("--trainer_strategy", type=str, default="ddp")
    parser.add_argument("--trainer_early_stopping", type=str_to_bool, default=True)
    parser.add_argument("--trainer_early_stopping_patience", type=int, default=25)
    parser.add_argument("--trainer_early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--trainer_gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--trainer_accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--trainer_resume", type=str_to_bool, default=False)

    parser.add_argument("--adaptive_adj_enabled", type=str_to_bool, default=False)
    parser.add_argument("--adaptive_adj_mode", type=str, default="edge")

    parser.add_argument("--edge_aware_enabled", type=str_to_bool, default=False)
    parser.add_argument("--edge_aware_mode", type=str, default="learnable_log_distance", choices=["none", "fixed_log_distance", "learnable_log_distance"])
    parser.add_argument("--edge_aware_epsilon_km", type=float, default=1e-3)
    parser.add_argument("--edge_aware_lambda_init", type=float, default=1.0)
    parser.add_argument("--edge_aware_per_head", type=str_to_bool, default=False)

    parser.add_argument("--masking_dynamic_masking", type=str_to_bool, default=True,)
    parser.add_argument(
        "--masking_mode",
        type=str,
        default="random_fraction",
        choices=["random_fraction", "crossfit", "all_available"],
    )
    parser.add_argument("--masking_n_folds", type=int, default=10)
    parser.add_argument(
        "--masking_eval_mode",
        type=str,
        default=None,
        choices=["random_fraction", "crossfit", "all_available"],
    )

    cli_overrides = set()
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            continue
        opt = arg.split("=", 1)[0]
        action = parser._option_string_actions.get(opt)
        if action is not None:
            cli_overrides.add(action.dest)

    args = parser.parse_args()

    cfg = flatten_config(load_toml_config(args.config))
    for k, v in cfg.items():
        if hasattr(args, k):
            if k in cli_overrides:
                continue
            setattr(args, k, v)

    if "data_train_years" in cfg:
        args.data_train_years = _as_year_tuple(cfg["data_train_years"], tuple(args.data_train_years))
    if "data_val_years" in cfg:
        args.data_val_years = _as_optional_year_tuple(cfg["data_val_years"])
    if "data_test_years" in cfg:
        args.data_test_years = _as_year_tuple(cfg["data_test_years"], tuple(args.data_test_years))
    if "data_exclude_train_years" in cfg:
        args.data_exclude_train_years = _as_year_ranges(cfg["data_exclude_train_years"])
    else:
        args.data_exclude_train_years = _as_year_ranges(args.data_exclude_train_years)

    if args.training_task_mode == "per_day":
        slurm_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
        slurm_procid = int(os.environ.get("SLURM_PROCID", "0"))
        if slurm_tasks > 1:
            if "training_task_num_workers" not in cli_overrides:
                args.training_task_num_workers = slurm_tasks
            if "training_task_worker_index" not in cli_overrides:
                args.training_task_worker_index = slurm_procid

    os.environ["GSI_PROGRESS_WORKER"] = str(args.training_task_progress_worker_index)

    if not args.data_seq_dir:
        raise ValueError("Please provide [data].seq_dir")

    if not args.run_name:
        raise ValueError("Please provide [meta].run_name")

    _print_rendezvous_env()

    seed_everything(args.masking_seed, workers=True,)

    seq_dir = Path(args.data_seq_dir)
    seq_files = sorted(seq_dir.glob("seq_*.pt"))
    if not seq_files:
        raise FileNotFoundError(f"No seq_*.pt files found in {seq_dir}")

    first_file = seq_files[0]
    sample = torch.load(first_file, map_location="cpu", weights_only=True)

    _, N, F_all = sample["x"].shape

    feature_idx = args.model_input_feature_indices
    if feature_idx is not None:
        feature_idx = [int(i) for i in feature_idx]
        in_channels = len(feature_idx)
    else:
        in_channels = F_all

    base_dir = Path(BASE_DIR)
    log_root = base_dir / "logs"

    def optional_tensor(key: str, dtype_method: str):
        value = sample.get(key)
        if value is None:
            return None
        return getattr(value, dtype_method)()

    edge_distance_a1_km = optional_tensor("edge_distance_a1_km", "float")
    distance_attention_enabled = str(
        getattr(args, "model_distance_attention", "none") or "none"
    ).lower() not in {"none", "off", "false", "standard", "naive"}
    if (
        edge_distance_a1_km is None
        and (
            (
                args.model_model_type in {"a1distancegat", "a1edgegat"}
                and args.edge_aware_enabled
            )
            or (
                args.model_model_type in {"gat", "gatgcn", "tgat", "tgatgcn"}
                and distance_attention_enabled
            )
        )
    ):
        if not args.data_graph_path:
            raise ValueError(
                f"model_type={args.model_model_type!r} with distance-aware attention needs "
                "edge_distance_a1_km in the sequence sample or [data].graph_path "
                "so distances can be computed from graph node coordinates."
            )
        edge_index_a1 = optional_tensor("edge_index_a1", "long")
        if edge_index_a1 is None:
            raise ValueError("edge_index_a1 is required to compute A1 distances.")
        edge_distance_a1_km = _edge_distance_a1_from_graph(
            args.data_graph_path,
            edge_index_a1,
        )
        rank_zero_info(
            "[GSI] Computed edge_distance_a1_km from graph_path for "
            f"{args.model_model_type}: {args.data_graph_path}"
        )
    if distance_attention_enabled:
        rank_zero_info(
            "[GSI] A1 distance-aware GAT attention enabled: "
            f"mode={args.model_distance_attention}, units=km, "
            f"epsilon_km={args.model_distance_epsilon_km}, "
            f"lambda_init={args.model_distance_lambda_init}, "
            f"lambda_mode={args.model_distance_lambda_mode}"
        )

    def make_model() -> SparseGaugeLightning:
        return SparseGaugeLightning(
            num_nodes=N,
            in_channels=in_channels,
            edge_index_a1=optional_tensor("edge_index_a1", "long"),
            edge_weight_a1=optional_tensor("edge_weight_a1", "float"),
            edge_distance_a1_km=edge_distance_a1_km,
            edge_index_a2=optional_tensor("edge_index_a2", "long"),
            edge_weight_a2=optional_tensor("edge_weight_a2", "float"),
            input_feature_indices=feature_idx,
            model_type=args.model_model_type,
            hidden_dim=args.model_hidden_dim,
            gsi_hidden=args.model_gsi_hidden,
            rnn_layers=args.model_rnn_layers,
            gcn_layers=args.model_gcn_layers,
            gat_layers=args.model_gat_layers,
            gat_heads=args.model_gat_heads,
            gat_dropout=args.model_gat_dropout,
            nheads=args.model_nheads,
            dropout=args.model_dropout,
            add_self_loops=args.model_add_self_loops,
            improved=args.model_improved,
            use_temporal_attention=args.model_use_temporal_attention,
            attn_hidden=args.model_attn_hidden,
            lr=args.optim_lr,
            weight_decay=args.optim_weight_decay,
            scheduler=args.scheduler_scheduler,
            t_max=args.scheduler_t_max,
            t_mult=args.scheduler_t_mult,
            eta_min=args.scheduler_eta_min,
            x_transform=args.transform_x_transform,
            y_transform=args.transform_y_transform,
            x_clip_min=args.transform_x_clip_min,
            y_clip_min=args.transform_y_clip_min,
            loss_name=args.loss_name,
            loss_delta=args.loss_delta,
            loss_n_quantiles=args.loss_n_quantiles,
            loss_crossing_weight=args.loss_crossing_weight,
            loss_quantiles=args.loss_quantiles,
            loss_quantile_weights=args.loss_quantile_weights,
            loss_higher_q_weight=args.loss_higher_q_weight,
            t_in_override=args.data_t_in_override,
            adaptive_adj_enabled=args.adaptive_adj_enabled,
            adaptive_adj_mode=args.adaptive_adj_mode,
            gauge_mask_channel=args.model_gauge_mask_channel,
            hide_receiver_gauge_features=args.model_hide_receiver_gauge_features,
            receiver_gauge_feature_channels=args.model_receiver_gauge_feature_channels,
            distance_attention=args.model_distance_attention,
            distance_epsilon_km=args.model_distance_epsilon_km,
            distance_lambda_init=args.model_distance_lambda_init,
            distance_per_head=args.model_distance_per_head,
            distance_lambda_mode=args.model_distance_lambda_mode,
            fusion_mode=args.model_fusion_mode,
            a1_hidden_dim=args.model_a1_hidden_dim,
            a1_feature_channels=args.model_a1_feature_channels,
            aux_feature_channels=args.model_aux_feature_channels,
            aux_hidden_dim=args.model_aux_hidden_dim,
            gcn_intermediate_dim=args.model_gcn_intermediate_dim,
            graph_source=args.graph_source,
            edge_aware_enabled=args.edge_aware_enabled,
            edge_aware_mode=args.edge_aware_mode,
            edge_aware_epsilon_km=args.edge_aware_epsilon_km,
            edge_aware_lambda_init=args.edge_aware_lambda_init,
            edge_aware_per_head=args.edge_aware_per_head,
        )

    def make_trainer(run_name: str, ckpt_root: Path) -> Trainer:
        ckpt_root.mkdir(parents=True, exist_ok=True)

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

        show_progress = _is_progress_worker(args.training_task_progress_worker_index)

        return Trainer(
            max_epochs=args.trainer_max_epochs,
            accelerator=args.trainer_accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
            devices=args.trainer_devices,
            num_nodes=args.trainer_num_nodes,
            strategy=args.trainer_strategy,
            precision=args.trainer_precision,
            log_every_n_steps=args.trainer_log_every_n_steps,
            gradient_clip_val=args.trainer_gradient_clip_val,
            accumulate_grad_batches=args.trainer_accumulate_grad_batches,
            enable_model_summary=False,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_progress_bar=show_progress,
        )

    def resolve_resume_ckpt(ckpt_root: Path) -> Optional[str]:
        ckpt_path = ckpt_root / "last.ckpt"

        if args.trainer_resume:
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"trainer.resume=true, but checkpoint does not exist: {ckpt_path}"
                )
            resume_ckpt = str(ckpt_path)
            rank_zero_info(f"Resuming training from checkpoint: {resume_ckpt}")
            return resume_ckpt

        if ckpt_path.exists():
            rank_zero_info(
                "Checkpoint resume is disabled; starting from a fresh model "
                f"initialization. Existing checkpoint left in place: {ckpt_path}"
            )
        else:
            rank_zero_info(
                "Checkpoint resume is disabled; starting from a fresh model "
                "initialization."
            )
        return None

    def save_best_checkpoint_alias(trainer: Trainer, ckpt_root: Path) -> None:
        checkpoint_cb = trainer.checkpoint_callback
        if checkpoint_cb is None:
            return
        best_model_path = getattr(checkpoint_cb, "best_model_path", "")
        if not best_model_path:
            return
        best_model = Path(best_model_path)
        if not best_model.exists():
            return
        best_alias = ckpt_root / "best.ckpt"
        shutil.copy2(best_model, best_alias)
        rank_zero_info(f"Saved best checkpoint alias: {best_alias}")

    if args.training_task_mode == "global":
        train_data = TGSIPTDataModule(
            seq_dir=args.data_seq_dir,
            batch_size=args.data_batch_size,
            num_workers=args.data_num_workers,
            train_years=tuple(args.data_train_years),
            val_years=_as_optional_year_tuple(args.data_val_years),
            test_years=tuple(args.data_test_years),
            exclude_train_years=_as_year_ranges(args.data_exclude_train_years),
            val_fraction=args.data_val_fraction,
            samples_per_epoch=args.masking_samples_per_epoch,
            context_fraction=args.masking_context_fraction,
            context_fraction_min=args.masking_context_fraction_min,
            context_fraction_max=args.masking_context_fraction_max,
            gauge_value_channel=args.masking_gauge_value_channel,
            gauge_mask_channel=args.masking_gauge_mask_channel,
            min_context_gauges=args.masking_min_context_gauges,
            min_loss_gauges=args.masking_min_loss_gauges,
            seed=args.masking_seed,
            dynamic_masking=args.masking_dynamic_masking,
            masking_mode=args.masking_mode,
            eval_masking_mode=args.masking_eval_mode,
            n_folds=args.masking_n_folds,
        )
        train_data.setup()

        ckpt_root = base_dir / "checkpoints" / args.run_name
        trainer = make_trainer(args.run_name, ckpt_root)
        trainer.fit(
            make_model(),
            datamodule=train_data,
            ckpt_path=resolve_resume_ckpt(ckpt_root),
        )
        save_best_checkpoint_alias(trainer, ckpt_root)
        return

    date_to_file = _index_files_by_date(seq_files)
    all_dates = _resolve_per_day_dates(args, date_to_file)
    all_dates_before_filter = len(all_dates)
    all_dates, skipped_dates = _filter_dates_by_train_precip(
        all_dates,
        date_to_file,
        threshold=args.training_task_min_train_precip_mean,
    )
    if args.training_task_min_train_precip_mean is not None:
        rank_zero_info(
            "[PER_DAY] "
            f"train precipitation mean threshold: > {args.training_task_min_train_precip_mean}. "
            f"Keeping {len(all_dates)} of {all_dates_before_filter} requested dates; "
            f"skipping {len(skipped_dates)}."
        )
        if skipped_dates:
            preview = ", ".join(
                f"{day}:{mean_precip:.4g}" for day, mean_precip in skipped_dates[:10]
            )
            rank_zero_info(f"[PER_DAY] First skipped dates mean precip: {preview}")
    if not all_dates:
        raise RuntimeError(
            "No per-day dates remain after applying "
            f"training_task.min_train_precip_mean={args.training_task_min_train_precip_mean}"
        )

    worker_dates = _worker_slice(
        all_dates,
        worker_index=args.training_task_worker_index,
        num_workers=args.training_task_num_workers,
    )

    rank_zero_info(
        "[PER_DAY] "
        f"worker {args.training_task_worker_index}/{args.training_task_num_workers} "
        f"will train {len(worker_dates)} of {len(all_dates)} requested dates."
    )
    if _is_progress_worker(args.training_task_progress_worker_index):
        print(
            "[PER_DAY_PROGRESS] "
            f"watching worker {args.training_task_worker_index}/{args.training_task_num_workers}; "
            f"assigned_dates={len(worker_dates)}; total_filtered_dates={len(all_dates)}",
            flush=True,
        )

    for day_idx, day in enumerate(worker_dates, start=1):
        day_file = date_to_file[day]
        run_name = _per_day_run_name(args.run_name, day, multi_day=len(all_dates) > 1)
        ckpt_root = base_dir / "checkpoints" / run_name
        remaining_before = len(worker_dates) - day_idx

        rank_zero_info(
            f"[PER_DAY] worker {args.training_task_worker_index}/{args.training_task_num_workers} "
            f"date {day_idx}/{len(worker_dates)}: training date={day} "
            f"file={day_file} run_name={run_name}"
        )
        if _is_progress_worker(args.training_task_progress_worker_index):
            print(
                "[PER_DAY_PROGRESS] "
                f"starting date {day_idx}/{len(worker_dates)} "
                f"({day}); remaining_after_this={remaining_before}",
                flush=True,
            )

        train_ds = TGSIRandomMaskedPTDataset(
            files=[day_file],
            samples_per_epoch=args.masking_samples_per_epoch,
            context_fraction=args.masking_context_fraction,
            context_fraction_min=args.masking_context_fraction_min,
            context_fraction_max=args.masking_context_fraction_max,
            gauge_value_channel=args.masking_gauge_value_channel,
            gauge_mask_channel=args.masking_gauge_mask_channel,
            min_context_gauges=args.masking_min_context_gauges,
            min_loss_gauges=args.masking_min_loss_gauges,
            seed=args.masking_seed,
            dynamic_masking=args.masking_dynamic_masking,
        )
        val_ds = TGSIHoldoutPTDataset(
            files=[day_file],
            gauge_value_channel=args.masking_gauge_value_channel,
            gauge_mask_channel=args.masking_gauge_mask_channel,
            min_context_gauges=args.masking_min_context_gauges,
            min_loss_gauges=args.masking_min_loss_gauges,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.data_batch_size,
            shuffle=False,
            num_workers=args.data_num_workers,
            pin_memory=True,
            collate_fn=collate_tgsi,
            drop_last=True,
            worker_init_fn=seed_tgsi_worker,
            persistent_workers=args.data_num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.data_batch_size,
            shuffle=False,
            num_workers=args.data_num_workers,
            pin_memory=True,
            collate_fn=collate_tgsi,
            drop_last=False,
            persistent_workers=args.data_num_workers > 0,
        )

        trainer = make_trainer(run_name, ckpt_root)
        trainer.fit(
            make_model(),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resolve_resume_ckpt(ckpt_root),
        )
        save_best_checkpoint_alias(trainer, ckpt_root)
        remaining = len(worker_dates) - day_idx
        rank_zero_info(
            f"[PER_DAY] worker {args.training_task_worker_index}/{args.training_task_num_workers} "
            f"completed {day_idx}/{len(worker_dates)} dates; remaining={remaining}; "
            f"last_date={day}"
        )
        if _is_progress_worker(args.training_task_progress_worker_index):
            print(
                "[PER_DAY_PROGRESS] "
                f"completed date {day_idx}/{len(worker_dates)} "
                f"({day}); remaining={remaining}",
                flush=True,
            )


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)

    main()
