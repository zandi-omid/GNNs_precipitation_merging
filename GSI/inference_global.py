#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import zipfile
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tomli as tomllib
import xarray as xr
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from train import SparseGaugeLightning
from utils.data.tgsi_seq_datamodule import TGSIHoldoutPTDataset, collate_tgsi


class AllGaugePTDataset(Dataset):
    """Load full all-gauge samples; fold masking is applied during inference."""

    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = _load_pt(self.files[idx])
        for key in ("gauge_fold_id", "train_available_mask", "eval_mask"):
            if key not in sample:
                raise KeyError(f"Cross-fit inference requires {key!r} in {self.files[idx]}")
        available = sample["train_available_mask"].bool() & torch.isfinite(sample["y"])
        out = dict(sample)
        out["context_mask"] = available
        out["loss_mask"] = sample["eval_mask"].bool() & torch.isfinite(sample["y"])
        out["train_mask"] = out["loss_mask"]
        return out


def load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for section, kv in cfg.items():
        if not isinstance(kv, dict):
            out[section] = kv
            continue
        if section == "meta" and "run_name" in kv:
            out["run_name"] = kv["run_name"]
            continue
        for k, v in kv.items():
            out[f"{section}_{k}"] = v
    return out


def get_rank_world() -> Tuple[int, int]:
    return int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))


def get_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 0:
            local_rank = int(os.environ.get("SLURM_LOCALID", rank % n))
            return torch.device(f"cuda:{local_rank % n}")
    return torch.device("cpu")


def _load_pt(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_iso_date(value: str) -> date_cls:
    try:
        return date_cls.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Expected date in YYYY-MM-DD format, got {value!r}") from exc


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


def _index_files_by_date(seq_dir: Path) -> Dict[str, Path]:
    out = {}
    for fp in sorted(seq_dir.glob("seq_*.pt")):
        s = _load_pt(fp)
        out[str(s["date"])] = fp
    if not out:
        raise FileNotFoundError(f"No seq_*.pt files found in {seq_dir}")
    return out


def load_node_coords(graph_path: Path) -> np.ndarray:
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)
    graph = payload["graph"]
    common_nodes = payload["common_nodes"]
    return np.array(
        [(graph.nodes[n]["lon"], graph.nodes[n]["lat"]) for n in common_nodes],
        dtype=np.float32,
    )


def approximate_lonlat_to_km(coords_lonlat: np.ndarray) -> np.ndarray:
    mean_lat = np.deg2rad(np.nanmean(coords_lonlat[:, 1]))
    coords_km = np.empty_like(coords_lonlat, dtype=np.float64)
    coords_km[:, 0] = coords_lonlat[:, 0] * 111.0 * np.cos(mean_lat)
    coords_km[:, 1] = coords_lonlat[:, 1] * 111.0
    return coords_km


def edge_distance_a1_from_graph(
    graph_path: Path,
    edge_index_a1: torch.Tensor,
) -> torch.Tensor:
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)

    if "edge_distance_a1_km" in payload:
        return torch.as_tensor(payload["edge_distance_a1_km"], dtype=torch.float32)

    graph = payload["graph"]
    common_nodes = payload["common_nodes"]
    coords_lonlat = np.array(
        [(graph.nodes[n]["lon"], graph.nodes[n]["lat"]) for n in common_nodes],
        dtype=np.float64,
    )
    coords_km = approximate_lonlat_to_km(coords_lonlat)
    edge_np = edge_index_a1.cpu().numpy()
    recv = edge_np[0].astype(np.int64)
    send = edge_np[1].astype(np.int64)
    distance = np.linalg.norm(coords_km[send] - coords_km[recv], axis=1)
    return torch.from_numpy(distance.astype(np.float32))


def resolve_dates(cfg: Dict[str, Any], date_to_file: Dict[str, Path]) -> List[str]:
    raw_dates = cfg.get("dates")
    raw_date = cfg.get("date")
    if raw_dates:
        raw_dates = [str(d) for d in raw_dates]
        if len(raw_dates) == 2:
            requested = _date_range(raw_dates[0], raw_dates[1])
        else:
            requested = raw_dates
    elif raw_date:
        requested = [str(raw_date)]
    else:
        raise ValueError("Provide [dates].date or [dates].dates.")

    missing = [d for d in requested if d not in date_to_file]
    if missing:
        available = sorted(date_to_file)
        raise FileNotFoundError(
            f"{len(missing)} requested dates have no sequence file. "
            f"First missing dates: {', '.join(missing[:10])}. "
            f"Available sequence date range is {available[0]} to {available[-1]}."
        )
    return requested


def compute_idw_all_nodes(
    node_coords: np.ndarray,
    donor_idx: np.ndarray,
    donor_vals: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    if len(donor_idx) == 0:
        return np.zeros(node_coords.shape[0], dtype=np.float32)
    donor_coords = node_coords[donor_idx].astype(np.float64)
    donor_vals = donor_vals.astype(np.float32)
    k_eff = min(int(k), len(donor_vals))
    tree = cKDTree(donor_coords)
    dists, inds = tree.query(node_coords.astype(np.float64), k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        inds = inds[:, None]
    eps = 1e-6
    if float(power) == 0.0:
        out = donor_vals[inds].mean(axis=1)
    else:
        w = 1.0 / np.maximum(dists, eps) ** float(power)
        vals = donor_vals[inds]
        out = (w * vals).sum(axis=1) / np.maximum(w.sum(axis=1), eps)
        exact = dists <= eps
        has_exact = exact.any(axis=1)
        if np.any(has_exact):
            first_exact = exact.argmax(axis=1)
            exact_vals = vals[np.arange(node_coords.shape[0]), first_exact]
            out[has_exact] = exact_vals[has_exact]
    return out.astype(np.float32)


def get_quantiles(model: SparseGaugeLightning, q_dim: int) -> np.ndarray:
    q = getattr(model, "quantiles", None)
    if isinstance(q, torch.Tensor) and q.numel() == q_dim:
        return q.detach().cpu().numpy().astype(np.float32)
    hp_q = getattr(model.hparams, "loss_quantiles", None)
    if hp_q is not None and len(hp_q) == q_dim:
        return np.asarray(hp_q, dtype=np.float32)
    return (np.arange(1, q_dim + 1, dtype=np.float32) / (q_dim + 1)).astype(np.float32)


def inverse_y_transform(y: torch.Tensor, y_transform: str) -> torch.Tensor:
    y_transform = (y_transform or "none").lower()
    if y_transform == "none":
        return y
    if y_transform == "log1p":
        return torch.expm1(y)
    raise ValueError(f"Unknown y_transform: {y_transform}")


def optional_tensor(sample: Dict[str, Any], key: str, dtype_method: str):
    value = sample.get(key)
    if value is None:
        return None
    return getattr(value, dtype_method)()


def load_model(
    ckpt: Path,
    sample: Dict[str, Any],
    graph_path: Path,
    device: torch.device,
    hide_receiver_gauge_features: bool = False,
    receiver_gauge_feature_channels: Optional[list[int]] = None,
    gauge_mask_channel: int = 1,
) -> SparseGaugeLightning:
    if not ckpt.exists():
        raise FileNotFoundError(f"Global inference checkpoint not found: {ckpt}")
    edge_distance_a1_km = optional_tensor(sample, "edge_distance_a1_km", "float")
    edge_index_a1 = optional_tensor(sample, "edge_index_a1", "long")
    if edge_distance_a1_km is None and edge_index_a1 is not None:
        edge_distance_a1_km = edge_distance_a1_from_graph(graph_path, edge_index_a1)
    model = SparseGaugeLightning.load_from_checkpoint(
        str(ckpt),
        map_location="cpu",
        strict=True,
        edge_index_a1=edge_index_a1,
        edge_weight_a1=optional_tensor(sample, "edge_weight_a1", "float"),
        edge_distance_a1_km=edge_distance_a1_km,
        edge_index_a2=optional_tensor(sample, "edge_index_a2", "long"),
        edge_weight_a2=optional_tensor(sample, "edge_weight_a2", "float"),
    )
    if hide_receiver_gauge_features:
        if receiver_gauge_feature_channels is None:
            receiver_gauge_feature_channels = [0, int(gauge_mask_channel)]
        _set_receiver_gauge_feature_channels(
            model,
            receiver_gauge_feature_channels,
        )
    model.eval().to(device)
    return model


def _set_receiver_gauge_feature_channels(
    model: SparseGaugeLightning,
    channels: list[int],
) -> None:
    channels = [int(c) for c in channels]
    wrapped = getattr(model, "model", None)
    if wrapped is None:
        raise AttributeError("Loaded Lightning model does not have a .model attribute.")
    if hasattr(wrapped, "receiver_gauge_feature_channels"):
        wrapped.receiver_gauge_feature_channels = channels
        return
    spatial = getattr(wrapped, "spatial", None)
    if spatial is not None and hasattr(spatial, "receiver_gauge_feature_channels"):
        spatial.receiver_gauge_feature_channels = channels
        return
    raise AttributeError(
        "This checkpoint model does not support receiver gauge feature masking. "
        f"Loaded nested model type: {type(wrapped).__name__}"
    )


def batch_idw(
    batch: Dict[str, Any],
    node_coords: np.ndarray,
    idw_k: int,
    idw_power: float,
) -> np.ndarray:
    coords = node_coords.astype(np.float64)
    y = batch["y"].numpy().astype(np.float32)
    context_mask = batch["context_mask"].numpy().astype(bool)
    out = []
    for b in range(y.shape[0]):
        donor_ok = context_mask[b] & np.isfinite(y[b])
        donor_idx = np.where(donor_ok)[0].astype(np.int64)
        donor_vals = y[b, donor_idx].astype(np.float32)
        out.append(
            compute_idw_all_nodes(
                node_coords=coords,
                donor_idx=donor_idx,
                donor_vals=donor_vals,
                k=idw_k,
                power=idw_power,
            )
        )
    return np.stack(out, axis=0).astype(np.float32)


@torch.no_grad()
def run_inference(
    model: SparseGaugeLightning,
    loader: DataLoader,
    device: torch.device,
    y_transform: str,
    clamp_min: Optional[float],
    amp: bool,
    node_coords: np.ndarray,
    idw_k: int,
    idw_power: float,
    crossfit_folds: int = 0,
    gauge_value_channel: int = 0,
    gauge_mask_channel: int = 1,
) -> Dict[str, np.ndarray]:
    use_amp = bool(amp) and device.type == "cuda"
    dates = []
    preds = []
    preds_q = []
    idws = []
    train_gauges = []
    train_masks = []
    context_gauges = []
    context_masks = []
    holdout_gauges = []
    holdout_masks = []
    era5 = []
    imerg = []
    train_precip_mean = []
    tau: Optional[np.ndarray] = None

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        fold_outputs = []
        if crossfit_folds:
            fold_id = batch["gauge_fold_id"].to(device).long()
            available = batch["train_available_mask"].to(device).bool()
            for held_fold in range(crossfit_folds):
                x_fold = x.clone()
                hidden = available & (fold_id == held_fold)
                x_fold[:, :, :, gauge_value_channel].masked_fill_(hidden[:, None, :], 0.0)
                x_fold[:, :, :, gauge_mask_channel].masked_fill_(hidden[:, None, :], 0.0)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        fold_outputs.append(model(x_fold))
                else:
                    fold_outputs.append(model(x_fold))
            y_hat = torch.stack(fold_outputs).mean(dim=0)
            for held_fold, fold_out in enumerate(fold_outputs):
                targets = available & (fold_id == held_fold)
                y_hat[targets] = fold_out[targets]
        elif use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_hat = model(x)
        else:
            y_hat = model(x)
        y_hat = inverse_y_transform(y_hat, y_transform)
        if clamp_min is not None:
            y_hat = torch.clamp(y_hat, min=float(clamp_min))
        if y_hat.ndim != 3:
            raise ValueError(f"Expected model output [B,N,C], got {tuple(y_hat.shape)}")

        if y_hat.shape[-1] == 1:
            pred = y_hat[:, :, 0]
        else:
            q = get_quantiles(model, y_hat.shape[-1])
            if tau is None:
                tau = q
                q50_idx = int(np.argmin(np.abs(tau - 0.5)))
                print(
                    f"[INFERENCE] Quantile output detected: Q={len(tau)}, "
                    f"q50_idx={q50_idx}, q50={float(tau[q50_idx]):.6f}",
                    flush=True,
                )
            pred = y_hat[:, :, q50_idx]
            preds_q.append(y_hat.detach().cpu().float().numpy())

        y_np = batch["y"].numpy().astype(np.float32)
        context_mask = batch["context_mask"].numpy().astype(bool)
        loss_mask = batch["loss_mask"].numpy().astype(bool)

        train_gauge = np.full_like(y_np, np.nan, dtype=np.float32)
        holdout_gauge = np.full_like(y_np, np.nan, dtype=np.float32)
        train_gauge[context_mask] = y_np[context_mask]
        holdout_gauge[loss_mask] = y_np[loss_mask]

        dates.extend(batch["date"])
        preds.append(pred.detach().cpu().float().numpy())
        if crossfit_folds:
            fold_id_np = batch["gauge_fold_id"].numpy().astype(np.int64)
            available_np = batch["train_available_mask"].numpy().astype(bool)
            idw_batch = []
            for b in range(y_np.shape[0]):
                fold_maps = []
                for held_fold in range(crossfit_folds):
                    donors = available_np[b] & (fold_id_np[b] != held_fold) & np.isfinite(y_np[b])
                    fold_maps.append(compute_idw_all_nodes(
                        node_coords, np.where(donors)[0], y_np[b, donors], idw_k, idw_power
                    ))
                combined = np.mean(fold_maps, axis=0).astype(np.float32)
                for held_fold, fold_map in enumerate(fold_maps):
                    targets = available_np[b] & (fold_id_np[b] == held_fold)
                    combined[targets] = fold_map[targets]
                idw_batch.append(combined)
            idws.append(np.stack(idw_batch))
        else:
            idws.append(batch_idw(batch, node_coords=node_coords, idw_k=idw_k, idw_power=idw_power))
        train_gauges.append(train_gauge)
        train_masks.append(context_mask.astype(np.uint8))
        context_gauges.append(train_gauge)
        context_masks.append(context_mask.astype(np.uint8))
        holdout_gauges.append(holdout_gauge)
        holdout_masks.append(loss_mask.astype(np.uint8))
        x_last = batch["x"][:, -1].numpy().astype(np.float32)
        era5.append(x_last[:, :, 2])
        imerg.append(x_last[:, :, 3])
        train_precip_mean.append(
            np.array(
                [
                    float(y_np[b, context_mask[b]].mean()) if context_mask[b].any() else 0.0
                    for b in range(y_np.shape[0])
                ],
                dtype=np.float32,
            )
        )

        if batch_idx == 1 or batch_idx % 10 == 0:
            print(f"[INFERENCE] batches_done={batch_idx}", flush=True)

    out = {
        "dates": np.array(dates, dtype="U10"),
        "pred": np.concatenate(preds, axis=0).astype(np.float32),
        "idw": np.concatenate(idws, axis=0).astype(np.float32),
        "train_gauge": np.concatenate(train_gauges, axis=0).astype(np.float32),
        "train_mask": np.concatenate(train_masks, axis=0).astype(np.uint8),
        "context_gauge": np.concatenate(context_gauges, axis=0).astype(np.float32),
        "context_mask": np.concatenate(context_masks, axis=0).astype(np.uint8),
        "holdout_gauge": np.concatenate(holdout_gauges, axis=0).astype(np.float32),
        "holdout_mask": np.concatenate(holdout_masks, axis=0).astype(np.uint8),
        "era5": np.concatenate(era5, axis=0).astype(np.float32),
        "imerg": np.concatenate(imerg, axis=0).astype(np.float32),
        "model_available": np.ones((len(dates),), dtype=np.uint8),
        "train_precip_mean": np.concatenate(train_precip_mean, axis=0).astype(np.float32),
    }
    if preds_q:
        out["pred_quantile"] = np.concatenate(preds_q, axis=0).astype(np.float32)
        out["quantile"] = tau.astype(np.float32)
    return out


def npz_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def merge_rank_outputs(out_dir: Path, world: int, merged_name: str) -> Path:
    rank_files = [out_dir / f"preds_rank{r:03d}.npz" for r in range(world)]
    while True:
        ready = [npz_ready(p) for p in rank_files]
        if all(ready):
            break
        missing = [p.name for p, ok in zip(rank_files, ready) if not ok]
        print(f"[Rank 0] Waiting for {len(missing)} rank files: {missing[:5]}", flush=True)
        time.sleep(10)

    zs = [np.load(p, allow_pickle=False) for p in rank_files]
    keys = list(zs[0].files)
    merged = {}
    for key in keys:
        if key == "quantile":
            merged[key] = zs[0][key]
        else:
            merged[key] = np.concatenate([z[key] for z in zs], axis=0)
    order = np.argsort(merged["dates"])
    for key in list(merged):
        if key != "quantile":
            merged[key] = merged[key][order]

    out = out_dir / merged_name
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **merged)
    if not npz_ready(tmp):
        raise RuntimeError(f"Merged NPZ failed archive validation: {tmp}")
    os.replace(tmp, out)
    for z in zs:
        z.close()
    print(f"[Rank 0] Wrote merged NPZ: {out}", flush=True)
    for key, val in merged.items():
        print(f"[Rank 0] {key}: {val.shape} {val.dtype}", flush=True)
    return out


def make_netcdf(
    npz_path: Path,
    out_nc: Path,
    seq_dir: Path,
    node_coords: np.ndarray,
) -> None:
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}

    dates = arrays["dates"].astype(str)
    time_coord = pd.DatetimeIndex(pd.to_datetime(dates)).values.astype("datetime64[ns]")
    node_coords = node_coords.astype(np.float32)
    N = node_coords.shape[0]
    coords = {
        "time": ("time", time_coord),
        "node": ("node", np.arange(N, dtype=np.int32)),
    }
    if "quantile" in arrays:
        coords["quantile"] = ("quantile", arrays["quantile"].astype(np.float32))

    data_vars = {
        "pred_gsi": (("time", "node"), arrays["pred"].astype(np.float32)),
        "idw_train": (("time", "node"), arrays["idw"].astype(np.float32)),
        "era5": (("time", "node"), arrays["era5"].astype(np.float32)),
        "imerg": (("time", "node"), arrays["imerg"].astype(np.float32)),
        "train_gauge": (("time", "node"), arrays["train_gauge"].astype(np.float32)),
        "train_mask": (("time", "node"), arrays["train_mask"].astype(np.uint8)),
        "context_gauge": (("time", "node"), arrays["context_gauge"].astype(np.float32)),
        "context_mask": (("time", "node"), arrays["context_mask"].astype(np.uint8)),
        "holdout_gauge": (("time", "node"), arrays["holdout_gauge"].astype(np.float32)),
        "holdout_mask": (("time", "node"), arrays["holdout_mask"].astype(np.uint8)),
        "model_available": (("time",), arrays["model_available"].astype(np.uint8).reshape(-1)),
        "train_precip_mean": (("time",), arrays["train_precip_mean"].astype(np.float32).reshape(-1)),
        "node_lon": (("node",), node_coords[:, 0].astype(np.float32)),
        "node_lat": (("node",), node_coords[:, 1].astype(np.float32)),
    }
    if "pred_quantile" in arrays:
        data_vars["pred_quantile"] = (
            ("time", "node", "quantile"),
            arrays["pred_quantile"].astype(np.float32),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "description": "GSI global checkpoint inference on TGSI holdout split.",
            "units": "mm/day",
            "npz": str(npz_path),
            "seq_dir": str(seq_dir),
        },
    )
    enc = {}
    for v in ds.data_vars:
        dtype = str(ds[v].dtype)
        if ds[v].ndim == 3:
            enc[v] = {"zlib": True, "complevel": 4, "dtype": dtype, "chunksizes": (1, N, ds.sizes["quantile"])}
        elif ds[v].ndim == 2:
            enc[v] = {"zlib": True, "complevel": 4, "dtype": dtype, "chunksizes": (1, N)}
        else:
            enc[v] = {"zlib": True, "complevel": 4, "dtype": dtype}
    enc["time"] = {
        "units": "days since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
    }
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    tmp_nc = out_nc.with_suffix(out_nc.suffix + f".tmp.{os.getpid()}")
    if tmp_nc.exists():
        tmp_nc.unlink()
    ds.to_netcdf(tmp_nc, encoding=enc)
    os.replace(tmp_nc, out_nc)
    print(f"[Rank 0] Wrote NetCDF: {out_nc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = flatten_config(load_toml(args.config))

    seq_dir = Path(cfg["data_seq_dir"])
    graph_path = Path(cfg["data_graph_path"])
    ckpt = Path(cfg["data_ckpt"])
    out_dir = Path(cfg["output_out_dir"])
    out_nc = Path(cfg["output_out_nc"])
    merged_name = str(cfg.get("output_merged_name", "preds_merged.npz"))
    save_nc = bool(cfg.get("output_save_nc", True))

    batch_size = int(cfg.get("data_batch_size", 64))
    num_workers = int(cfg.get("data_num_workers", 4))
    gauge_value_channel = int(cfg.get("data_gauge_value_channel", 0))
    gauge_mask_channel = int(cfg.get("data_gauge_mask_channel", 1))
    hide_receiver_gauge_features = bool(cfg.get("model_hide_receiver_gauge_features", False))
    receiver_gauge_feature_channels = cfg.get("model_receiver_gauge_feature_channels")
    if receiver_gauge_feature_channels is not None:
        receiver_gauge_feature_channels = [
            int(c) for c in receiver_gauge_feature_channels
        ]
    min_context_gauges = int(cfg.get("data_min_context_gauges", 5))
    min_loss_gauges = int(cfg.get("data_min_loss_gauges", 5))
    y_transform = str(cfg.get("transforms_y_transform", "none"))
    clamp_min = cfg.get("transforms_clamp_min", 0.0)
    amp = bool(cfg.get("compute_amp", True))
    idw_k = int(cfg.get("idw_k", 10))
    idw_power = float(cfg.get("idw_power", 2.0))
    crossfit_folds = int(cfg.get("crossfit_n_folds", 0)) if bool(cfg.get("crossfit_enabled", False)) else 0

    rank, world = get_rank_world()
    device = get_device(rank)
    torch.set_num_threads(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out_dir / "resolved_config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    date_to_file = _index_files_by_date(seq_dir)
    dates = resolve_dates(
        {"dates": cfg.get("dates_dates"), "date": cfg.get("dates_date")},
        date_to_file,
    )
    files = [date_to_file[d] for d in dates]
    my_indices = list(range(rank, len(files), world))
    my_files = [files[i] for i in my_indices]
    if not my_files:
        print(f"[Rank {rank}/{world}] no assigned dates; writing empty output.", flush=True)

    sample = _load_pt(files[0])
    node_coords = load_node_coords(graph_path)
    model = load_model(
        ckpt=ckpt,
        sample=sample,
        graph_path=graph_path,
        device=device,
        hide_receiver_gauge_features=hide_receiver_gauge_features,
        receiver_gauge_feature_channels=receiver_gauge_feature_channels,
        gauge_mask_channel=gauge_mask_channel,
    )
    checkpoint_t_in = getattr(model.hparams, "t_in_override", None)
    print(
        f"[Rank {rank}/{world}] device={device} total_dates={len(files)} mine={len(my_files)} "
        f"ckpt={ckpt} model_type={getattr(model, 'model_type', None)} "
        f"t_in_override={checkpoint_t_in} "
        f"hide_receiver_gauge_features={hide_receiver_gauge_features}",
        flush=True,
    )

    dataset = AllGaugePTDataset(my_files) if crossfit_folds else TGSIHoldoutPTDataset(
        files=my_files,
        gauge_value_channel=gauge_value_channel,
        gauge_mask_channel=gauge_mask_channel,
        min_context_gauges=min_context_gauges,
        min_loss_gauges=min_loss_gauges,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_tgsi,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    out = run_inference(
        model=model,
        loader=loader,
        device=device,
        y_transform=y_transform,
        clamp_min=clamp_min,
        amp=amp,
        node_coords=node_coords,
        idw_k=idw_k,
        idw_power=idw_power,
        crossfit_folds=crossfit_folds,
        gauge_value_channel=gauge_value_channel,
        gauge_mask_channel=gauge_mask_channel,
    )
    rank_out = out_dir / f"preds_rank{rank:03d}.npz"
    tmp_out = out_dir / f"preds_rank{rank:03d}.tmp"
    with open(tmp_out, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp_out, rank_out)
    print(
        f"[Rank {rank}] wrote {rank_out} dates={out['dates'].shape} pred={out['pred'].shape}",
        flush=True,
    )

    if rank == 0:
        merged = merge_rank_outputs(out_dir, world, merged_name=merged_name)
        if save_nc:
            make_netcdf(
                merged,
                out_nc=out_nc,
                seq_dir=seq_dir,
                node_coords=node_coords,
            )


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)
    main()
