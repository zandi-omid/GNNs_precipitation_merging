#!/usr/bin/env python3
# coding: utf-8
"""
Distributed inference for trained TGCN model (SLURM multi-rank sharding)
----------------------------------------------------------------------
- Reads config from TOML
- Lists seq_*.pt files
- Filters to TEST years (by sample["date"])
- Shards files across SLURM ranks: my_files = all_files[rank::world_size]
- Each rank runs inference on its shard on its GPU
- Writes per-rank npz: out_dir/preds_rank{rank:03d}.npz
- Rank 0 merges per-rank files into one: out_dir/preds_merged.npz (sorted by date)

Outputs (merged):
  dates: [n_samples]  (YYYY-MM-DD)
  pred_point: [n_samples, N]  (median for quantile model; point for MSE model)
  pred_mean:  [n_samples, N]  (expected value from quantiles; only for quantile model)
  pred_q_sel: [n_samples, N, K] (selected quantiles; only for quantile model)
  tau_sel:    [K]
  pred_q_full:[n_samples, N, Q] (optional; only for quantile model)
  tau_full:   [Q]
"""

from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import tomli as tomllib

# Import your Lightning module
from training_task import TGCNLightning

# Quantile helpers you placed in:
# gnn_precipitation_retrieval/utils/inference/inference_utils.py
from utils.inference.inference_utils import (
    get_tau_full,
    pick_tau_indices,
    quantile_expected_value,
)

# -------------------------------
# TOML helpers (same style as training)
# -------------------------------
def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten TOML sections into one dict of keys.
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
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"Expected [y0, y1] for years, got: {v}")

# -------------------------------
# SLURM rank / GPU binding
# -------------------------------
def get_rank_and_world() -> Tuple[int, int]:
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    return rank, world

def get_device_for_rank(rank: int) -> torch.device:
    n_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(1, n_gpus if n_gpus > 0 else 1)))
    if torch.cuda.is_available() and n_gpus > 0:
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

# -------------------------------
# Dataset
# -------------------------------
class SeqPTDataset(Dataset):
    """
    Loads seq_*.pt files saved as dicts:
      x: [T,N,F], y: [N], y_mask: [N], date: str, ...
    Returns:
      dict with x: [T,N,F], date: str
    """
    def __init__(self, files: List[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.files[idx]
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            d = torch.load(p, map_location="cpu")
        return {"x": d["x"].float(), "date": d["date"]}

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)  # [B,T,N,F]
    dates = [b["date"] for b in batch]
    return {"x": x, "date": dates}

# -------------------------------
# Transform inversion (targets)
# -------------------------------
def inverse_y_transform(y_hat: torch.Tensor, y_transform: str) -> torch.Tensor:
    y_transform = (y_transform or "none").lower()
    if y_transform == "log1p":
        return torch.expm1(y_hat)
    if y_transform == "none":
        return y_hat
    raise ValueError(f"Unknown y_transform: {y_transform}")

# -------------------------------
# File listing + filtering by years
# -------------------------------
def filter_files_by_years(files: List[Path], years: Tuple[int, int]) -> List[Path]:
    y0, y1 = years
    out: List[Path] = []
    for p in files:
        try:
            try:
                d = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                d = torch.load(p, map_location="cpu")
            date = d.get("date", None)
            if date is None:
                continue
            yr = int(str(date)[:4])
            if y0 <= yr <= y1:
                out.append(p)
        except Exception:
            continue
    return out

# -------------------------------
# Inference
# -------------------------------
@torch.no_grad()
def run_inference(
    model: TGCNLightning,
    loader: DataLoader,
    device: torch.device,
    *,
    y_transform: str,
    clamp_min: float = 0.0,
    amp: bool = True,
    tau_sel: Optional[list[float]] = None,
    save_full_quantiles: bool = True,
    compute_expected: bool = True,
    enforce_monotonic: bool = True,
) -> Dict[str, np.ndarray]:

    model.eval()
    model.to(device)

    # Detect whether this is a quantile model
    # (loss_name saved in hparams, and/or out_channels > 1)
    loss_name = (getattr(model, "loss_name", "") or "").lower()
    out_channels = int(getattr(getattr(model, "model", None), "out_channels", 1))
    is_quantile = (loss_name == "quantile") or (out_channels > 1)

    # Quantile grid from model (should match training: make_quantiles(Q))
    tau_full = get_tau_full(model) if is_quantile else None  # torch.Tensor [Q] on CPU
    sel_idx = pick_tau_indices(tau_full, tau_sel) if (is_quantile and tau_sel) else None

    all_dates: list[str] = []
    out_point: list[np.ndarray] = []
    out_full: list[np.ndarray] = []
    out_sel: list[np.ndarray] = []
    out_mean: list[np.ndarray] = []

    use_amp = amp and (device.type == "cuda")

    for batch in loader:
        x = batch["x"].to(device)     # [B,T,N,F]
        dates = batch["date"]         # list[str]

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_hat = model(x)
        else:
            y_hat = model(x)

        # y_hat is either [B,N,1] or [B,N,Q]
        y_hat = inverse_y_transform(y_hat, y_transform)

        if clamp_min is not None:
            y_hat = torch.clamp(y_hat, min=float(clamp_min))

        all_dates.extend(dates)

        if not is_quantile:
            # [B,N,1] -> [B,N]
            out_point.append(y_hat.squeeze(-1).detach().cpu().float().numpy())
            continue

        # Quantile model: ensure shape [B,N,Q]
        if y_hat.ndim != 3:
            raise ValueError(f"Expected quantile output [B,N,Q], got {tuple(y_hat.shape)}")

        if enforce_monotonic:
            # Sort across Q to enforce non-crossing at inference time
            y_hat = torch.sort(y_hat, dim=-1).values

        # median index = closest tau to 0.5
        tau_np = tau_full.numpy()
        mid_idx = int(np.argmin(np.abs(tau_np - 0.5)))
        med = y_hat[:, :, mid_idx]  # [B,N]
        out_point.append(med.detach().cpu().float().numpy())

        if save_full_quantiles:
            out_full.append(y_hat.detach().cpu().float().numpy())

        if sel_idx is not None:
            out_sel.append(y_hat[:, :, sel_idx].detach().cpu().float().numpy())

        if compute_expected:
            mean = quantile_expected_value(y_hat, tau_full)  # [B,N]
            out_mean.append(mean.detach().cpu().float().numpy())

    result: Dict[str, np.ndarray] = {
        "dates": np.array(all_dates, dtype="U10"),
        "pred_point": np.concatenate(out_point, axis=0) if out_point else np.empty((0, 0), np.float32),
    }

    if is_quantile:
        result["tau_full"] = tau_full.numpy().astype(np.float32)

        if save_full_quantiles:
            result["pred_q_full"] = (
                np.concatenate(out_full, axis=0) if out_full else np.empty((0, 0, 0), np.float32)
            )

        if sel_idx is not None:
            result["tau_sel"] = np.array(tau_sel, dtype=np.float32)
            result["pred_q_sel"] = (
                np.concatenate(out_sel, axis=0) if out_sel else np.empty((0, 0, 0), np.float32)
            )

        if compute_expected:
            result["pred_mean"] = (
                np.concatenate(out_mean, axis=0) if out_mean else np.empty((0, 0), np.float32)
            )

    return result

# -------------------------------
# Merge helper
# -------------------------------
def merge_rank_outputs(out_dir: Path, world: int, merged_name: str = "preds_merged.npz") -> Path:
    """
    Rank 0 waits for all per-rank files to exist, then merges and sorts by date.
    Supports arbitrary prediction keys (pred_point, pred_mean, pred_q_sel, pred_q_full, etc).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_files = [out_dir / f"preds_rank{r:03d}.npz" for r in range(world)]

    # Wait until all exist
    while True:
        missing = [p for p in rank_files if not p.exists()]
        if not missing:
            break
        time.sleep(10)

    # Load all rank outputs
    zs = [np.load(p, allow_pickle=False) for p in rank_files]

    # Determine keys (assume all ranks save the same keys)
    keys = list(zs[0].files)
    if "dates" not in keys:
        raise ValueError("Per-rank npz must contain 'dates'")

    # Concatenate per key
    merged: Dict[str, np.ndarray] = {}
    for k in keys:
        arrs = [z[k] for z in zs]
        if k in ("tau_full", "tau_sel"):
            # identical across ranks (keep rank0)
            merged[k] = arrs[0]
        else:
            merged[k] = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]

    # Sort by date string (and apply same order to arrays with first dim = n_samples)
    dates = merged["dates"]
    order = np.argsort(dates)
    merged["dates"] = dates[order]

    for k, v in list(merged.items()):
        if k in ("dates", "tau_full", "tau_sel"):
            continue
        # Sort arrays that are sample-aligned
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == order.shape[0]:
            merged[k] = v[order]

    merged_path = out_dir / merged_name
    np.savez_compressed(merged_path, **merged)
    return merged_path

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")

    # NEW: CLI override for selected quantiles
    parser.add_argument(
        "--tau-sel",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        help="Selected quantiles to save (only used for quantile models).",
    )

    # Optional convenience flags
    parser.add_argument("--save-full-quantiles", action="store_true", help="Save full Q quantiles (pred_q_full).")
    parser.add_argument("--no-save-full-quantiles", action="store_true", help="Do NOT save full Q quantiles.")
    parser.add_argument("--no-expected", action="store_true", help="Do NOT compute expected value (pred_mean).")
    parser.add_argument("--no-monotonic", action="store_true", help="Do NOT enforce monotonic quantiles (sorting).")

    args = parser.parse_args()

    cfg = flatten_config(load_toml_config(args.config))

    # Required
    seq_dir = Path(cfg["seq_dir"])
    ckpt = Path(cfg["ckpt"])
    out_dir = Path(cfg.get("out_dir", seq_dir / "inference_outputs"))

    # Inference
    tau_sel = cfg.get("tau_sel", None)
    save_full_q = bool(cfg.get("save_full_quantiles", True))
    compute_expected = bool(cfg.get("compute_expected", True))
    enforce_monotonic = bool(cfg.get("enforce_monotonic", True))

    # Split
    test_years = _as_year_tuple(cfg.get("test_years", [2020, 2024]), (2020, 2024))

    # Compute
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 4))
    amp = bool(cfg.get("amp", True))

    # Transforms (must match training)
    y_transform = str(cfg.get("y_transform", "none"))
    clamp_min = float(cfg.get("clamp_min", 0.0))

    rank, world = get_rank_and_world()
    device = get_device_for_rank(rank)

    # Avoid CPU oversubscription
    torch.set_num_threads(1)

    # Resolve tau selection: CLI overrides TOML if provided
    tau_sel = args.tau_sel if args.tau_sel is not None else cfg.get("tau_sel", None)

    # Resolve saving options
    if args.save_full_quantiles and args.no_save_full_quantiles:
        raise ValueError("Choose only one of --save-full-quantiles or --no-save-full-quantiles")
    save_full_quantiles = bool(cfg.get("save_full_quantiles", False))
    if args.save_full_quantiles:
        save_full_quantiles = True
    if args.no_save_full_quantiles:
        save_full_quantiles = False

    compute_expected = not args.no_expected
    enforce_monotonic = not args.no_monotonic

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "inference_resolved_config.json", "w") as f:
            json.dump(
                {
                    "seq_dir": str(seq_dir),
                    "ckpt": str(ckpt),
                    "out_dir": str(out_dir),
                    "test_years": list(test_years),
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "amp": amp,
                    "y_transform": y_transform,
                    "clamp_min": clamp_min,
                    "world_size": world,
                    "tau_sel": tau_sel,
                    "save_full_quantiles": save_full_quantiles,
                    "compute_expected": compute_expected,
                    "enforce_monotonic": enforce_monotonic,
                },
                f,
                indent=2,
            )

    # 1) list + filter
    all_files = sorted(seq_dir.glob("seq_*.pt"))
    if not all_files:
        raise RuntimeError(f"No seq_*.pt files found in: {seq_dir}")

    test_files = filter_files_by_years(all_files, test_years)

    # 2) shard
    my_files = test_files[rank::world]

    print(
        f"[Rank {rank}/{world}] device={device} | total_seq={len(all_files)} "
        f"| test_seq={len(test_files)} | my_seq={len(my_files)}",
        flush=True,
    )

    if len(my_files) == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        empty_path = out_dir / f"preds_rank{rank:03d}.npz"
        empty = {
            "dates": np.array([], dtype="U10"),
            "pred_point": np.empty((0, 0), dtype=np.float32),
            "pred_mean": np.empty((0, 0), dtype=np.float32),
            "pred_q_sel": np.empty((0, 0, 0), dtype=np.float32),
            "tau_sel": np.array(tau_sel if tau_sel is not None else [], dtype=np.float32),
        }
        if save_full_quantiles:
            empty["pred_q_full"] = np.empty((0, 0, 0), dtype=np.float32)
            # tau_full unknown here; rank0 will still write it from its own inference
        np.savez_compressed(empty_path, **empty)

        if rank == 0:
            merged = merge_rank_outputs(out_dir, world)
            print(f"[Rank 0] ✅ Saved merged predictions to: {merged}", flush=True)
        return

    # 3) loader
    ds = SeqPTDataset(my_files)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 4) load model (+ graph buffers)
    first_file = sorted(Path(seq_dir).glob("seq_*.pt"))[0]
    try:
        sample = torch.load(first_file, map_location="cpu", weights_only=True)
    except TypeError:
        sample = torch.load(first_file, map_location="cpu")

    edge_index = sample["edge_index"].long()
    edge_weight = sample.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    _, N, _F = sample["x"].shape  # x: [T, N, F]

    model = TGCNLightning.load_from_checkpoint(
        str(ckpt),
        map_location="cpu",
        strict=False,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=N,
    )

    # 5) run inference
    out = run_inference(
        model=model,
        loader=loader,
        device=device,
        y_transform=y_transform,
        clamp_min=clamp_min,
        amp=amp,
        tau_sel=tau_sel,
        save_full_quantiles=save_full_q,
        compute_expected=compute_expected,
        enforce_monotonic=enforce_monotonic,
    )

    # 6) write per-rank output
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_out = out_dir / f"preds_rank{rank:03d}.npz"
    np.savez_compressed(rank_out, **out)

    msg = f"[Rank {rank}] ✅ Wrote: {rank_out} dates={out['dates'].shape} point={out['pred_point'].shape}"
    if "pred_mean" in out:
        msg += f" mean={out['pred_mean'].shape}"
    if "pred_q_sel" in out:
        msg += f" q_sel={out['pred_q_sel'].shape}"
    if "pred_q_full" in out:
        msg += f" q_full={out['pred_q_full'].shape}"
    print(msg, flush=True)

    # 7) merge on rank 0
    if rank == 0:
        merged = merge_rank_outputs(out_dir, world)
        print(f"[Rank 0] ✅ Saved merged predictions to: {merged}", flush=True)

if __name__ == "__main__":
    # avoid SLURM env confusion if running interactively
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)
    main()