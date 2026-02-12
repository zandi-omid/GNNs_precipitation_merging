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
  preds: [n_samples, N]  (mm/day if y_transform inverted)
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
# SLURM rank / GPU binding (same logic as your orbit script)
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
        # weights_only exists on newer torch; keep it robust
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            d = torch.load(p, map_location="cpu")
        # We only need x + date for inference
        return {"x": d["x"].float(), "date": d["date"]}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # x: stack to [B,T,N,F]
    x = torch.stack([b["x"] for b in batch], dim=0)
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
            # Skip corrupted/unreadable samples
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
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_dates: List[str] = []
    all_preds: List[np.ndarray] = []

    use_amp = amp and (device.type == "cuda")

    for batch in loader:
        x = batch["x"].to(device)          # [B,T,N,F]
        dates = batch["date"]              # list[str], length B

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_hat = model(x)           # [B,N] (your forward should apply x_transform)
        else:
            y_hat = model(x)

        # invert y transform to mm/day
        y_hat = inverse_y_transform(y_hat, y_transform)

        # physical clamp
        if clamp_min is not None:
            y_hat = torch.clamp(y_hat, min=float(clamp_min))

        all_dates.extend(dates)
        all_preds.append(y_hat.detach().cpu().float().numpy())

    preds = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, 0), dtype=np.float32)
    dates = np.array(all_dates, dtype="U10")
    return dates, preds


# -------------------------------
# Merge helper
# -------------------------------
def merge_rank_outputs(out_dir: Path, world: int, merged_name: str = "preds_merged.npz") -> Path:
    """
    Rank 0 waits for all per-rank files to exist, then merges and sorts by date.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_files = [out_dir / f"preds_rank{r:03d}.npz" for r in range(world)]

    # Wait until all exist
    while True:
        missing = [p for p in rank_files if not p.exists()]
        if not missing:
            break
        time.sleep(10)

    all_dates = []
    all_preds = []

    for p in rank_files:
        z = np.load(p, allow_pickle=False)
        all_dates.append(z["dates"])
        all_preds.append(z["preds"])

    dates = np.concatenate(all_dates, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    # Sort by date string
    order = np.argsort(dates)
    dates = dates[order]
    preds = preds[order]

    merged_path = out_dir / merged_name
    np.savez_compressed(merged_path, dates=dates, preds=preds)
    return merged_path


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    args = parser.parse_args()

    cfg = flatten_config(load_toml_config(args.config))

    # Required
    seq_dir = Path(cfg["seq_dir"])
    ckpt = Path(cfg["ckpt"])
    out_dir = Path(cfg.get("out_dir", seq_dir / "inference_outputs"))

    # Split
    test_years = _as_year_tuple(cfg.get("test_years", [2020, 2024]), (2020, 2024))

    # Compute
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 4))
    amp = bool(cfg.get("amp", True))

    # Transforms (must match training)
    y_transform = str(cfg.get("y_transform", "log1p"))
    clamp_min = float(cfg.get("clamp_min", 0.0))

    rank, world = get_rank_and_world()
    device = get_device_for_rank(rank)

    # Avoid CPU oversubscription
    torch.set_num_threads(1)

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        # dump resolved config for reproducibility
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
                },
                f,
                indent=2,
            )

    # 1) list + filter
    all_files = sorted(seq_dir.glob("seq_*.pt"))
    if not all_files:
        raise RuntimeError(f"No seq_*.pt files found in: {seq_dir}")

    test_files = filter_files_by_years(all_files, test_years)

    # 2) shard like orbit script
    my_files = test_files[rank::world]

    print(
        f"[Rank {rank}/{world}] device={device} | total_seq={len(all_files)} "
        f"| test_seq={len(test_files)} | my_seq={len(my_files)}",
        flush=True,
    )

    if len(my_files) == 0:
        # still participate so rank 0 can merge cleanly
        out_dir.mkdir(parents=True, exist_ok=True)
        empty_path = out_dir / f"preds_rank{rank:03d}.npz"
        np.savez_compressed(empty_path, dates=np.array([], dtype="U10"), preds=np.empty((0, 0), dtype=np.float32))
        if rank == 0:
            merged = merge_rank_outputs(out_dir, world)
            print(f"[Rank 0] ✅ Saved merged predictions to: {merged}", flush=True)
        return

    # 3) dataset/loader for this shard
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

    # 4) load model
    # --- get graph from any seq_*.pt (same graph for all samples) ---
    first_file = sorted(Path(seq_dir).glob("seq_*.pt"))[0]
    sample = torch.load(first_file, map_location="cpu", weights_only=True)

    edge_index = sample["edge_index"].long()
    edge_weight = sample.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    _, N, F = sample["x"].shape  # x: [T, N, F]

    model = TGCNLightning.load_from_checkpoint(
        str(ckpt),
        map_location="cpu",
        strict=False,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=N,
    )

    # 5) run inference
    dates, preds = run_inference(
        model=model,
        loader=loader,
        device=device,
        y_transform=y_transform,
        clamp_min=clamp_min,
        amp=amp,
    )

    # 6) write per-rank output
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_out = out_dir / f"preds_rank{rank:03d}.npz"
    np.savez_compressed(rank_out, dates=dates, preds=preds)
    print(f"[Rank {rank}] ✅ Wrote: {rank_out}  dates={dates.shape} preds={preds.shape}", flush=True)

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