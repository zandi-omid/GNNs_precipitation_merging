#!/usr/bin/env python3
# coding: utf-8
"""
Distributed inference + robust merge + optional NetCDF postprocessing
--------------------------------------------------------------------
Stage 1:
  - Reads config from TOML
  - Lists seq_*.pt files
  - Filters to TEST years (by sample["date"])
  - Shards files across SLURM ranks: my_files = all_files[rank::world_size]
  - Each rank runs inference on its shard on its GPU
  - Writes per-rank npz: out_dir/preds_rank{rank:03d}.npz

Stage 2:
  - Rank 0 robustly merges per-rank files into one: out_dir/preds_merged.npz

Stage 3 (optional, rank 0 only):
  - Converts merged npz to daily NetCDF maps using graph + seq files

Outputs (merged NPZ):
  dates: [n_samples]  (YYYY-MM-DD)
  pred_point: [n_samples, N]  (median for quantile model; point for MSE model)
  pred_mean:  [n_samples, N]  (expected value from quantiles; only for quantile model)
  pred_q_sel: [n_samples, N, K] (selected quantiles; only for quantile model)
  tau_sel:    [K]
  pred_q_full:[n_samples, N, Q] (optional; only for quantile model)
  tau_full:   [Q]

Optional TOML section:
[postprocess]
enabled = true
graph_pkl = "/path/to/graph_with_features_labels.pkl"
out_nc = "/path/to/pred_inputs_daily_maps.nc"
save_det = true
save_quantiles = false
no_expected_mean = false
no_median = false
merged_name = "preds_merged.npz"
"""

from __future__ import annotations

import os
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import tomli as tomllib

# Import Lightning module
from training_task import TGCNLightning

# Quantile helpers
from utils.inference.inference_utils import (
    get_tau_full,
    pick_tau_indices,
    quantile_expected_value,
)


# ============================================================
# TOML helpers
# ============================================================
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


# ============================================================
# SLURM rank / GPU binding
# ============================================================
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


# ============================================================
# Generic helpers
# ============================================================
def _load_pt(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ============================================================
# Dataset
# ============================================================
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
        d = _load_pt(p)
        return {"x": d["x"].float(), "date": d["date"]}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)  # [B,T,N,F]
    dates = [b["date"] for b in batch]
    return {"x": x, "date": dates}


# ============================================================
# Transform inversion (targets)
# ============================================================
def inverse_y_transform(y_hat: torch.Tensor, y_transform: str) -> torch.Tensor:
    y_transform = (y_transform or "none").lower()
    if y_transform == "log1p":
        return torch.expm1(y_hat)
    if y_transform == "none":
        return y_hat
    raise ValueError(f"Unknown y_transform: {y_transform}")


# ============================================================
# File listing + filtering by years
# ============================================================
def filter_files_by_years(files: List[Path], years: Tuple[int, int]) -> List[Path]:
    y0, y1 = years
    out: List[Path] = []
    for p in files:
        try:
            d = _load_pt(p)
            date = d.get("date", None)
            if date is None:
                continue
            yr = int(str(date)[:4])
            if y0 <= yr <= y1:
                out.append(p)
        except Exception:
            continue
    return out


# ============================================================
# Inference
# ============================================================
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

    loss_name = (getattr(model, "loss_name", "") or "").lower()
    out_channels = int(getattr(getattr(model, "model", None), "out_channels", 1))
    is_quantile = (loss_name == "quantile") or (out_channels > 1)

    tau_full = get_tau_full(model) if is_quantile else None
    sel_idx = pick_tau_indices(tau_full, tau_sel) if (is_quantile and tau_sel) else None

    all_dates: list[str] = []
    out_point: list[np.ndarray] = []
    out_full: list[np.ndarray] = []
    out_sel: list[np.ndarray] = []
    out_mean: list[np.ndarray] = []

    use_amp = amp and (device.type == "cuda")

    for batch in loader:
        x = batch["x"].to(device)     # [B,T,N,F]
        dates = batch["date"]

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_hat = model(x)
        else:
            y_hat = model(x)

        y_hat = inverse_y_transform(y_hat, y_transform)

        if clamp_min is not None:
            y_hat = torch.clamp(y_hat, min=float(clamp_min))

        all_dates.extend(dates)

        if not is_quantile:
            out_point.append(y_hat.squeeze(-1).detach().cpu().float().numpy())
            continue

        if y_hat.ndim != 3:
            raise ValueError(f"Expected quantile output [B,N,Q], got {tuple(y_hat.shape)}")

        if enforce_monotonic:
            y_hat = torch.sort(y_hat, dim=-1).values

        tau_np = tau_full.numpy()
        mid_idx = int(np.argmin(np.abs(tau_np - 0.5)))
        med = y_hat[:, :, mid_idx]
        out_point.append(med.detach().cpu().float().numpy())

        if save_full_quantiles:
            out_full.append(y_hat.detach().cpu().float().numpy())

        if sel_idx is not None:
            out_sel.append(y_hat[:, :, sel_idx].detach().cpu().float().numpy())

        if compute_expected:
            mean = quantile_expected_value(y_hat, tau_full)
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


# ============================================================
# Robust NPZ merge
# ============================================================
def merge_rank_outputs(out_dir: Path, world: int, merged_name: str = "preds_merged.npz") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_files = [out_dir / f"preds_rank{r:03d}.npz" for r in range(world)]

    while True:
        ready = True
        for p in rank_files:
            if (not p.exists()) or (p.stat().st_size == 0):
                ready = False
                break
        if ready:
            break
        time.sleep(5)

    zs = None
    last_err = None
    for attempt in range(12):
        try:
            zs = [np.load(p, allow_pickle=False) for p in rank_files]
            last_err = None
            break
        except (EOFError, OSError, ValueError) as e:
            last_err = e
            print(
                f"[Rank 0] Waiting for rank outputs to become readable "
                f"(attempt {attempt + 1}/12): {e}",
                flush=True,
            )
            time.sleep(5)

    if zs is None:
        raise RuntimeError(f"Failed to read all rank npz files after retries: {last_err}")

    keys = list(zs[0].files)
    if "dates" not in keys:
        raise ValueError("Per-rank npz must contain 'dates'")

    merged: Dict[str, np.ndarray] = {}
    for k in keys:
        arrs = [z[k] for z in zs]
        if k in ("tau_full", "tau_sel"):
            merged[k] = arrs[0]
        else:
            merged[k] = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]

    dates = merged["dates"]
    order = np.argsort(dates)
    merged["dates"] = dates[order]

    for k, v in list(merged.items()):
        if k in ("dates", "tau_full", "tau_sel"):
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == order.shape[0]:
            merged[k] = v[order]

    merged_path = out_dir / merged_name
    np.savez_compressed(merged_path, **merged)

    print(f"[Rank 0] Merged NPZ written to: {merged_path}", flush=True)
    for k in merged:
        print(f"[Rank 0]   {k}: {merged[k].shape} {merged[k].dtype}", flush=True)

    return merged_path


# ============================================================
# Postprocess helpers
# ============================================================
def _node_list_from_graph(G) -> list[tuple[int, int]]:
    return sorted(G.nodes())


def _make_static_maps(G, node_list, ny, nx):
    lon2d = np.full((ny, nx), np.nan, dtype=np.float32)
    lat2d = np.full((ny, nx), np.nan, dtype=np.float32)
    elev2d = np.full((ny, nx), np.nan, dtype=np.float32)
    valid = np.zeros((ny, nx), dtype=np.uint8)

    for (i, j) in node_list:
        attrs = G.nodes[(i, j)]
        valid[i, j] = 1
        if "lon" in attrs:
            lon2d[i, j] = np.float32(attrs["lon"])
        if "lat" in attrs:
            lat2d[i, j] = np.float32(attrs["lat"])
        if "elevation" in attrs:
            elev2d[i, j] = np.float32(attrs["elevation"])
        elif "elev" in attrs:
            elev2d[i, j] = np.float32(attrs["elev"])

    return lon2d, lat2d, elev2d, valid


def _read_preds_npz(npz_path: Path, tau_sel: Optional[list[float]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    dates = z["dates"].astype(str)
    out: Dict[str, np.ndarray] = {}

    if "preds" in keys:
        out["pred_det"] = z["preds"].astype(np.float32)

    if "pred_point" in keys and "pred_det" not in out:
        out["pred_det"] = z["pred_point"].astype(np.float32)

    if "pred_point" in keys:
        out["pred_median"] = z["pred_point"].astype(np.float32)

    if "pred_mean" in keys:
        out["pred_expected_mean"] = z["pred_mean"].astype(np.float32)

    if "pred_q_sel" in keys and "tau_sel" in keys:
        tau_file = z["tau_sel"].astype(np.float32)
        pred_q = z["pred_q_sel"].astype(np.float32)

        if tau_sel is None or len(tau_sel) == 0:
            out["tau"] = tau_file
            out["pred_q"] = pred_q
        else:
            tau_req = np.array(tau_sel, dtype=np.float32)
            idx = np.array([int(np.argmin(np.abs(tau_file - t))) for t in tau_req], dtype=int)
            out["tau"] = tau_file[idx]
            out["pred_q"] = pred_q[:, :, idx]

    return dates, out


def _load_inputs_from_seq(seq_dir: Path, dates: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    seq_files = sorted(seq_dir.glob("seq_*.pt"))
    if not seq_files:
        raise RuntimeError(f"No seq_*.pt found in: {seq_dir}")

    date_to_file: Dict[str, Path] = {}
    for f in seq_files:
        d = _load_pt(f)
        date_to_file[str(d["date"])] = f

    T = len(dates)
    imerg_node_np = np.full((T, N), np.nan, dtype=np.float32)
    era5_node_np = np.full((T, N), np.nan, dtype=np.float32)

    for t, day in enumerate(dates):
        f = date_to_file.get(str(day), None)
        if f is None:
            continue
        d = _load_pt(f)
        x = d["x"].numpy().astype(np.float32)   # [Twin, N, F]
        x_last = x[-1]                          # [N, F]
        era5_node_np[t, :] = x_last[:, 0]
        imerg_node_np[t, :] = x_last[:, 1]

    return era5_node_np, imerg_node_np


def _map_nodes_to_grid(node_list, ny, nx, node_arr: np.ndarray) -> np.ndarray:
    T, N = node_arr.shape
    out = np.full((T, ny, nx), np.nan, dtype=np.float32)
    for k, (i, j) in enumerate(node_list):
        out[:, i, j] = node_arr[:, k]
    return out


def _map_nodes_to_grid_q(node_list, ny, nx, node_arr_q: np.ndarray) -> np.ndarray:
    T, N, K = node_arr_q.shape
    out = np.full((T, ny, nx, K), np.nan, dtype=np.float32)
    for k_node, (i, j) in enumerate(node_list):
        out[:, i, j, :] = node_arr_q[:, k_node, :]
    return out


def _build_gauge_maps(G, node_list, dates: np.ndarray, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(dates)
    gauge_mask = np.zeros((T, ny, nx), dtype=np.uint8)
    gauge_map = np.full((T, ny, nx), np.nan, dtype=np.float32)

    date_to_t = {d: t for t, d in enumerate(pd.to_datetime(dates).strftime("%Y-%m-%d"))}

    for (i, j) in node_list:
        tdict = G.nodes[(i, j)].get("target", None)
        if not tdict:
            continue
        for day_str, val in tdict.items():
            t = date_to_t.get(day_str, None)
            if t is None:
                continue
            gauge_mask[t, i, j] = 1
            gauge_map[t, i, j] = np.float32(val)

    return gauge_map, gauge_mask


def npz_to_nc(
    graph_pkl: Path,
    pred_npz: Path,
    seq_dir: Path,
    out_nc: Path,
    tau_sel: Optional[list[float]] = None,
    save_quantiles: bool = False,
    no_expected_mean: bool = False,
    no_median: bool = False,
    save_det: bool = False,
) -> Path:
    with open(graph_pkl, "rb") as f:
        payload = pickle.load(f)
    G = payload["graph"]

    node_list = _node_list_from_graph(G)
    ny = max(i for i, _ in node_list) + 1
    nx = max(j for _, j in node_list) + 1
    N = len(node_list)

    lon2d, lat2d, elev2d, valid = _make_static_maps(G, node_list, ny, nx)

    dates_str, pred_pack = _read_preds_npz(pred_npz, tau_sel=tau_sel)
    T = len(dates_str)

    def _check_N(arr, name):
        if arr is None:
            return
        if arr.shape[1] != N:
            raise RuntimeError(
                f"Mismatch: {name} has N={arr.shape[1]} but graph has N={N}. "
                "Node ordering differs. Use exact common_nodes ordering from seq builder."
            )

    if "pred_det" in pred_pack:
        _check_N(pred_pack["pred_det"], "pred_det")
    if "pred_median" in pred_pack:
        _check_N(pred_pack["pred_median"], "pred_median")
    if "pred_expected_mean" in pred_pack:
        _check_N(pred_pack["pred_expected_mean"], "pred_expected_mean")
    if "pred_q" in pred_pack and pred_pack["pred_q"].shape[1] != N:
        raise RuntimeError(f"Mismatch: pred_q has N={pred_pack['pred_q'].shape[1]} but graph has N={N}.")

    time_coord = pd.to_datetime(dates_str).to_numpy()

    era5_node_np, imerg_node_np = _load_inputs_from_seq(seq_dir, dates_str, N)
    era5_map = _map_nodes_to_grid(node_list, ny, nx, era5_node_np)
    imerg_map = _map_nodes_to_grid(node_list, ny, nx, imerg_node_np)

    data_vars = {
        "imerg": (("time", "y", "x"), imerg_map),
        "era5": (("time", "y", "x"), era5_map),
        "valid_pixel": (("y", "x"), valid),
        "elevation": (("y", "x"), elev2d),
        "lon": (("y", "x"), lon2d),
        "lat": (("y", "x"), lat2d),
    }

    if "pred_median" in pred_pack and not no_median:
        data_vars["pred_median"] = (
            ("time", "y", "x"),
            _map_nodes_to_grid(node_list, ny, nx, pred_pack["pred_median"]),
        )

    if "pred_expected_mean" in pred_pack and not no_expected_mean:
        data_vars["pred_expected_mean"] = (
            ("time", "y", "x"),
            _map_nodes_to_grid(node_list, ny, nx, pred_pack["pred_expected_mean"]),
        )

    if "pred_det" in pred_pack and save_det:
        data_vars["pred_det"] = (
            ("time", "y", "x"),
            _map_nodes_to_grid(node_list, ny, nx, pred_pack["pred_det"]),
        )

    if save_quantiles and ("pred_q" in pred_pack) and ("tau" in pred_pack):
        data_vars["pred_q"] = (
            ("time", "y", "x", "tau"),
            _map_nodes_to_grid_q(node_list, ny, nx, pred_pack["pred_q"]),
        )

    gauge_map, gauge_mask = _build_gauge_maps(G, node_list, dates_str, ny, nx)
    data_vars["gauge"] = (("time", "y", "x"), gauge_map)
    data_vars["gauge_mask"] = (("time", "y", "x"), gauge_mask)

    coords = {
        "time": time_coord,
        "y": np.arange(ny, dtype=np.int32),
        "x": np.arange(nx, dtype=np.int32),
    }
    if save_quantiles and ("tau" in pred_pack):
        coords["tau"] = pred_pack["tau"].astype(np.float32)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "description": "Daily maps: TGCN predictions + IMERG/ERA5 + gauge mapped onto DEM grid",
            "units_pred": "mm/day",
            "units_imerg": "mm/day",
            "units_era5": "mm/day",
            "units_gauge": "mm/day",
            "pred_npz": str(pred_npz),
            "graph_pkl": str(graph_pkl),
            "seq_dir": str(seq_dir),
        },
    )

    enc = {
        "imerg": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "era5": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "gauge": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "gauge_mask": {"zlib": True, "complevel": 4, "dtype": "uint8", "chunksizes": (1, ny, nx)},
        "valid_pixel": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "elevation": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lon": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    for name in ["pred_median", "pred_expected_mean", "pred_det"]:
        if name in ds.data_vars:
            enc[name] = {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)}

    if "pred_q" in ds.data_vars:
        enc["pred_q"] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (1, ny, nx, len(ds["tau"])),
        }

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc, encoding=enc)

    print(f"[Rank 0] Wrote NetCDF: {out_nc}", flush=True)
    print(f"[Rank 0]   time: {dates_str[0]} -> {dates_str[-1]} (T={T})", flush=True)
    print(f"[Rank 0]   grid: ny={ny}, nx={nx}, nodes={N}", flush=True)
    print(f"[Rank 0]   gauge points total (mask sum): {int(gauge_mask.sum())}", flush=True)

    if "pred_median" in ds.data_vars:
        print(f"[Rank 0]   pred_median: {ds['pred_median'].shape}", flush=True)
    if "pred_expected_mean" in ds.data_vars:
        print(f"[Rank 0]   pred_expected_mean: {ds['pred_expected_mean'].shape}", flush=True)
    if "pred_det" in ds.data_vars:
        print(f"[Rank 0]   pred_det: {ds['pred_det'].shape}", flush=True)
    if "pred_q" in ds.data_vars:
        print(f"[Rank 0]   pred_q: {ds['pred_q'].shape} tau: {ds['tau'].values}", flush=True)

    return out_nc


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")

    parser.add_argument(
        "--tau-sel",
        type=float,
        nargs="+",
        default=None,
        help="Selected quantiles to save. If omitted, uses TOML [inference].tau_sel.",
    )

    parser.add_argument("--save-full-quantiles", action="store_true", help="Save full Q quantiles (pred_q_full).")
    parser.add_argument("--no-save-full-quantiles", action="store_true", help="Do NOT save full Q quantiles.")
    parser.add_argument("--no-expected", action="store_true", help="Do NOT compute expected value (pred_mean).")
    parser.add_argument("--no-monotonic", action="store_true", help="Do NOT enforce monotonic quantiles.")

    args = parser.parse_args()

    cfg = flatten_config(load_toml_config(args.config))

    # Required inference config
    seq_dir = Path(cfg["seq_dir"])
    ckpt = Path(cfg["ckpt"])
    out_dir = Path(cfg.get("out_dir", seq_dir / "inference_outputs"))
    test_years = _as_year_tuple(cfg.get("test_years", [2020, 2024]), (2020, 2024))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 4))
    amp = bool(cfg.get("amp", True))
    y_transform = str(cfg.get("y_transform", "none"))
    clamp_min = float(cfg.get("clamp_min", 0.0))

    # Inference options
    tau_sel = args.tau_sel if args.tau_sel is not None else cfg.get(
        "tau_sel", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )

    if args.save_full_quantiles and args.no_save_full_quantiles:
        raise ValueError("Choose only one of --save-full-quantiles or --no-save-full-quantiles")

    save_full_quantiles = bool(cfg.get("save_full_quantiles", False))
    if args.save_full_quantiles:
        save_full_quantiles = True
    if args.no_save_full_quantiles:
        save_full_quantiles = False

    compute_expected = bool(cfg.get("compute_expected", True))
    if args.no_expected:
        compute_expected = False

    enforce_monotonic = bool(cfg.get("enforce_monotonic", True))
    if args.no_monotonic:
        enforce_monotonic = False

    # Postprocess options
    do_postprocess = bool(cfg.get("enabled", False))
    graph_pkl = cfg.get("graph_pkl", None)
    out_nc = cfg.get("out_nc", None)
    merged_name = str(cfg.get("merged_name", "preds_merged.npz"))
    save_det_post = bool(cfg.get("save_det", False))
    save_quantiles_post = bool(cfg.get("save_quantiles", False))
    no_expected_mean_post = bool(cfg.get("no_expected_mean", False))
    no_median_post = bool(cfg.get("no_median", False))

    if do_postprocess:
        if graph_pkl is None:
            raise ValueError("[postprocess].graph_pkl is required when [postprocess].enabled=true")
        if out_nc is None:
            raise ValueError("[postprocess].out_nc is required when [postprocess].enabled=true")

    rank, world = get_rank_and_world()
    device = get_device_for_rank(rank)

    torch.set_num_threads(1)

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
                    "postprocess_enabled": do_postprocess,
                    "graph_pkl": graph_pkl,
                    "out_nc": out_nc,
                    "save_det_post": save_det_post,
                    "save_quantiles_post": save_quantiles_post,
                    "no_expected_mean_post": no_expected_mean_post,
                    "no_median_post": no_median_post,
                    "merged_name": merged_name,
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
        }
        np.savez_compressed(empty_path, **empty)

        if rank == 0:
            time.sleep(10)
            merged = merge_rank_outputs(out_dir, world, merged_name=merged_name)
            print(f"[Rank 0] Saved merged predictions to: {merged}", flush=True)
            if do_postprocess:
                npz_to_nc(
                    graph_pkl=Path(graph_pkl),
                    pred_npz=merged,
                    seq_dir=seq_dir,
                    out_nc=Path(out_nc),
                    tau_sel=tau_sel,
                    save_quantiles=save_quantiles_post,
                    no_expected_mean=no_expected_mean_post,
                    no_median=no_median_post,
                    save_det=save_det_post,
                )
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

    # 4) load model
    first_file = sorted(Path(seq_dir).glob("seq_*.pt"))[0]
    sample = _load_pt(first_file)

    edge_index = sample["edge_index"].long()
    edge_weight = sample.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    _, N, _F = sample["x"].shape

    model = TGCNLightning.load_from_checkpoint(
        str(ckpt),
        map_location="cpu",
        strict=False,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=N,
    )

    print(
        f"[Rank {rank}] Loaded checkpoint. "
        f"input_feature_indices={getattr(model, 'input_feature_indices', None)} "
        f"in_channels={model.model.backbone.in_channels}",
        flush=True,
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
        save_full_quantiles=save_full_quantiles,
        compute_expected=compute_expected,
        enforce_monotonic=enforce_monotonic,
    )

    # 6) write per-rank output
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_out = out_dir / f"preds_rank{rank:03d}.npz"
    np.savez_compressed(rank_out, **out)

    msg = f"[Rank {rank}] Wrote: {rank_out} dates={out['dates'].shape} point={out['pred_point'].shape}"
    if "pred_mean" in out:
        msg += f" mean={out['pred_mean'].shape}"
    if "pred_q_sel" in out:
        msg += f" q_sel={out['pred_q_sel'].shape}"
    if "pred_q_full" in out:
        msg += f" q_full={out['pred_q_full'].shape}"
    print(msg, flush=True)

    # 7) merge on rank 0 + optional nc conversion
    if rank == 0:
        time.sleep(10)
        merged = merge_rank_outputs(out_dir, world, merged_name=merged_name)
        print(f"[Rank 0] Saved merged predictions to: {merged}", flush=True)

        if do_postprocess:
            npz_to_nc(
                graph_pkl=Path(graph_pkl),
                pred_npz=merged,
                seq_dir=seq_dir,
                out_nc=Path(out_nc),
                tau_sel=tau_sel,
                save_quantiles=save_quantiles_post,
                no_expected_mean=no_expected_mean_post,
                no_median=no_median_post,
                save_det=save_det_post,
            )


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)
    main()