#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import time
import zipfile
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tomli as tomllib
import xarray as xr
from scipy.spatial import cKDTree

from build_df import (
    _a1_gauge_features_for_targets,
    _empty_neighbor_features,
    _neighbor_feature_columns,
    _to_numpy,
)
import train_MLP as train_mlp_module
from train_MLP import (
    MLPLightning,
    _compact_date,
)


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


def load_mlp_checkpoint_quietly(ckpt_file: Path) -> MLPLightning:
    # MLPLightning prints the full quantile list during __init__. That is useful
    # for training, but per-day inference loads ~1,500 checkpoints and the logs
    # become noisy enough to obscure real failures.
    sink = io.StringIO()
    old_rank_zero_info = train_mlp_module.rank_zero_info
    train_mlp_module.rank_zero_info = lambda *args, **kwargs: None
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return MLPLightning.load_from_checkpoint(str(ckpt_file), map_location="cpu")
    finally:
        train_mlp_module.rank_zero_info = old_rank_zero_info


def _load_pt(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _index_files_by_date(seq_dir: Path) -> Dict[str, Path]:
    out = {}
    for fp in sorted(seq_dir.glob("seq_*.pt")):
        s = _load_pt(fp)
        out[str(s["date"])] = fp
    if not out:
        raise FileNotFoundError(f"No seq_*.pt files found in {seq_dir}")
    return out


def resolve_dates(cfg: Dict[str, Any], date_to_file: Dict[str, Path]) -> List[str]:
    raw_dates = cfg.get("dates")
    raw_date = cfg.get("date")
    if raw_dates:
        raw_dates = [str(d) for d in raw_dates]
        if len(raw_dates) == 2:
            requested = _date_range(raw_dates[0], raw_dates[1])
            dates = [d for d in requested if d in date_to_file]
            missing = len(requested) - len(dates)
            if missing:
                available = sorted(date_to_file)
                print(
                    f"[MLP_INFER] Skipping {missing} missing dates from requested range "
                    f"{raw_dates[0]} to {raw_dates[1]}. Available sequence range is "
                    f"{available[0]} to {available[-1]}.",
                    flush=True,
                )
        else:
            dates = raw_dates
    elif raw_date:
        dates = [str(raw_date)]
    else:
        raise ValueError("Provide [dates].date or [dates].dates.")

    missing = [d for d in dates if d not in date_to_file]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} requested dates have no sequence file. "
            f"First missing dates: {', '.join(missing[:10])}"
        )
    if not dates:
        raise FileNotFoundError("No requested dates have sequence files.")
    return dates


def load_node_coords_from_graph(graph_path: Path) -> np.ndarray:
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)
    graph = payload["graph"]
    nodes = payload.get("common_nodes", list(graph.nodes))
    return np.asarray(
        [(float(graph.nodes[n]["lon"]), float(graph.nodes[n]["lat"])) for n in nodes],
        dtype=np.float32,
    )


def sample_node_coords(sample: Dict[str, Any], graph_path: Optional[Path]) -> np.ndarray:
    if "node_coords" in sample:
        return _to_numpy(sample["node_coords"]).astype(np.float32)
    if graph_path is None:
        raise KeyError("Sequence sample has no node_coords; provide [data].graph_path.")
    return load_node_coords_from_graph(graph_path)


def train_precip_mean(sample: Dict[str, Any]) -> float:
    y = sample["y"].float()
    mask = sample["train_available_mask"].bool() & torch.isfinite(y)
    if int(mask.sum()) == 0:
        return 0.0
    return float(y[mask].mean().item())


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
        if np.any(exact.any(axis=1)):
            first_exact = exact.argmax(axis=1)
            exact_vals = vals[np.arange(node_coords.shape[0]), first_exact]
            out[exact.any(axis=1)] = exact_vals[exact.any(axis=1)]
    return out.astype(np.float32)


def checkpoint_for_day(ckpt_root: Path, day: str, checkpoint_name: str) -> Path:
    return ckpt_root / _compact_date(day) / checkpoint_name


def quantiles_from_model(model: MLPLightning, default_n: int) -> np.ndarray:
    q = getattr(model, "quantiles", None)
    if isinstance(q, torch.Tensor) and q.numel() > 0:
        return q.detach().cpu().numpy().astype(np.float32)
    q_default = np.arange(1, int(default_n) + 1, dtype=np.float32)
    return q_default / (len(q_default) + 1)


def make_zero_output(
    day: str,
    sample: Dict[str, Any],
    node_coords: np.ndarray,
    q: np.ndarray,
    idw: np.ndarray,
    mean_train: float,
    era5_idx: int,
    imerg_idx: int,
) -> Dict[str, np.ndarray]:
    y = sample["y"].float().numpy().astype(np.float32)
    train_mask = sample["train_available_mask"].bool().numpy()
    eval_mask = sample["eval_mask"].bool().numpy()
    train_gauge = np.full_like(y, np.nan, dtype=np.float32)
    holdout_gauge = np.full_like(y, np.nan, dtype=np.float32)
    train_gauge[train_mask] = y[train_mask]
    holdout_gauge[eval_mask] = y[eval_mask]
    x_last = _to_numpy(sample["x"])[-1].astype(np.float32)
    n_nodes = int(y.shape[0])
    return {
        "date": np.array(day, dtype="U10"),
        "pred": np.zeros(n_nodes, dtype=np.float32),
        "pred_quantile": np.zeros((n_nodes, len(q)), dtype=np.float32),
        "idw": idw.astype(np.float32),
        "train_gauge": train_gauge,
        "train_mask": train_mask.astype(np.uint8),
        "holdout_gauge": holdout_gauge,
        "holdout_mask": eval_mask.astype(np.uint8),
        "era5": x_last[:, era5_idx].astype(np.float32),
        "imerg": x_last[:, imerg_idx].astype(np.float32),
        "train_precip_mean": np.array(mean_train, dtype=np.float32),
        "model_available": np.array(0, dtype=np.uint8),
    }


def build_all_node_features(
    sample: Dict[str, Any],
    node_coords: np.ndarray,
    feature_cols: List[str],
    k_gauges: int,
    idw_k: int,
    idw_power: float,
    era5_idx: int,
    imerg_idx: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x = _to_numpy(sample["x"]).astype(np.float32)
    y = _to_numpy(sample["y"]).astype(np.float32)
    target_idx = np.arange(y.shape[0], dtype=np.int64)
    x_last = x[-1]
    aux_rows = {
        "ERA5": x_last[:, era5_idx].astype(np.float32),
        "IMERG": x_last[:, imerg_idx].astype(np.float32),
    }
    if set(feature_cols).issubset(aux_rows):
        # Auxiliary-only inference must not derive any predictor from gauges.
        return (
            pd.DataFrame(aux_rows)[feature_cols],
            np.full(len(target_idx), np.nan, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    train_mask = _to_numpy(sample["train_available_mask"]).astype(bool) & np.isfinite(y)
    donor_idx = np.where(train_mask)[0].astype(np.int64)
    donor_vals = y[donor_idx].astype(np.float32)

    idw_train = compute_idw_all_nodes(
        node_coords=node_coords,
        donor_idx=donor_idx,
        donor_vals=donor_vals,
        k=idw_k,
        power=idw_power,
    )

    if len(donor_idx) == 0:
        neighbor_features = [_empty_neighbor_features(k_gauges) for _ in target_idx]
        idw_loo = np.full(len(target_idx), np.nan, dtype=np.float32)
    else:
        edge_index_a1 = _to_numpy(sample["edge_index_a1"]).astype(np.int64)
        neighbor_features, idw_loo, _ = _a1_gauge_features_for_targets(
            node_coords=node_coords.astype(np.float64),
            donor_idx=donor_idx,
            donor_vals=donor_vals,
            target_idx=target_idx,
            edge_index_a1=edge_index_a1,
            k_gauges=k_gauges,
            idw_k=idw_k,
            idw_power=idw_power,
        )

    rows = {
        **aux_rows,
    }
    rows["idw_loo"] = idw_loo.astype(np.float32)
    for col in _neighbor_feature_columns(k_gauges):
        rows[col] = np.asarray([nf[col] for nf in neighbor_features], dtype=np.float32)
    df_features = pd.DataFrame(rows)
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"Requested feature columns are unavailable: {missing}")
    return df_features[feature_cols], idw_train, donor_idx, donor_vals


@torch.no_grad()
def predict_nodes(
    model: MLPLightning,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    model.eval().to(device)
    preds = []
    use_amp = bool(amp) and device.type == "cuda"
    for start in range(0, len(features), int(batch_size)):
        xb = torch.from_numpy(features[start:start + int(batch_size)]).to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yb = model(xb)
        else:
            yb = model(xb)
        preds.append(yb.detach().cpu().float().numpy())
    out = np.concatenate(preds, axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(out, 0.0).astype(np.float32)


def infer_one_day(
    day: str,
    seq_file: Path,
    ckpt_file: Path,
    df_all: pd.DataFrame,
    node_coords_ref: np.ndarray,
    q_default: np.ndarray,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    sample = _load_pt(seq_file)
    node_coords = sample_node_coords(sample, cfg.get("graph_path"))
    if node_coords.shape != node_coords_ref.shape or not np.allclose(node_coords, node_coords_ref):
        raise ValueError(f"Node coordinates changed in {seq_file}")

    feature_cols = cfg["feature_cols"]
    mean_train = train_precip_mean(sample)
    features_df, idw_train, _, _ = build_all_node_features(
        sample=sample,
        node_coords=node_coords,
        feature_cols=feature_cols,
        k_gauges=int(cfg["k_gauges"]),
        idw_k=int(cfg["idw_k"]),
        idw_power=float(cfg["idw_power"]),
        era5_idx=int(cfg["era5_idx"]),
        imerg_idx=int(cfg["imerg_idx"]),
    )

    if (
        cfg["min_train_precip_mean"] is not None
        and mean_train <= float(cfg["min_train_precip_mean"])
    ):
        return make_zero_output(
            day, sample, node_coords, q_default, idw_train, mean_train,
            int(cfg["era5_idx"]), int(cfg["imerg_idx"]),
        )

    if not ckpt_file.exists():
        if bool(cfg["strict_checkpoints"]):
            raise FileNotFoundError(f"Missing per-day checkpoint for {day}: {ckpt_file}")
        return make_zero_output(
            day, sample, node_coords, q_default, idw_train, mean_train,
            int(cfg["era5_idx"]), int(cfg["imerg_idx"]),
        )

    features = features_df.to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    model = load_mlp_checkpoint_quietly(ckpt_file)
    q = quantiles_from_model(model, len(q_default))
    y_hat = predict_nodes(
        model=model,
        features=features,
        device=device,
        batch_size=int(cfg["node_batch_size"]),
        amp=bool(cfg["amp"]),
    )
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if y_hat.ndim == 2:
        pred_q = y_hat.astype(np.float32)
        q50_idx = int(np.argmin(np.abs(q - 0.5)))
        pred = pred_q[:, q50_idx].astype(np.float32)
    else:
        pred = y_hat.astype(np.float32)
        pred_q = pred[:, None]

    y = sample["y"].float().numpy().astype(np.float32)
    train_mask = sample["train_available_mask"].bool().numpy()
    eval_mask = sample["eval_mask"].bool().numpy()
    train_gauge = np.full_like(y, np.nan, dtype=np.float32)
    holdout_gauge = np.full_like(y, np.nan, dtype=np.float32)
    train_gauge[train_mask] = y[train_mask]
    holdout_gauge[eval_mask] = y[eval_mask]
    x_last = _to_numpy(sample["x"])[-1].astype(np.float32)

    return {
        "date": np.array(day, dtype="U10"),
        "pred": pred,
        "pred_quantile": pred_q,
        "idw": idw_train.astype(np.float32),
        "train_gauge": train_gauge,
        "train_mask": train_mask.astype(np.uint8),
        "holdout_gauge": holdout_gauge,
        "holdout_mask": eval_mask.astype(np.uint8),
        "era5": x_last[:, int(cfg["era5_idx"])].astype(np.float32),
        "imerg": x_last[:, int(cfg["imerg_idx"])].astype(np.float32),
        "train_precip_mean": np.array(mean_train, dtype=np.float32),
        "model_available": np.array(1, dtype=np.uint8),
    }


def stack_outputs(items: List[Dict[str, np.ndarray]], q_default: np.ndarray) -> Dict[str, np.ndarray]:
    if not items:
        return {"dates": np.array([], dtype="U10"), "quantile": q_default.astype(np.float32)}
    keys = [
        "pred", "pred_quantile", "idw", "train_gauge", "train_mask",
        "holdout_gauge", "holdout_mask", "era5", "imerg",
        "train_precip_mean", "model_available",
    ]
    out = {"dates": np.array([str(x["date"]) for x in items], dtype="U10")}
    for key in keys:
        out[key] = np.stack([x[key] for x in items], axis=0)
    out["quantile"] = q_default.astype(np.float32)
    return out


def npz_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def merge_rank_outputs(
    out_dir: Path,
    world: int,
    merged_name: str,
    return_arrays: bool = False,
) -> Path | Tuple[Path, Dict[str, np.ndarray]]:
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
    if return_arrays:
        return out, merged
    return out


def make_netcdf_from_arrays(
    arrays: Dict[str, np.ndarray],
    out_nc: Path,
    node_coords: np.ndarray,
    seq_dir: Path,
    ckpt_root: Path,
    npz_path: Optional[Path] = None,
) -> None:
    dates = arrays["dates"].astype(str)
    time_coord = pd.DatetimeIndex(pd.to_datetime(dates)).values.astype("datetime64[ns]")
    n_nodes = node_coords.shape[0]
    coords = {
        "time": ("time", time_coord),
        "node": ("node", np.arange(n_nodes, dtype=np.int32)),
        "quantile": ("quantile", arrays["quantile"].astype(np.float32)),
    }
    data_vars = {
        "pred_gsi": (("time", "node"), arrays["pred"].astype(np.float32)),
        "pred_mlp": (("time", "node"), arrays["pred"].astype(np.float32)),
        "pred_quantile": (("time", "node", "quantile"), arrays["pred_quantile"].astype(np.float32)),
        "pred_quantile_mlp": (("time", "node", "quantile"), arrays["pred_quantile"].astype(np.float32)),
        "idw_train": (("time", "node"), arrays["idw"].astype(np.float32)),
        "era5": (("time", "node"), arrays["era5"].astype(np.float32)),
        "imerg": (("time", "node"), arrays["imerg"].astype(np.float32)),
        "train_gauge": (("time", "node"), arrays["train_gauge"].astype(np.float32)),
        "train_mask": (("time", "node"), arrays["train_mask"].astype(np.uint8)),
        "holdout_gauge": (("time", "node"), arrays["holdout_gauge"].astype(np.float32)),
        "holdout_mask": (("time", "node"), arrays["holdout_mask"].astype(np.uint8)),
        "model_available": (("time",), arrays["model_available"].astype(np.uint8).reshape(-1)),
        "train_precip_mean": (("time",), arrays["train_precip_mean"].astype(np.float32).reshape(-1)),
        "node_lon": (("node",), node_coords[:, 0].astype(np.float32)),
        "node_lat": (("node",), node_coords[:, 1].astype(np.float32)),
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "description": (
                "ANN per-day MLP inference from one checkpoint per date. Days without "
                "a checkpoint, or below the configured precipitation threshold, are "
                "written as zero-precipitation maps."
            ),
            "units": "mm/day",
            "npz": "" if npz_path is None else str(npz_path),
            "seq_dir": str(seq_dir),
            "ckpt_root": str(ckpt_root),
        },
    )
    enc = {}
    for v in ds.data_vars:
        dtype = str(ds[v].dtype)
        if ds[v].ndim == 3:
            enc[v] = {"zlib": True, "complevel": 4, "dtype": dtype, "chunksizes": (1, n_nodes, ds.sizes["quantile"])}
        elif ds[v].ndim == 2:
            enc[v] = {"zlib": True, "complevel": 4, "dtype": dtype, "chunksizes": (1, n_nodes)}
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
    csv_path = Path(cfg["data_csv_path"])
    ckpt_root = Path(cfg["data_ckpt_root"])
    graph_path_raw = cfg.get("data_graph_path", "")
    graph_path = Path(graph_path_raw) if str(graph_path_raw).strip() else None
    out_dir = Path(cfg["output_out_dir"])
    out_nc = Path(cfg["output_out_nc"])
    checkpoint_name = str(cfg.get("data_checkpoint_name", "best.ckpt"))
    feature_cols = [str(c) for c in cfg["data_feature_cols"]]

    runtime_cfg = {
        "graph_path": graph_path,
        "feature_cols": feature_cols,
        "k_gauges": int(cfg.get("data_k_gauges", 10)),
        "era5_idx": int(cfg.get("data_era5_idx", 2)),
        "imerg_idx": int(cfg.get("data_imerg_idx", 3)),
        "node_batch_size": int(cfg.get("compute_node_batch_size", 65536)),
        "amp": bool(cfg.get("compute_amp", True)),
        "idw_k": int(cfg.get("idw_k", 10)),
        "idw_power": float(cfg.get("idw_power", 2.0)),
        "min_train_precip_mean": cfg.get("dates_min_train_precip_mean", None),
        "strict_checkpoints": bool(cfg.get("data_strict_checkpoints", False)),
        "val_fraction": float(cfg.get("training_val_fraction", 0.20)),
        "seed": int(cfg.get("training_seed", 42)),
    }

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
    my_dates = dates[rank::world]

    df_all = pd.read_csv(csv_path, parse_dates=["date"])
    df_all["date_str"] = df_all["date"].dt.strftime("%Y-%m-%d")
    df_all = df_all.dropna(subset=feature_cols + ["gauge"]).copy()

    sample0 = _load_pt(date_to_file[dates[0]])
    node_coords = sample_node_coords(sample0, graph_path)
    q_default = np.arange(1, int(cfg.get("model_n_quantiles", 32)) + 1, dtype=np.float32)
    q_default = q_default / (len(q_default) + 1)

    print(
        f"[Rank {rank}/{world}] device={device} dates_total={len(dates)} mine={len(my_dates)} "
        f"ckpt_root={ckpt_root}",
        flush=True,
    )
    print(f"[Rank {rank}] feature_cols={feature_cols}", flush=True)

    outputs = []
    zero_days = 0
    for i, day in enumerate(my_dates, start=1):
        ckpt = checkpoint_for_day(ckpt_root, day, checkpoint_name)
        if i == 1 or i % 25 == 0 or i == len(my_dates):
            print(
                f"[Rank {rank}] {i}/{len(my_dates)} day={day} "
                f"ckpt={'yes' if ckpt.exists() else 'no-zero'}",
                flush=True,
            )
        item = infer_one_day(
            day=day,
            seq_file=date_to_file[day],
            ckpt_file=ckpt,
            df_all=df_all,
            node_coords_ref=node_coords,
            q_default=q_default,
            device=device,
            cfg=runtime_cfg,
        )
        zero_days += int(item["model_available"].item() == 0)
        outputs.append(item)

    out = stack_outputs(outputs, q_default)
    rank_out = out_dir / f"preds_rank{rank:03d}.npz"
    tmp_out = out_dir / f"preds_rank{rank:03d}.tmp"
    with open(tmp_out, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp_out, rank_out)
    print(
        f"[Rank {rank}] wrote {rank_out} dates={out['dates'].shape} zero_days={zero_days}",
        flush=True,
    )

    if rank == 0:
        merged, merged_arrays = merge_rank_outputs(
            out_dir,
            world,
            merged_name=str(cfg.get("output_merged_name", "preds_merged.npz")),
            return_arrays=True,
        )
        if bool(cfg.get("output_save_nc", True)):
            make_netcdf_from_arrays(
                merged_arrays,
                out_nc=out_nc,
                node_coords=node_coords,
                seq_dir=seq_dir,
                ckpt_root=ckpt_root,
                npz_path=merged,
            )


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_ID", None)
    main()
