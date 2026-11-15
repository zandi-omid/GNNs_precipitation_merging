#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tomli as tomllib

from inference_MLP_perday import (
    _index_files_by_date,
    _load_pt,
    build_all_node_features,
    get_device,
    get_rank_world,
    load_mlp_checkpoint_quietly,
    make_netcdf_from_arrays,
    merge_rank_outputs,
    predict_nodes,
    quantiles_from_model,
    resolve_dates,
    sample_node_coords,
    stack_outputs,
    train_precip_mean,
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


def _as_year_tuple(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    return int(value[0]), int(value[1])


def _as_year_ranges(value: Any) -> Tuple[Tuple[int, int], ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ()
        if len(value) == 2 and not isinstance(value[0], (list, tuple)):
            return (_as_year_tuple(value, (0, 0)),)
        return tuple(_as_year_tuple(item, (0, 0)) for item in value)
    raise ValueError(f"Expected [[y0, y1], ...], got: {value}")


def load_feature_scaler(
    csv_path: Path,
    feature_cols: List[str],
    train_years: Tuple[int, int],
    exclude_train_years: Tuple[Tuple[int, int], ...] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    usecols = ["date", *feature_cols]
    df = pd.read_csv(csv_path, parse_dates=["date"], usecols=usecols)
    train_mask = df["date"].dt.year.between(*train_years)
    for lo, hi in exclude_train_years:
        train_mask &= ~df["date"].dt.year.between(lo, hi)
    train = df[train_mask].dropna(subset=feature_cols)
    if len(train) == 0:
        raise RuntimeError(
            f"No training rows found in {csv_path} for train_years={train_years} "
            f"exclude_train_years={exclude_train_years}"
        )
    means = train[feature_cols].mean().to_numpy(dtype=np.float32)
    stds = (
        train[feature_cols]
        .std()
        .replace(0, 1.0)
        .fillna(1.0)
        .to_numpy(dtype=np.float32)
    )
    return means, stds


def _make_output_item(
    day: str,
    sample: Dict[str, Any],
    pred: np.ndarray,
    pred_q: np.ndarray,
    idw_train: np.ndarray,
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

    x_last = sample["x"][-1].float().numpy().astype(np.float32)
    return {
        "date": np.array(day, dtype="U10"),
        "pred": pred.astype(np.float32),
        "pred_quantile": pred_q.astype(np.float32),
        "idw": idw_train.astype(np.float32),
        "train_gauge": train_gauge,
        "train_mask": train_mask.astype(np.uint8),
        "holdout_gauge": holdout_gauge,
        "holdout_mask": eval_mask.astype(np.uint8),
        "era5": x_last[:, int(era5_idx)].astype(np.float32),
        "imerg": x_last[:, int(imerg_idx)].astype(np.float32),
        "train_precip_mean": np.array(mean_train, dtype=np.float32),
        "model_available": np.array(1, dtype=np.uint8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = flatten_config(load_toml(args.config))

    seq_dir = Path(cfg["data_seq_dir"])
    graph_path_raw = cfg.get("data_graph_path", "")
    graph_path = Path(graph_path_raw) if str(graph_path_raw).strip() else None
    csv_path = Path(cfg["data_csv_path"])
    ckpt = Path(cfg["data_ckpt"])
    out_dir = Path(cfg["output_out_dir"])
    out_nc = Path(cfg["output_out_nc"])
    feature_cols = [str(c) for c in cfg["data_feature_cols"]]

    train_years = _as_year_tuple(cfg.get("data_train_years"), (2005, 2015))
    exclude_train_years = _as_year_ranges(cfg.get("data_exclude_train_years"))
    normalize = bool(cfg.get("data_normalize_features", True))
    k_gauges = int(cfg.get("data_k_gauges", 10))
    era5_idx = int(cfg.get("data_era5_idx", 2))
    imerg_idx = int(cfg.get("data_imerg_idx", 3))
    node_batch_size = int(cfg.get("compute_node_batch_size", 65536))
    amp = bool(cfg.get("compute_amp", True))
    idw_k = int(cfg.get("idw_k", 10))
    idw_power = float(cfg.get("idw_power", 2.0))
    n_quantiles = int(cfg.get("model_n_quantiles", 32))
    crossfit_folds = int(cfg.get("crossfit_n_folds", 0)) if bool(cfg.get("crossfit_enabled", False)) else 0

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing global MLP checkpoint: {ckpt}")

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

    model = load_mlp_checkpoint_quietly(ckpt)
    is_quantile_model = str(getattr(model, "loss_name", "quantile")).lower() == "quantile"
    q = (
        quantiles_from_model(model, n_quantiles)
        if is_quantile_model
        else np.array([0.5], dtype=np.float32)
    )
    if normalize:
        means, stds = load_feature_scaler(
            csv_path,
            feature_cols,
            train_years,
            exclude_train_years=exclude_train_years,
        )
    else:
        means = np.zeros(len(feature_cols), dtype=np.float32)
        stds = np.ones(len(feature_cols), dtype=np.float32)

    sample0 = _load_pt(date_to_file[dates[0]])
    node_coords = sample_node_coords(sample0, graph_path)

    if rank == 0:
        print(
            f"[GLOBAL_MLP_INFER] checkpoint={ckpt} dates={dates[0]}..{dates[-1]} "
            f"n_dates={len(dates)} world={world}",
            flush=True,
        )
        print(
            f"[GLOBAL_MLP_INFER] feature_cols={feature_cols} normalize={normalize} "
            f"train_years={train_years} exclude_train_years={exclude_train_years}",
            flush=True,
        )
        print(
            f"[GLOBAL_MLP_INFER] loss_name={model.loss_name} "
            f"output_channels={len(q)}",
            flush=True,
        )
    print(
        f"[Rank {rank}/{world}] device={device} assigned_dates={len(my_dates)}",
        flush=True,
    )

    outputs = []
    for i, day in enumerate(my_dates, start=1):
        sample = _load_pt(date_to_file[day])
        coords = sample_node_coords(sample, graph_path)
        if coords.shape != node_coords.shape or not np.allclose(coords, node_coords):
            raise ValueError(f"Node coordinates changed in {date_to_file[day]}")

        if crossfit_folds:
            if "gauge_fold_id" not in sample:
                raise KeyError(f"Cross-fit inference requires gauge_fold_id in {date_to_file[day]}")
            fold_id = sample["gauge_fold_id"].long().numpy()
            available = sample["train_available_mask"].bool().numpy()
            fold_predictions = []
            fold_idw = []
            for held_fold in range(crossfit_folds):
                folded_sample = dict(sample)
                donor_mask = available & (fold_id != held_fold)
                folded_sample["train_available_mask"] = torch.from_numpy(donor_mask)
                features_df, idw_fold, _, _ = build_all_node_features(
                    sample=folded_sample, node_coords=node_coords,
                    feature_cols=feature_cols, k_gauges=k_gauges,
                    idw_k=idw_k, idw_power=idw_power,
                    era5_idx=era5_idx, imerg_idx=imerg_idx,
                )
                features = features_df.to_numpy(dtype=np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                features = (features - means.reshape(1, -1)) / stds.reshape(1, -1)
                fold_predictions.append(predict_nodes(
                    model=model, features=features, device=device,
                    batch_size=node_batch_size, amp=amp,
                ))
                fold_idw.append(idw_fold)
            y_hat = np.mean(fold_predictions, axis=0).astype(np.float32)
            idw_train = np.mean(fold_idw, axis=0).astype(np.float32)
            for held_fold, (fold_pred, idw_fold) in enumerate(zip(fold_predictions, fold_idw)):
                targets = available & (fold_id == held_fold)
                y_hat[targets] = fold_pred[targets]
                idw_train[targets] = idw_fold[targets]
        else:
            features_df, idw_train, _, _ = build_all_node_features(
                sample=sample, node_coords=node_coords,
                feature_cols=feature_cols, k_gauges=k_gauges,
                idw_k=idw_k, idw_power=idw_power,
                era5_idx=era5_idx, imerg_idx=imerg_idx,
            )
            features = features_df.to_numpy(dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = (features - means.reshape(1, -1)) / stds.reshape(1, -1)
            y_hat = predict_nodes(
                model=model, features=features, device=device,
                batch_size=node_batch_size, amp=amp,
            )
        if y_hat.ndim == 2:
            pred_q = y_hat.astype(np.float32)
            q50_idx = int(np.argmin(np.abs(q - 0.5)))
            pred = pred_q[:, q50_idx].astype(np.float32)
        else:
            pred = y_hat.astype(np.float32)
            pred_q = pred[:, None]

        outputs.append(
            _make_output_item(
                day=day,
                sample=sample,
                pred=pred,
                pred_q=pred_q,
                idw_train=idw_train,
                mean_train=train_precip_mean(sample),
                era5_idx=era5_idx,
                imerg_idx=imerg_idx,
            )
        )

        if i == 1 or i % 50 == 0 or i == len(my_dates):
            print(f"[Rank {rank}] {i}/{len(my_dates)} day={day}", flush=True)

    out = stack_outputs(outputs, q)
    rank_out = out_dir / f"preds_rank{rank:03d}.npz"
    tmp_out = out_dir / f"preds_rank{rank:03d}.tmp"
    with open(tmp_out, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp_out, rank_out)
    print(f"[Rank {rank}] wrote {rank_out} dates={out['dates'].shape}", flush=True)

    if rank == 0:
        merged, arrays = merge_rank_outputs(
            out_dir,
            world,
            merged_name=str(cfg.get("output_merged_name", "preds_merged.npz")),
            return_arrays=True,
        )
        if bool(cfg.get("output_save_nc", True)):
            make_netcdf_from_arrays(
                arrays,
                out_nc=out_nc,
                node_coords=node_coords,
                seq_dir=seq_dir,
                ckpt_root=ckpt.parent,
                npz_path=merged,
            )


if __name__ == "__main__":
    main()
