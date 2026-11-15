#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tomli as tomllib
from tqdm import tqdm

import sys

ANN_DIR = Path(__file__).resolve().parent / "ANN"
if str(ANN_DIR) not in sys.path:
    sys.path.insert(0, str(ANN_DIR))

from inference_MLP_perday import _index_files_by_date, _load_pt, build_all_node_features, sample_node_coords


def load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(cfg: dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build date/node auxiliary nearest-gauge features for GSI StackCV meta learners."
    )
    p.add_argument("--config", required=True)
    return p.parse_args()


def aux_feature_cols(k: int, include_mask: bool = True) -> list[str]:
    cols: list[str] = []
    for rank in range(1, int(k) + 1):
        cols.extend([f"gauge_{rank:02d}_value", f"gauge_{rank:02d}_distance_km"])
        if include_mask:
            cols.append(f"gauge_{rank:02d}_mask")
    return cols


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    seq_dir = Path(cfg_get(cfg, "data", "seq_dir"))
    graph_path = Path(cfg_get(cfg, "data", "graph_path"))
    out_csv = Path(cfg_get(cfg, "output", "csv"))
    k_gauges = int(cfg_get(cfg, "features", "k_gauges", 10))
    idw_k = int(cfg_get(cfg, "features", "idw_k", 10))
    idw_power = float(cfg_get(cfg, "features", "idw_power", 2.0))
    era5_idx = int(cfg_get(cfg, "features", "era5_idx", 2))
    imerg_idx = int(cfg_get(cfg, "features", "imerg_idx", 3))
    target_mask_name = str(cfg_get(cfg, "features", "target_mask", "eval_mask"))
    date_start = cfg_get(cfg, "dates", "start", None)
    date_end = cfg_get(cfg, "dates", "end", None)

    feature_cols_with_mask = aux_feature_cols(k_gauges, include_mask=True)
    output_cols = []
    for rank in range(1, k_gauges + 1):
        output_cols.extend([f"gauge_{rank:02d}_value", f"gauge_{rank:02d}_distance_km"])

    date_to_file = _index_files_by_date(seq_dir)
    dates = sorted(date_to_file)
    if date_start:
        dates = [d for d in dates if d >= str(date_start)]
    if date_end:
        dates = [d for d in dates if d <= str(date_end)]
    if not dates:
        raise FileNotFoundError(f"No dates selected from {seq_dir}")

    sample0 = _load_pt(date_to_file[dates[0]])
    node_coords = sample_node_coords(sample0, graph_path)
    rows = []

    for day in tqdm(dates, desc="Building StackCV meta aux"):
        sample = _load_pt(date_to_file[day])
        if target_mask_name not in sample:
            raise KeyError(f"{date_to_file[day]} has no mask {target_mask_name!r}")
        target_mask = sample[target_mask_name].detach().cpu().numpy().astype(bool)
        if not target_mask.any():
            continue
        coords = sample_node_coords(sample, graph_path)
        if coords.shape != node_coords.shape or not np.allclose(coords, node_coords):
            raise ValueError(f"Node coordinates changed in {date_to_file[day]}")
        features_df, _, _, _ = build_all_node_features(
            sample=sample,
            node_coords=node_coords,
            feature_cols=feature_cols_with_mask,
            k_gauges=k_gauges,
            idw_k=idw_k,
            idw_power=idw_power,
            era5_idx=era5_idx,
            imerg_idx=imerg_idx,
        )
        nodes = np.where(target_mask)[0].astype(np.int64)
        sub = features_df.iloc[nodes].reset_index(drop=True)
        out = pd.DataFrame({"date": day, "node": nodes})
        mask_cols = [f"gauge_{rank:02d}_mask" for rank in range(1, k_gauges + 1)]
        out["n_a1_donors"] = sub[mask_cols].sum(axis=1).astype(np.int16)
        for col in output_cols:
            out[col] = sub[col].astype(np.float32)
        rows.append(out)

    if not rows:
        raise RuntimeError("No auxiliary rows were built.")
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows = all_rows.drop_duplicates(["date", "node"]).sort_values(["date", "node"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_csv, index=False)
    print(f"Saved StackCV meta aux CSV: {out_csv}")
    print(f"Rows: {len(all_rows)} nodes={all_rows['node'].nunique()} dates={all_rows['date'].nunique()}")


if __name__ == "__main__":
    main()
