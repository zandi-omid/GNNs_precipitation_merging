#!/usr/bin/env python3
# coding: utf-8

"""
Build a tabular dataframe for the MLP baseline from the GSI sequence dataset.

Each output row is one observed training-gauge node on one day.  The target
gauge is removed from the donor set before computing neighboring-gauge
features, which avoids label leakage.

Default feature family
----------------------
    ERA5, IMERG, idw_loo,
    gauge_01_value, gauge_01_distance_km, gauge_01_mask,
    ...
    gauge_K_value,  gauge_K_distance_km,  gauge_K_mask

By default, neighboring gauges are selected from active A1 senders only.  This
gives the MLP the same local sparse-gauge context that A1 GAT receives, but
flattened into a tabular row.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


# ============================================================
# DEFAULT CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQ_DIR = Path(
    PROJECT_ROOT / "GSI/data/"
    "pyg_sequences_tgsi_T030_2005-01-01_to_2024-12-31_"
    "gauge2meshA1_100km_A2knn8_manual_xy_nonzero"
)

GRAPH_PATH = Path(
    PROJECT_ROOT / "data/graphs/"
    "gsi_a1_r100_a2_knn8_manual_sparse.pkl"
)

OUT_CSV = Path(
    PROJECT_ROOT / "ANN/df_CSV_files/"
    "df_gsi_a1a2_kgauge10_idw_loo.csv"
)

GAUGE_VALUE_IDX = 0
GAUGE_MASK_IDX = 1
ERA5_IDX = 2
IMERG_IDX = 3

K_GAUGES = 10
IDW_K = 10
IDW_POWER = 2.0

DATE_START = None
DATE_END = None


# ============================================================
# HELPERS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build a GSI tabular MLP dataframe with nearby-gauge features."
    )
    p.add_argument("--seq_dir", type=Path, default=SEQ_DIR)
    p.add_argument("--graph_path", type=Path, default=GRAPH_PATH)
    p.add_argument("--out_csv", type=Path, default=OUT_CSV)
    p.add_argument("--k_gauges", type=int, default=K_GAUGES)
    p.add_argument("--idw_k", type=int, default=IDW_K)
    p.add_argument("--idw_power", type=float, default=IDW_POWER)
    p.add_argument(
        "--min_neighbor_donors",
        type=int,
        default=1,
        help="Drop target rows with fewer active neighboring donor gauges.",
    )
    p.add_argument("--date_start", default=DATE_START)
    p.add_argument("--date_end", default=DATE_END)
    p.add_argument("--gauge_value_idx", type=int, default=GAUGE_VALUE_IDX)
    p.add_argument("--gauge_mask_idx", type=int, default=GAUGE_MASK_IDX)
    p.add_argument("--era5_idx", type=int, default=ERA5_IDX)
    p.add_argument("--imerg_idx", type=int, default=IMERG_IDX)
    p.add_argument(
        "--aux_feature",
        action="append",
        default=None,
        metavar="NAME:INDEX",
        help=(
            "Auxiliary column and sequence channel. Repeat for multiple inputs. "
            "Defaults to ERA5:2 and IMERG:3."
        ),
    )
    p.add_argument(
        "--aux_only",
        action="store_true",
        help="Write only auxiliary predictors and gauge targets; do not construct gauge-neighbor or IDW features.",
    )
    p.add_argument(
        "--neighbor_source",
        choices=["a1", "kdtree"],
        default="a1",
        help="Use active A1 sender gauges or unrestricted K-nearest gauges.",
    )
    p.add_argument(
        "--crossfit_folds",
        type=int,
        default=0,
        help="Use stored gauge_fold_id and exclude every target's entire fold from donors.",
    )
    return p.parse_args()


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_node_coords_from_graph(graph_path: Path) -> np.ndarray:
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "graph" not in payload:
        raise KeyError(f"Could not find graph payload in {graph_path}")
    graph = payload["graph"]
    nodes = payload.get("common_nodes", None)
    if nodes is None:
        nodes = list(graph.nodes)
    coords = []
    for node in nodes:
        data = graph.nodes[node]
        coords.append([float(data["lon"]), float(data["lat"])])
    return np.asarray(coords, dtype=np.float64)


def _date_in_range(date_str: str, start: str | None, end: str | None) -> bool:
    d = pd.Timestamp(date_str)
    if start is not None and d < pd.Timestamp(start):
        return False
    if end is not None and d > pd.Timestamp(end):
        return False
    return True


def _ensure_2d_query(dists: np.ndarray, inds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.ndim(dists) == 1:
        dists = dists[:, None]
        inds = inds[:, None]
    return dists, inds


def _idw_from_neighbors(
    distances: np.ndarray,
    values: np.ndarray,
    power: float,
) -> float:
    if len(values) == 0:
        return np.nan
    vals = values.astype(np.float64)
    if power == 0:
        return float(np.nanmean(vals))
    eps = 1e-6
    w = 1.0 / np.maximum(distances.astype(np.float64), eps) ** float(power)
    return float(np.sum(w * vals) / np.maximum(np.sum(w), eps))


def _haversine_km(lonlat_a: np.ndarray, lonlat_b: np.ndarray) -> np.ndarray:
    lon1 = np.deg2rad(lonlat_a[:, 0].astype(np.float64))
    lat1 = np.deg2rad(lonlat_a[:, 1].astype(np.float64))
    lon2 = np.deg2rad(lonlat_b[:, 0].astype(np.float64))
    lat2 = np.deg2rad(lonlat_b[:, 1].astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return (6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float64)


def _neighbor_feature_columns(k_gauges: int) -> list[str]:
    cols = []
    for rank in range(1, int(k_gauges) + 1):
        prefix = f"gauge_{rank:02d}"
        cols.extend(
            [
                f"{prefix}_value",
                f"{prefix}_distance_km",
                f"{prefix}_mask",
            ]
        )
    return cols


def _empty_neighbor_features(k_gauges: int) -> dict[str, float]:
    feats = {}
    for rank in range(1, int(k_gauges) + 1):
        prefix = f"gauge_{rank:02d}"
        feats[f"{prefix}_value"] = 0.0
        feats[f"{prefix}_distance_km"] = 0.0
        feats[f"{prefix}_mask"] = 0.0
    return feats


def _nearest_gauge_features_for_targets(
    node_coords: np.ndarray,
    donor_idx: np.ndarray,
    donor_vals: np.ndarray,
    target_idx: np.ndarray,
    k_gauges: int,
    idw_k: int,
    idw_power: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """
    Build K-nearest gauge features for each target node.

    If a target node is itself a donor, that donor is removed before features
    are created. Distances are in the same coordinate units as node_coords;
    GSI node coordinates are lon/lat on a 0.1-degree grid, so this is an
    approximate local distance. The column name keeps `_km` for compatibility
    with the intended tabular interface.
    """
    n_target = len(target_idx)
    out_features = []
    out_idw = np.full(n_target, np.nan, dtype=np.float32)
    out_donor_count = np.zeros(n_target, dtype=np.int32)

    if len(donor_idx) == 0:
        return [_empty_neighbor_features(k_gauges) for _ in range(n_target)], out_idw, out_donor_count

    donor_coords = node_coords[donor_idx].astype(np.float64)
    donor_vals64 = donor_vals.astype(np.float64)
    donor_lookup = {int(node): pos for pos, node in enumerate(donor_idx)}

    k_query = min(max(int(k_gauges), int(idw_k)) + 1, len(donor_idx))
    tree = cKDTree(donor_coords)
    dists, inds = tree.query(node_coords[target_idx].astype(np.float64), k=k_query)
    dists, inds = _ensure_2d_query(dists, inds)

    for row_i, node in enumerate(target_idx):
        neigh = inds[row_i]
        dist = dists[row_i]

        self_pos = donor_lookup.get(int(node))
        if self_pos is not None:
            keep = neigh != self_pos
            neigh = neigh[keep]
            dist = dist[keep]

        valid = np.isfinite(dist)
        neigh = neigh[valid]
        dist = dist[valid]
        if len(neigh):
            dist_km = _haversine_km(
                np.repeat(node_coords[int(node)][None, :], len(neigh), axis=0),
                donor_coords[neigh],
            )
        else:
            dist_km = dist

        out_donor_count[row_i] = int(len(neigh))

        idw_neigh = neigh[: int(idw_k)]
        idw_dist = dist_km[: int(idw_k)]
        if len(idw_neigh):
            out_idw[row_i] = np.float32(
                _idw_from_neighbors(
                    distances=idw_dist,
                    values=donor_vals64[idw_neigh],
                    power=idw_power,
                )
            )

        feats = _empty_neighbor_features(k_gauges)
        for j, donor_pos in enumerate(neigh[: int(k_gauges)], start=1):
            prefix = f"gauge_{j:02d}"
            feats[f"{prefix}_value"] = float(donor_vals64[donor_pos])
            feats[f"{prefix}_distance_km"] = float(dist_km[j - 1])
            feats[f"{prefix}_mask"] = 1.0

        out_features.append(feats)

    return out_features, out_idw, out_donor_count


def _a1_gauge_features_for_targets(
    node_coords: np.ndarray,
    donor_idx: np.ndarray,
    donor_vals: np.ndarray,
    target_idx: np.ndarray,
    edge_index_a1: np.ndarray,
    k_gauges: int,
    idw_k: int,
    idw_power: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    n_target = len(target_idx)
    out_features = []
    out_idw = np.full(n_target, np.nan, dtype=np.float32)
    out_donor_count = np.zeros(n_target, dtype=np.int32)

    donor_lookup = {int(node): pos for pos, node in enumerate(donor_idx)}
    recv = edge_index_a1[0].astype(np.int64)
    send = edge_index_a1[1].astype(np.int64)
    by_recv: dict[int, list[int]] = {}
    for r, s in zip(recv, send):
        by_recv.setdefault(int(r), []).append(int(s))

    for row_i, node in enumerate(target_idx):
        candidates = []
        for sender in by_recv.get(int(node), []):
            donor_pos = donor_lookup.get(int(sender))
            if donor_pos is None or int(sender) == int(node):
                continue
            candidates.append((int(sender), int(donor_pos)))

        feats = _empty_neighbor_features(k_gauges)
        if not candidates:
            out_features.append(feats)
            continue

        sender_nodes = np.asarray([x[0] for x in candidates], dtype=np.int64)
        donor_pos = np.asarray([x[1] for x in candidates], dtype=np.int64)
        dist_km = _haversine_km(
            np.repeat(node_coords[int(node)][None, :], len(sender_nodes), axis=0),
            node_coords[sender_nodes],
        )
        order = np.argsort(dist_km)
        sender_nodes = sender_nodes[order]
        donor_pos = donor_pos[order]
        dist_km = dist_km[order]

        out_donor_count[row_i] = int(len(sender_nodes))
        idw_n = min(int(idw_k), len(sender_nodes))
        if idw_n:
            out_idw[row_i] = np.float32(
                _idw_from_neighbors(
                    distances=dist_km[:idw_n],
                    values=donor_vals[donor_pos[:idw_n]],
                    power=idw_power,
                )
            )

        for j, pos in enumerate(donor_pos[: int(k_gauges)], start=1):
            prefix = f"gauge_{j:02d}"
            feats[f"{prefix}_value"] = float(donor_vals[pos])
            feats[f"{prefix}_distance_km"] = float(dist_km[j - 1])
            feats[f"{prefix}_mask"] = 1.0

        out_features.append(feats)

    return out_features, out_idw, out_donor_count


def _required_mask(sample: dict, key: str, fallback_key: str | None = None):
    if key in sample:
        return sample[key]
    if fallback_key is not None and fallback_key in sample:
        return sample[fallback_key]
    raise KeyError(f"Missing required mask {key!r}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    seq_files = sorted(args.seq_dir.glob("seq_*.pt"))
    if not seq_files:
        raise RuntimeError(f"No seq_*.pt files found in: {args.seq_dir}")

    if args.aux_feature:
        aux_features = {}
        for value in args.aux_feature:
            try:
                name, index = value.rsplit(":", 1)
                aux_features[name] = int(index)
            except ValueError as exc:
                raise ValueError(f"Invalid --aux_feature {value!r}; expected NAME:INDEX") from exc
    else:
        aux_features = {"ERA5": args.era5_idx, "IMERG": args.imerg_idx}

    rows = []
    feature_cols = list(aux_features)
    if not args.aux_only:
        feature_cols.extend(["idw_loo", *_neighbor_feature_columns(args.k_gauges)])

    print(f"Sequence directory: {args.seq_dir}")
    print(f"Graph path: {args.graph_path}")
    print(f"Found {len(seq_files):,} sequence files")
    print(f"Output CSV: {args.out_csv}")
    print(f"K nearest gauge columns: {args.k_gauges}")
    print(f"Neighbor source: {args.neighbor_source}")
    print(f"Auxiliary-only mode: {args.aux_only}")
    print(f"IDW leave-one-out: k={args.idw_k}, power={args.idw_power}")
    print(f"Feature columns ({len(feature_cols)}):")
    print(feature_cols)

    graph_node_coords = _load_node_coords_from_graph(args.graph_path)

    for seq_file in tqdm(seq_files, desc="Building GSI tabular dataframe"):
        d = _load_pt(seq_file)

        date_str = str(d["date"])
        if not _date_in_range(date_str, args.date_start, args.date_end):
            continue

        x = _to_numpy(d["x"])  # [T, N, F]
        y = _to_numpy(d["y"]).astype(np.float32)
        train_mask = _to_numpy(
            _required_mask(d, "train_available_mask", "y_train_mask")
        ).astype(bool)
        if "node_coords" in d:
            node_coords = _to_numpy(d["node_coords"]).astype(np.float64)
        else:
            node_coords = graph_node_coords

        if x.ndim != 3:
            raise ValueError(f"{seq_file} has x shape {x.shape}, expected [T,N,F]")

        _, n_nodes, n_features = x.shape
        required_feature_idx = max(
            args.gauge_value_idx,
            args.gauge_mask_idx,
            *aux_features.values(),
        )
        if n_features <= required_feature_idx:
            raise RuntimeError(
                f"{seq_file.name} has only F={n_features} features, but "
                f"feature index {required_feature_idx} is required."
            )
        if node_coords.shape != (n_nodes, 2):
            raise ValueError(
                f"{seq_file.name}: node_coords has shape {node_coords.shape}, "
                f"expected {(n_nodes, 2)}"
            )

        x_last = x[-1, :, :]
        valid = train_mask & np.isfinite(y)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        if args.aux_only:
            for node_idx in valid_idx:
                row = {
                    "date": date_str,
                    "seq_file": seq_file.name,
                    "node_idx": int(node_idx),
                    "lon": float(node_coords[node_idx, 0]),
                    "lat": float(node_coords[node_idx, 1]),
                    "gauge": float(y[node_idx]),
                    "feature_names": "|".join(str(v) for v in d.get("feature_names", [])),
                }
                for name, index in aux_features.items():
                    row[name] = float(x_last[node_idx, index])
                rows.append(row)
            continue

        if args.crossfit_folds:
            if "gauge_fold_id" not in d:
                raise KeyError(f"{seq_file.name} has no gauge_fold_id")
            fold_id = _to_numpy(d["gauge_fold_id"]).astype(np.int64)
            target_groups = [valid_idx[fold_id[valid_idx] == fold] for fold in range(args.crossfit_folds)]
        else:
            target_groups = [valid_idx]

        edge_index_a1 = _to_numpy(d["edge_index_a1"]).astype(np.int64)
        for target_idx in target_groups:
            if len(target_idx) == 0:
                continue
            if args.crossfit_folds:
                held_fold = int(fold_id[target_idx[0]])
                donor_idx = valid_idx[fold_id[valid_idx] != held_fold]
            else:
                donor_idx = valid_idx
            donor_vals = y[donor_idx]
            if args.neighbor_source == "a1":
                neighbor_features, idw_loo, donor_counts = _a1_gauge_features_for_targets(
                    node_coords=node_coords, donor_idx=donor_idx, donor_vals=donor_vals,
                    target_idx=target_idx, edge_index_a1=edge_index_a1,
                    k_gauges=args.k_gauges, idw_k=args.idw_k, idw_power=args.idw_power,
                )
            else:
                neighbor_features, idw_loo, donor_counts = _nearest_gauge_features_for_targets(
                    node_coords=node_coords, donor_idx=donor_idx, donor_vals=donor_vals,
                    target_idx=target_idx, k_gauges=args.k_gauges,
                    idw_k=args.idw_k, idw_power=args.idw_power,
                )

            for local_i, node_idx in enumerate(target_idx):
                if int(donor_counts[local_i]) < int(args.min_neighbor_donors):
                    continue
                row = {
                    "date": date_str,
                    "seq_file": seq_file.name,
                    "node_idx": int(node_idx),
                    "lon": float(node_coords[node_idx, 0]),
                    "lat": float(node_coords[node_idx, 1]),
                    "idw_loo": float(idw_loo[local_i]),
                    "gauge": float(y[node_idx]),
                    "n_gauge_donors": int(donor_counts[local_i]),
                    "neighbor_source": args.neighbor_source,
                    "gauge_fold": int(fold_id[node_idx]) if args.crossfit_folds else -1,
                    "feature_names": "|".join(str(v) for v in d.get("feature_names", [])),
                }
                for name, index in aux_features.items():
                    row[name] = float(x_last[node_idx, index])
                row.update(neighbor_features[local_i])
                rows.append(row)

    df = pd.DataFrame(rows)

    print("\nDone building dataframe.")
    print(f"Rows: {len(df):,}")
    print("Columns:")
    print(df.columns.tolist())

    if len(df) > 0:
        summary_cols = [*aux_features, "gauge"]
        if not args.aux_only:
            summary_cols.extend(["idw_loo", "n_gauge_donors"])
            summary_cols.extend(_neighbor_feature_columns(min(args.k_gauges, 2)))
        print("\nHead:")
        print(df.head())

        print("\nSummary:")
        print(df[summary_cols].describe())

        print("\nNaN counts:")
        print(df[summary_cols].isna().sum())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"\nSaved dataframe to:\n{args.out_csv}")
    print("\nSuggested MLP feature_cols:")
    print(feature_cols)


if __name__ == "__main__":
    main()
