#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse
import pickle
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from train_stack_meta_mlp_gsi import MetaMLP


ROOT = Path(__file__).resolve().parent
ANN_DIR = ROOT / "ANN"
if str(ANN_DIR) not in sys.path:
    sys.path.insert(0, str(ANN_DIR))

from build_df import _haversine_km, _load_pt, _to_numpy


def load_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply a trained StackCV PyTorch quantile meta-MLP to full-node GSI/MLP inference maps."
    )
    p.add_argument("--config", required=True, type=Path)
    return p.parse_args()


def _load_node_coords_from_graph(graph_path: Path) -> np.ndarray:
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)
    graph = payload["graph"]
    nodes = payload.get("common_nodes", list(graph.nodes))
    return np.asarray(
        [(float(graph.nodes[n]["lon"]), float(graph.nodes[n]["lat"])) for n in nodes],
        dtype=np.float64,
    )


def _load_edge_index_a1(seq_dir: Path) -> np.ndarray:
    seq_files = sorted(seq_dir.glob("seq_*.pt"))
    if not seq_files:
        raise FileNotFoundError(f"No seq_*.pt files found in {seq_dir}")
    sample = _load_pt(seq_files[0])
    print(f"Loaded static A1 graph from: {seq_files[0]}")
    return _to_numpy(sample["edge_index_a1"]).astype(np.int64)


def _precompute_a1_sorted_senders(
    edge_index_a1: np.ndarray,
    node_coords: np.ndarray,
    n_nodes: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    recv = edge_index_a1[0].astype(np.int64)
    send = edge_index_a1[1].astype(np.int64)
    by_recv: list[list[int]] = [[] for _ in range(n_nodes)]
    for r, s in zip(recv, send):
        if 0 <= int(r) < n_nodes and 0 <= int(s) < n_nodes and int(r) != int(s):
            by_recv[int(r)].append(int(s))

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for node, senders_list in enumerate(by_recv):
        if not senders_list:
            out.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            continue
        senders = np.asarray(senders_list, dtype=np.int64)
        dists = _haversine_km(
            np.repeat(node_coords[node][None, :], len(senders), axis=0),
            node_coords[senders],
        ).astype(np.float32)
        order = np.argsort(dists)
        out.append((senders[order], dists[order]))
    return out


def _nearest_a1_features_from_gauge_map(
    y: np.ndarray,
    train_mask: np.ndarray,
    sorted_senders: list[tuple[np.ndarray, np.ndarray]],
    k_gauges: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    train_mask = np.asarray(train_mask).astype(bool) & np.isfinite(y)
    n_nodes = len(y)
    features = np.zeros((n_nodes, int(k_gauges) * 2), dtype=np.float32)

    for node in range(n_nodes):
        senders, dists = sorted_senders[node]
        if len(senders) == 0:
            continue
        active = train_mask[senders]
        if not np.any(active):
            continue
        active_senders = senders[active]
        active_dists = dists[active]
        take = min(int(k_gauges), len(active_senders))
        for rank in range(take):
            base = rank * 2
            sender = int(active_senders[rank])
            features[node, base] = float(y[sender])
            features[node, base + 1] = float(active_dists[rank])
    return features


def _select_quantile(ds: xr.Dataset, var: str, q: float) -> xr.DataArray:
    qvals = ds["quantile"].values.astype(float)
    idx = int(np.argmin(np.abs(qvals - float(q))))
    print(f"{var}: requested q={q:.3f}; nearest={float(qvals[idx]):.3f}; idx={idx}")
    return ds[var].isel(quantile=idx)


def _q_col(prefix: str, q: float) -> str:
    return f"{prefix}_q{int(round(float(q) * 100)):02d}"


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    nc_gatgcn = Path(cfg_get(cfg, "paths", "nc_gatgcn"))
    nc_mlp = Path(cfg_get(cfg, "paths", "nc_mlp"))
    model_path = Path(cfg_get(cfg, "paths", "model_path"))
    seq_dir = Path(cfg_get(cfg, "paths", "seq_dir"))
    graph_path = Path(cfg_get(cfg, "paths", "graph_path"))
    out_nc = Path(cfg_get(cfg, "paths", "out_nc"))

    k_gauges = int(cfg_get(cfg, "features", "k_gauges", 10))
    batch_size = int(cfg_get(cfg, "runtime", "batch_size", 16384))
    output_quantiles = [float(q) for q in cfg_get(cfg, "outputs", "quantiles", [0.65, 0.70, 0.75])]

    print("Opening GATGCN:", nc_gatgcn)
    ds_gat = xr.open_dataset(nc_gatgcn)
    print("Opening MLP:", nc_mlp)
    ds_mlp = xr.open_dataset(nc_mlp)
    if not np.array_equal(ds_gat["time"].values, ds_mlp["time"].values):
        raise ValueError("GATGCN and MLP time coordinates differ.")
    if not np.array_equal(ds_gat["node"].values, ds_mlp["node"].values):
        raise ValueError("GATGCN and MLP node coordinates differ.")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_cols = list(ckpt["feature_cols"])
    model = MetaMLP(
        in_dim=len(feature_cols),
        hidden_dims=[int(v) for v in ckpt["hidden_dims"]],
        dropout=float(ckpt["dropout"]),
        out_dim=len(ckpt["quantiles"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])
    quantiles = np.asarray(ckpt["quantiles"], dtype=np.float32)
    point_idx = int(ckpt["point_idx"])
    output_idx = {
        float(q): int(np.argmin(np.abs(quantiles - float(q)))) for q in output_quantiles
    }
    print("Meta quantiles:", quantiles.tolist())
    print("Point idx:", point_idx, float(quantiles[point_idx]))
    for q, idx in output_idx.items():
        print(f"Output q={q:.3f} -> nearest={float(quantiles[idx]):.3f}; idx={idx}")

    base_arrays = {
        "gatgcn_q65": _select_quantile(ds_gat, "pred_quantile", 0.65),
        "gatgcn_q70": _select_quantile(ds_gat, "pred_quantile", 0.70),
        "gatgcn_q75": _select_quantile(ds_gat, "pred_quantile", 0.75),
        "mlp_q65": _select_quantile(ds_mlp, "pred_quantile_mlp", 0.65),
        "mlp_q70": _select_quantile(ds_mlp, "pred_quantile_mlp", 0.70),
        "mlp_q75": _select_quantile(ds_mlp, "pred_quantile_mlp", 0.75),
    }

    times = pd.to_datetime(ds_gat["time"].values)
    n_time = len(times)
    n_nodes = int(ds_gat.sizes["node"])

    node_coords = _load_node_coords_from_graph(graph_path)
    if len(node_coords) != n_nodes:
        raise ValueError(f"Graph node count {len(node_coords)} != NetCDF nodes {n_nodes}")
    edge_index_a1 = _load_edge_index_a1(seq_dir)
    sorted_senders = _precompute_a1_sorted_senders(edge_index_a1, node_coords, n_nodes)

    pred_point = np.full((n_time, n_nodes), np.nan, dtype=np.float32)
    pred_q = {
        float(q): np.full((n_time, n_nodes), np.nan, dtype=np.float32)
        for q in output_quantiles
    }

    aux_cols = []
    for rank in range(1, k_gauges + 1):
        aux_cols.extend([f"gauge_{rank:02d}_value", f"gauge_{rank:02d}_distance_km"])

    for t_idx, ts in enumerate(tqdm(times, desc="Apply stack meta-MLP")):
        day = ts.strftime("%Y-%m-%d")
        x = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
        for col_i, col in enumerate(feature_cols):
            if col in base_arrays:
                x[:, col_i] = np.asarray(base_arrays[col].isel(time=t_idx).values, dtype=np.float32)
            elif col in aux_cols:
                continue
            else:
                raise KeyError(f"Unsupported stack feature column {col!r}")

        aux = _nearest_a1_features_from_gauge_map(
            y=np.asarray(ds_gat["train_gauge"].isel(time=t_idx).values, dtype=np.float32),
            train_mask=np.asarray(ds_gat["train_mask"].isel(time=t_idx).values),
            sorted_senders=sorted_senders,
            k_gauges=k_gauges,
        )
        for rank in range(1, k_gauges + 1):
            value_col = f"gauge_{rank:02d}_value"
            dist_col = f"gauge_{rank:02d}_distance_km"
            if value_col in feature_cols:
                x[:, feature_cols.index(value_col)] = aux[:, (rank - 1) * 2]
            if dist_col in feature_cols:
                x[:, feature_cols.index(dist_col)] = aux[:, (rank - 1) * 2 + 1]

        ok = np.all(np.isfinite(x), axis=1)
        xz = ((x[ok] - x_mean) / x_std).astype(np.float32)
        outs = []
        with torch.no_grad():
            for start in range(0, len(xz), batch_size):
                xb = torch.from_numpy(xz[start:start + batch_size]).to(device)
                outs.append(model(xb).detach().cpu().numpy())
        out_scaled = np.concatenate(outs, axis=0).astype(np.float32)
        out_mm = np.clip(out_scaled * y_std + y_mean, 0.0, None)
        pred_point[t_idx, ok] = out_mm[:, point_idx]
        for q, idx in output_idx.items():
            pred_q[q][t_idx, ok] = out_mm[:, idx]

    data_vars = {
        "pred_stack": (("time", "node"), pred_point),
        "node_lon": (("node",), ds_gat["node_lon"].values.astype(np.float32)),
        "node_lat": (("node",), ds_gat["node_lat"].values.astype(np.float32)),
    }
    for q in output_quantiles:
        data_vars[_q_col("pred_stack", q)] = (("time", "node"), pred_q[float(q)])

    out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ds_gat["time"].values,
            "node": ds_gat["node"].values.astype(np.int32),
        },
        attrs={
            "description": "StackCV PyTorch quantile meta-MLP applied to full GSI node maps.",
            "model_path": str(model_path),
            "nc_gatgcn": str(nc_gatgcn),
            "nc_mlp": str(nc_mlp),
            "seq_dir": str(seq_dir),
            "units": "mm/day",
        },
    )
    encoding = {
        name: {"zlib": True, "complevel": 2, "dtype": "float32", "chunksizes": (1, n_nodes)}
        for name in out.data_vars
        if out[name].dims == ("time", "node")
    }
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_nc, encoding=encoding)
    print("Saved stack grid NetCDF:", out_nc)

    ds_gat.close()
    ds_mlp.close()


if __name__ == "__main__":
    main()
