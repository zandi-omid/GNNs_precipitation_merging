#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import torch


# -------------------------
# Helpers
# -------------------------
def _load_pt(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_preds_npz(npz_path: Path, tau_sel: Optional[list[float]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      dates: (T,) str
      data: dict with keys among:
        - pred_det: (T,N)  (for MSE/Huber etc)
        - pred_median: (T,N)
        - pred_expected_mean: (T,N)
        - tau: (K,)
        - pred_q: (T,N,K)
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    dates = z["dates"].astype(str)

    out: Dict[str, np.ndarray] = {}

    # ---- Deterministic case (older scripts) ----
    # some runs might have "preds" (T,N)
    if "preds" in keys:
        out["pred_det"] = z["preds"].astype(np.float32)

    # ---- Newer deterministic case could use pred_point ----
    if "pred_point" in keys and "pred_det" not in out:
        # In quantile mode pred_point is median; in mse mode it is just the prediction
        out["pred_det"] = z["pred_point"].astype(np.float32)

    # ---- Quantile extras (if present) ----
    if "pred_point" in keys:
        out["pred_median"] = z["pred_point"].astype(np.float32)

    if "pred_mean" in keys:
        out["pred_expected_mean"] = z["pred_mean"].astype(np.float32)

    # Selected quantiles
    if "pred_q_sel" in keys and "tau_sel" in keys:
        tau_file = z["tau_sel"].astype(np.float32)  # (K,)
        pred_q = z["pred_q_sel"].astype(np.float32)  # (T,N,K)

        if tau_sel is None or len(tau_sel) == 0:
            out["tau"] = tau_file
            out["pred_q"] = pred_q
        else:
            # pick closest indices to requested tau_sel
            tau_req = np.array(tau_sel, dtype=np.float32)
            idx = np.array([int(np.argmin(np.abs(tau_file - t))) for t in tau_req], dtype=int)
            out["tau"] = tau_file[idx]
            out["pred_q"] = pred_q[:, :, idx]

    return dates, out


def _node_list_from_graph(G) -> list[tuple[int, int]]:
    # IMPORTANT: must match seq-builder node ordering.
    # If your seq builder used sorted(common_nodes), keep this consistent.
    return sorted(G.nodes())


def _make_static_maps(G, node_list, ny, nx):
    lon2d  = np.full((ny, nx), np.nan, dtype=np.float32)
    lat2d  = np.full((ny, nx), np.nan, dtype=np.float32)
    elev2d = np.full((ny, nx), np.nan, dtype=np.float32)
    valid  = np.zeros((ny, nx), dtype=np.uint8)

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


def _load_inputs_from_seq(seq_dir: Path, dates: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load x[-1] ERA5/IMERG from seq_*.pt for each date.
    Returns:
      era5_node_np:  (T,N)
      imerg_node_np: (T,N)
    """
    seq_files = sorted(seq_dir.glob("seq_*.pt"))
    if not seq_files:
        raise RuntimeError(f"No seq_*.pt found in: {seq_dir}")

    date_to_file: Dict[str, Path] = {}
    for f in seq_files:
        d = _load_pt(f)
        date_to_file[str(d["date"])] = f

    T = len(dates)
    imerg_node_np = np.full((T, N), np.nan, dtype=np.float32)
    era5_node_np  = np.full((T, N), np.nan, dtype=np.float32)

    for t, day in enumerate(dates):
        f = date_to_file.get(str(day), None)
        if f is None:
            continue
        d = _load_pt(f)
        x = d["x"].numpy().astype(np.float32)   # [Twin, N, F]
        x_last = x[-1]                          # [N, F]
        # feature order: [ERA5, IMERG]
        era5_node_np[t, :]  = x_last[:, 0]
        imerg_node_np[t, :] = x_last[:, 1]

    return era5_node_np, imerg_node_np


def _map_nodes_to_grid(node_list, ny, nx, node_arr: np.ndarray) -> np.ndarray:
    """
    node_arr: (T,N) -> grid: (T,ny,nx)
    """
    T, N = node_arr.shape
    out = np.full((T, ny, nx), np.nan, dtype=np.float32)
    for k, (i, j) in enumerate(node_list):
        out[:, i, j] = node_arr[:, k]
    return out


def _map_nodes_to_grid_q(node_list, ny, nx, node_arr_q: np.ndarray) -> np.ndarray:
    """
    node_arr_q: (T,N,K) -> grid: (T,ny,nx,K)
    """
    T, N, K = node_arr_q.shape
    out = np.full((T, ny, nx, K), np.nan, dtype=np.float32)
    for k_node, (i, j) in enumerate(node_list):
        out[:, i, j, :] = node_arr_q[:, k_node, :]
    return out


def _build_gauge_maps(G, node_list, dates: np.ndarray, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      gauge_map:  (T,ny,nx)
      gauge_mask: (T,ny,nx) uint8
    """
    T = len(dates)
    gauge_mask = np.zeros((T, ny, nx), dtype=np.uint8)
    gauge_map  = np.full((T, ny, nx), np.nan, dtype=np.float32)

    # date string -> time index
    date_to_t = {d: t for t, d in enumerate(pd.to_datetime(dates).strftime("%Y-%m-%d"))}

    for (i, j) in node_list:
        tdict = G.nodes[(i, j)].get("target", None)  # { "YYYY-MM-DD": value }
        if not tdict:
            continue
        for day_str, val in tdict.items():
            t = date_to_t.get(day_str, None)
            if t is None:
                continue
            gauge_mask[t, i, j] = 1
            gauge_map[t, i, j]  = np.float32(val)

    return gauge_map, gauge_mask


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Create daily (time,y,x) maps from node predictions + inputs.")
    parser.add_argument("--graph-pkl", type=str, required=True)
    parser.add_argument("--pred-npz", type=str, required=True)
    parser.add_argument("--seq-dir", type=str, required=True)
    parser.add_argument("--out-nc", type=str, required=True)

    parser.add_argument(
        "--tau-sel",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        help="Requested quantiles to write (uses closest available tau_sel in NPZ).",
    )

    parser.add_argument("--save-quantiles", action="store_true", help="Write pred_q (selected quantiles) if available.")
    parser.add_argument("--no-expected-mean", action="store_true", help="Do not write pred_expected_mean even if available.")
    parser.add_argument("--no-median", action="store_true", help="Do not write pred_median even if available.")
    parser.add_argument("--save-det", action="store_true", help="Write pred_det (for MSE runs) even if median exists.")

    args = parser.parse_args()

    graph_pkl = Path(args.graph_pkl)
    pred_npz  = Path(args.pred_npz)
    seq_dir   = Path(args.seq_dir)
    out_nc    = Path(args.out_nc)

    # 1) Load graph
    with open(graph_pkl, "rb") as f:
        payload = pickle.load(f)
    G = payload["graph"]

    node_list = _node_list_from_graph(G)
    ny = max(i for i, _ in node_list) + 1
    nx = max(j for _, j in node_list) + 1
    N = len(node_list)

    lon2d, lat2d, elev2d, valid = _make_static_maps(G, node_list, ny, nx)

    # 2) Load predictions NPZ (flexible)
    dates_str, pred_pack = _read_preds_npz(pred_npz, tau_sel=args.tau_sel)
    T = len(dates_str)

    # sanity check N
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
    if "pred_q" in pred_pack:
        if pred_pack["pred_q"].shape[1] != N:
            raise RuntimeError(
                f"Mismatch: pred_q has N={pred_pack['pred_q'].shape[1]} but graph has N={N}."
            )

    # time coord
    time = pd.to_datetime(dates_str).to_numpy()

    # 3) Load x[-1] inputs (ERA5/IMERG)
    era5_node_np, imerg_node_np = _load_inputs_from_seq(seq_dir, dates_str, N)

    # 4) Map to grids
    era5_map  = _map_nodes_to_grid(node_list, ny, nx, era5_node_np)
    imerg_map = _map_nodes_to_grid(node_list, ny, nx, imerg_node_np)

    data_vars = {
        "imerg": (("time", "y", "x"), imerg_map),
        "era5":  (("time", "y", "x"), era5_map),
        "valid_pixel": (("y", "x"), valid),
        "elevation":   (("y", "x"), elev2d),
        "lon":         (("y", "x"), lon2d),
        "lat":         (("y", "x"), lat2d),
    }

    # Predictions
    if "pred_median" in pred_pack and not args.no_median:
        pred_median_map = _map_nodes_to_grid(node_list, ny, nx, pred_pack["pred_median"])
        data_vars["pred_median"] = (("time", "y", "x"), pred_median_map)

    if "pred_expected_mean" in pred_pack and not args.no_expected_mean:
        pred_mean_map = _map_nodes_to_grid(node_list, ny, nx, pred_pack["pred_expected_mean"])
        data_vars["pred_expected_mean"] = (("time", "y", "x"), pred_mean_map)

    if "pred_det" in pred_pack and args.save_det:
        pred_det_map = _map_nodes_to_grid(node_list, ny, nx, pred_pack["pred_det"])
        data_vars["pred_det"] = (("time", "y", "x"), pred_det_map)

    if args.save_quantiles and ("pred_q" in pred_pack) and ("tau" in pred_pack):
        pred_q_map = _map_nodes_to_grid_q(node_list, ny, nx, pred_pack["pred_q"])  # (T,ny,nx,K)
        data_vars["pred_q"] = (("time", "y", "x", "tau"), pred_q_map)

    # 4b) Gauge maps
    gauge_map, gauge_mask = _build_gauge_maps(G, node_list, dates_str, ny, nx)
    data_vars["gauge"] = (("time", "y", "x"), gauge_map)
    data_vars["gauge_mask"] = (("time", "y", "x"), gauge_mask)

    coords = {
        "time": time,
        "y": np.arange(ny, dtype=np.int32),
        "x": np.arange(nx, dtype=np.int32),
    }
    if args.save_quantiles and ("tau" in pred_pack):
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

    # 5) Encoding
    enc = {
        "imerg": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "era5":  {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "gauge": {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "gauge_mask": {"zlib": True, "complevel": 4, "dtype": "uint8", "chunksizes": (1, ny, nx)},
        "valid_pixel": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "elevation":   {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lon":         {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat":         {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    for name in ["pred_median", "pred_expected_mean", "pred_det"]:
        if name in ds.data_vars:
            enc[name] = {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)}

    if "pred_q" in ds.data_vars:
        enc["pred_q"] = {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx, len(ds["tau"]))}

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc, encoding=enc)

    # Summary print
    print(f"✅ Wrote: {out_nc}")
    print(f"   time: {dates_str[0]} -> {dates_str[-1]} (T={T})")
    print(f"   grid: ny={ny}, nx={nx}, nodes={N}")
    print(f"   gauge points total (mask sum): {int(gauge_mask.sum())}")

    if "pred_median" in ds.data_vars:
        print("   pred_median:", ds["pred_median"].shape)
    if "pred_expected_mean" in ds.data_vars:
        print("   pred_expected_mean:", ds["pred_expected_mean"].shape)
    if "pred_det" in ds.data_vars:
        print("   pred_det:", ds["pred_det"].shape)
    if "pred_q" in ds.data_vars:
        print("   pred_q:", ds["pred_q"].shape, "tau:", ds["tau"].values)


if __name__ == "__main__":
    main()