#!/usr/bin/env python3
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import torch

# -------------------------
# INPUTS (edit if needed)
# -------------------------
GRAPH_PKL = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels.pkl")

PRED_NPZ  = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_log_normal_test2020_2024/preds_merged.npz")

OUT_NC = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_log_normal_test2020_2024/pred_inputs_daily_maps.nc")


def main():
    # -------------------------
    # 1) Load graph
    # -------------------------
    with open(GRAPH_PKL, "rb") as f:
        payload = pickle.load(f)

    G = payload["graph"]

    # IMPORTANT: ideally this should match the node order used in seq-building.
    node_list = sorted(G.nodes())  # list[(i,j)]

    ny = max(i for i, _ in node_list) + 1
    nx = max(j for _, j in node_list) + 1
    N = len(node_list)

    # Static maps
    lon2d  = np.full((ny, nx), np.nan, dtype=np.float32)
    lat2d  = np.full((ny, nx), np.nan, dtype=np.float32)
    elev2d = np.full((ny, nx), np.nan, dtype=np.float32)
    valid  = np.zeros((ny, nx), dtype=np.uint8)

    # Fill static maps if node attrs exist
    for (i, j) in node_list:
        attrs = G.nodes[(i, j)]
        valid[i, j] = 1
        if "lon" in attrs:
            lon2d[i, j] = np.float32(attrs["lon"])
        if "lat" in attrs:
            lat2d[i, j] = np.float32(attrs["lat"])
        # common keys you might have used:
        if "elevation" in attrs:
            elev2d[i, j] = np.float32(attrs["elevation"])
        elif "elev" in attrs:
            elev2d[i, j] = np.float32(attrs["elev"])

    # -------------------------
    # 2) Load predictions
    # -------------------------
    p = np.load(PRED_NPZ, allow_pickle=True)
    dates = p["dates"].astype(str)          # (T,)
    preds = p["preds"].astype(np.float32)   # (T,N)
    print("N nodes in graph:", len(node_list), "N nodes in preds:", preds.shape[1])

    T = preds.shape[0]
    if preds.shape[1] != N:
        raise RuntimeError(
            f"Mismatch: preds has N={preds.shape[1]} but graph has N={N}. "
            "This means node ordering differs. Use the exact common_nodes ordering from seq builder."
        )

    # Use datetime64 for xarray
    time = pd.to_datetime(dates).to_numpy()

    # -------------------------
    # 3) Load x[-1] inputs (ERA5/IMERG) from the SAME seq_*.pt files
    # -------------------------
    SEQ_DIR = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T030_all_years")

    # Build a date -> filename map
    seq_files = sorted(SEQ_DIR.glob("seq_*.pt"))
    date_to_file = {}
    for f in seq_files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        date_to_file[str(d["date"])] = f  # "YYYY-MM-DD"

    # Pre-allocate (T, N)
    imerg_node_np = np.full((T, N), np.nan, dtype=np.float32)
    era5_node_np  = np.full((T, N), np.nan, dtype=np.float32)

    for t, day in enumerate(dates):
        f = date_to_file.get(day, None)
        if f is None:
            continue
        d = torch.load(f, map_location="cpu", weights_only=True)
        x = d["x"].numpy().astype(np.float32)   # [Twin, N, F]
        x_last = x[-1]                          # [N, F]  <-- target day t
        # feature order: [ERA5, IMERG]
        era5_node_np[t, :]  = x_last[:, 0]
        imerg_node_np[t, :] = x_last[:, 1]

    # -------------------------
    # 4) Map node-vectors into (time, y, x)
    # -------------------------
    pred_map  = np.full((T, ny, nx), np.nan, dtype=np.float32)
    imerg_map = np.full((T, ny, nx), np.nan, dtype=np.float32)
    era5_map  = np.full((T, ny, nx), np.nan, dtype=np.float32)

    for k, (i, j) in enumerate(node_list):
        pred_map[:, i, j]  = preds[:, k]
        imerg_map[:, i, j] = imerg_node_np[:, k]
        era5_map[:, i, j]  = era5_node_np[:, k]

    # -------------------------
    # 4b) Gauge mask + gauge values (NEW)
    # -------------------------
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

    # -------------------------
    # 5) Write NetCDF
    # -------------------------
    ds = xr.Dataset(
        data_vars={
            "pred":       (("time", "y", "x"), pred_map),
            "imerg":      (("time", "y", "x"), imerg_map),
            "era5":       (("time", "y", "x"), era5_map),

            "gauge":      (("time", "y", "x"), gauge_map),      # NEW
            "gauge_mask": (("time", "y", "x"), gauge_mask),

            "valid_pixel": (("y", "x"), valid),
            "elevation":   (("y", "x"), elev2d),
            "lon":         (("y", "x"), lon2d),
            "lat":         (("y", "x"), lat2d),
        },
        coords={
            "time": time,
            "y": np.arange(ny, dtype=np.int32),
            "x": np.arange(nx, dtype=np.int32),
        },
        attrs={
            "description": "Daily maps: TGCN pred + IMERG/ERA5 + gauge mapped onto DEM grid",
            "units_pred": "mm/day",
            "units_imerg": "mm/day",
            "units_era5": "mm/day",
            "units_gauge": "mm/day",
        },
    )

    enc = {
        # chunk by day; compress
        "pred":       {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "imerg":      {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "era5":       {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
        "gauge":      {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},  # NEW
        "gauge_mask": {"zlib": True, "complevel": 4, "dtype": "uint8",   "chunksizes": (1, ny, nx)},

        "valid_pixel": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "elevation":   {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lon":         {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat":         {"zlib": True, "complevel": 4, "dtype": "float32"},
    }

    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT_NC, encoding=enc)

    print(f"✅ Wrote: {OUT_NC}")
    print(f"   pred/imerg/era5/gauge: {pred_map.shape} (time,y,x)")
    print(f"   time: {str(dates[0])} -> {str(dates[-1])}  (T={T})")
    print(f"   grid: ny={ny}, nx={nx}, nodes={N}")
    print(f"   gauge points total (mask sum): {int(gauge_mask.sum())}")


if __name__ == "__main__":
    main()