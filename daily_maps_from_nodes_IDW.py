#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import pickle
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
GRAPH_PKL = "/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels_avg10.pkl"
SEQ_DIR   = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T030_all_years_avg10"
OUT_NC    = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/IDW_test2020_2024/pred_inputs_daily_maps_IDW.nc"

TEST_START = "2020-01-01"
TEST_END   = "2024-12-31"

IDW_K = 10
N_WORKERS = min(os.cpu_count(), 16)

# ============================================================
# HELPERS
# ============================================================
def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _node_list_from_graph(G):
    return sorted(G.nodes())


def _make_static_maps(G, node_list, ny, nx):
    lon2d  = np.full((ny, nx), np.nan, dtype=np.float32)
    lat2d  = np.full((ny, nx), np.nan, dtype=np.float32)
    elev2d = np.full((ny, nx), np.nan, dtype=np.float32)
    valid  = np.zeros((ny, nx), dtype=np.uint8)

    for (i, j) in node_list:
        attrs = G.nodes[(i, j)]
        valid[i, j] = 1
        lon2d[i, j] = np.float32(attrs["lon"])
        lat2d[i, j] = np.float32(attrs["lat"])
        elev2d[i, j] = np.float32(attrs["elevation"])

    return lon2d, lat2d, elev2d, valid


def _load_inputs_from_seq(seq_dir: Path, dates: np.ndarray, N: int):
    seq_files = sorted(seq_dir.glob("seq_*.pt"))
    if not seq_files:
        raise RuntimeError(f"No seq_*.pt found in: {seq_dir}")

    date_to_file = {}
    for f in seq_files:
        d = _load_pt(f)
        date_to_file[str(d["date"])] = f

    T = len(dates)
    era5_node = np.full((T, N), np.nan, dtype=np.float32)
    imerg_node = np.full((T, N), np.nan, dtype=np.float32)

    for t, day in enumerate(dates):
        f = date_to_file.get(str(day), None)
        if f is None:
            continue
        d = _load_pt(f)
        x = d["x"].numpy().astype(np.float32)   # [Twin, N, F]
        x_last = x[-1]                          # [N, F]
        era5_node[t, :]  = x_last[:, 0]
        imerg_node[t, :] = x_last[:, 1]

    return era5_node, imerg_node


def _map_nodes_to_grid(node_list, ny, nx, node_arr):
    T, N = node_arr.shape
    out = np.full((T, ny, nx), np.nan, dtype=np.float32)
    for k, (i, j) in enumerate(node_list):
        out[:, i, j] = node_arr[:, k]
    return out


def _idw_from_donors(query_coords, donor_coords, donor_vals, k=10, power=2.0, exclude_self_index=None):
    """
    query_coords: [Q, 2]
    donor_coords: [D, 2]
    donor_vals  : [D]
    exclude_self_index:
        - None for full-map interpolation
        - integer array of shape [Q] giving donor index to exclude for each query
    """
    Q = len(query_coords)
    out = np.full(Q, np.nan, dtype=np.float32)

    if len(donor_coords) == 0:
        return out

    k_query = min(k + 1, len(donor_coords))
    tree = cKDTree(donor_coords, balanced_tree=False, compact_nodes=False)
    dists, inds = tree.query(query_coords, k=k_query)

    if k_query == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    for qi in range(Q):
        kept_d = []
        kept_v = []

        for dist, jj in zip(dists[qi], inds[qi]):
            if exclude_self_index is not None and jj == exclude_self_index[qi]:
                continue

            kept_d.append(float(dist))
            kept_v.append(float(donor_vals[jj]))

            if len(kept_v) == k:
                break

        if len(kept_v) == 0:
            continue

        kept_d = np.asarray(kept_d, dtype=np.float64)
        kept_v = np.asarray(kept_v, dtype=np.float64)

        if np.any(kept_d == 0.0):
            out[qi] = np.float32(kept_v[kept_d == 0.0][0])
        else:
            if power == 0:
                out[qi] = np.float32(np.mean(kept_v))
            else:
                w = 1.0 / np.power(kept_d, power)
                out[qi] = np.float32(np.sum(w * kept_v) / np.sum(w))

    return out


def _process_one_day(args):
    (
        t_idx,
        day_str,
        day_nodes,
        day_vals,
        day_coords,
        node_list,
        node_coords_all,
        node_to_k,
        ny,
        nx,
        idw_k,
    ) = args

    N = len(node_list)

    gauge_map_day = np.full((ny, nx), np.nan, dtype=np.float32)
    gauge_mask_day = np.zeros((ny, nx), dtype=np.uint8)

    idw_loo_p0_day = np.full(N, np.nan, dtype=np.float32)
    idw_loo_p2_day = np.full(N, np.nan, dtype=np.float32)
    idw_map_p0_day = np.full(N, np.nan, dtype=np.float32)
    idw_map_p2_day = np.full(N, np.nan, dtype=np.float32)

    # fill gauge maps
    for node, val in zip(day_nodes, day_vals):
        i, j = node
        gauge_map_day[i, j] = np.float32(val)
        gauge_mask_day[i, j] = 1

    D = len(day_nodes)
    if D == 0:
        return (t_idx, gauge_map_day, gauge_mask_day,
                idw_loo_p0_day, idw_loo_p2_day,
                idw_map_p0_day, idw_map_p2_day)

    # --------------------------------------------------
    # A) Leave-one-out at gauge-supported nodes only
    # --------------------------------------------------
    if D >= 2:
        query_coords_loo = day_coords
        exclude_idx = np.arange(D, dtype=int)

        loo_p0 = _idw_from_donors(
            query_coords_loo, day_coords, day_vals,
            k=idw_k, power=0.0,
            exclude_self_index=exclude_idx
        )
        loo_p2 = _idw_from_donors(
            query_coords_loo, day_coords, day_vals,
            k=idw_k, power=2.0,
            exclude_self_index=exclude_idx
        )

        for node, v0, v2 in zip(day_nodes, loo_p0, loo_p2):
            kk = node_to_k[node]
            idw_loo_p0_day[kk] = v0
            idw_loo_p2_day[kk] = v2

    # --------------------------------------------------
    # B) Full-map interpolation at all valid nodes
    # --------------------------------------------------
    full_p0 = _idw_from_donors(
        node_coords_all, day_coords, day_vals,
        k=idw_k, power=0.0,
        exclude_self_index=None
    )
    full_p2 = _idw_from_donors(
        node_coords_all, day_coords, day_vals,
        k=idw_k, power=2.0,
        exclude_self_index=None
    )

    idw_map_p0_day[:] = full_p0
    idw_map_p2_day[:] = full_p2

    return (t_idx, gauge_map_day, gauge_mask_day,
            idw_loo_p0_day, idw_loo_p2_day,
            idw_map_p0_day, idw_map_p2_day)


# ============================================================
# MAIN
# ============================================================
print("Loading graph...")
with open(GRAPH_PKL, "rb") as f:
    payload = pickle.load(f)

G = payload["graph"]
time_axis = payload["time_axis"]

dates = pd.to_datetime(time_axis)
keep = (dates >= pd.Timestamp(TEST_START)) & (dates <= pd.Timestamp(TEST_END))
dates_test = dates[keep].strftime("%Y-%m-%d").to_numpy()

node_list = _node_list_from_graph(G)
ny = max(i for i, _ in node_list) + 1
nx = max(j for _, j in node_list) + 1
N = len(node_list)

print(f"Nodes: {N}, grid: {ny} x {nx}, test days: {len(dates_test)}")

lon2d, lat2d, elev2d, valid = _make_static_maps(G, node_list, ny, nx)

node_to_k = {node: k for k, node in enumerate(node_list)}
node_coords_all = np.array(
    [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_list],
    dtype=np.float64
)

print("Loading ERA5 / IMERG from seq files...")
era5_node, imerg_node = _load_inputs_from_seq(Path(SEQ_DIR), dates_test, N)
era5_map = _map_nodes_to_grid(node_list, ny, nx, era5_node)
imerg_map = _map_nodes_to_grid(node_list, ny, nx, imerg_node)

# ------------------------------------------------------------
# Build day-wise gauge table from graph targets
# ------------------------------------------------------------
print("Collecting gauge targets from graph...")
date_to_t = {d: t for t, d in enumerate(dates_test)}

rows = []
for node in node_list:
    tdict = G.nodes[node].get("target", None)
    if not tdict:
        continue
    lon = float(G.nodes[node]["lon"])
    lat = float(G.nodes[node]["lat"])
    for day_str, val in tdict.items():
        if day_str not in date_to_t:
            continue
        rows.append((day_str, node, float(val), lon, lat))

daily_df = pd.DataFrame(rows, columns=["DATE_STR", "node", "PRCP", "lon", "lat"])
print(f"Gauge-supported node/day rows in test period: {len(daily_df):,}")

# Prepare output arrays
T = len(dates_test)
gauge_map = np.full((T, ny, nx), np.nan, dtype=np.float32)
gauge_mask = np.zeros((T, ny, nx), dtype=np.uint8)

idw_loo_p0_node = np.full((T, N), np.nan, dtype=np.float32)
idw_loo_p2_node = np.full((T, N), np.nan, dtype=np.float32)
idw_map_p0_node = np.full((T, N), np.nan, dtype=np.float32)
idw_map_p2_node = np.full((T, N), np.nan, dtype=np.float32)

# Build tasks
tasks = []
for day_str, day_df in daily_df.groupby("DATE_STR", sort=True):
    t_idx = date_to_t[day_str]
    day_nodes = day_df["node"].tolist()
    day_vals = day_df["PRCP"].to_numpy(dtype=np.float64)
    day_coords = day_df[["lon", "lat"]].to_numpy(dtype=np.float64)

    tasks.append((
        t_idx,
        day_str,
        day_nodes,
        day_vals,
        day_coords,
        node_list,
        node_coords_all,
        node_to_k,
        ny,
        nx,
        IDW_K,
    ))

print(f"Running parallel IDW over {len(tasks)} days with {N_WORKERS} workers...")

with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(_process_one_day, task) for task in tasks]

    for fut in tqdm(as_completed(futures), total=len(futures), desc="Daily IDW"):
        (
            t_idx,
            gauge_map_day,
            gauge_mask_day,
            idw_loo_p0_day,
            idw_loo_p2_day,
            idw_map_p0_day,
            idw_map_p2_day,
        ) = fut.result()

        gauge_map[t_idx] = gauge_map_day
        gauge_mask[t_idx] = gauge_mask_day
        idw_loo_p0_node[t_idx] = idw_loo_p0_day
        idw_loo_p2_node[t_idx] = idw_loo_p2_day
        idw_map_p0_node[t_idx] = idw_map_p0_day
        idw_map_p2_node[t_idx] = idw_map_p2_day

# Map node arrays to grids
idw_loo_p0_map = _map_nodes_to_grid(node_list, ny, nx, idw_loo_p0_node)
idw_loo_p2_map = _map_nodes_to_grid(node_list, ny, nx, idw_loo_p2_node)
idw_map_p0_map = _map_nodes_to_grid(node_list, ny, nx, idw_map_p0_node)
idw_map_p2_map = _map_nodes_to_grid(node_list, ny, nx, idw_map_p2_node)

coords = {
    "time": pd.to_datetime(dates_test).to_numpy(),
    "y": np.arange(ny, dtype=np.int32),
    "x": np.arange(nx, dtype=np.int32),
}

data_vars = {
    "imerg":       (("time", "y", "x"), imerg_map),
    "era5":        (("time", "y", "x"), era5_map),
    "valid_pixel": (("y", "x"), valid),
    "elevation":   (("y", "x"), elev2d),
    "lon":         (("y", "x"), lon2d),
    "lat":         (("y", "x"), lat2d),
    "gauge":       (("time", "y", "x"), gauge_map),
    "gauge_mask":  (("time", "y", "x"), gauge_mask),

    # leave-one-out evaluation products
    "idw_loo_p0":  (("time", "y", "x"), idw_loo_p0_map),
    "idw_loo_p2":  (("time", "y", "x"), idw_loo_p2_map),

    # full estimated fields
    "idw_map_p0":  (("time", "y", "x"), idw_map_p0_map),
    "idw_map_p2":  (("time", "y", "x"), idw_map_p2_map),
}

ds = xr.Dataset(
    data_vars=data_vars,
    coords=coords,
    attrs={
        "description": "Daily maps: IDW baselines + IMERG/ERA5 + gauge mapped onto DEM grid",
        "units_pred": "mm/day",
        "units_imerg": "mm/day",
        "units_era5": "mm/day",
        "units_gauge": "mm/day",
        "graph_pkl": str(GRAPH_PKL),
        "seq_dir": str(SEQ_DIR),
        "test_period": f"{TEST_START} to {TEST_END}",
        "idw_k": IDW_K,
        "idw_power_0": 0,
        "idw_power_2": 2,
        "idw_loo_note": "Leave-one-out at gauge-supported nodes only",
        "idw_map_note": "Full-field interpolation using all active gauges of the day",
    },
)

enc = {
    "imerg":       {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "era5":        {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "gauge":       {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "gauge_mask":  {"zlib": True, "complevel": 4, "dtype": "uint8",   "chunksizes": (1, ny, nx)},
    "idw_loo_p0":  {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "idw_loo_p2":  {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "idw_map_p0":  {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "idw_map_p2":  {"zlib": True, "complevel": 4, "dtype": "float32", "chunksizes": (1, ny, nx)},
    "valid_pixel": {"zlib": True, "complevel": 4, "dtype": "uint8"},
    "elevation":   {"zlib": True, "complevel": 4, "dtype": "float32"},
    "lon":         {"zlib": True, "complevel": 4, "dtype": "float32"},
    "lat":         {"zlib": True, "complevel": 4, "dtype": "float32"},
}

out_path = Path(OUT_NC)
out_path.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(out_path, encoding=enc)

print(f"\nWrote: {out_path}")
print(f"time: {dates_test[0]} -> {dates_test[-1]} (T={len(dates_test)})")
print(f"grid: ny={ny}, nx={nx}, nodes={N}")
print(f"gauge points total: {int(gauge_mask.sum())}")
print(f"idw_loo_p0 shape: {ds['idw_loo_p0'].shape}")
print(f"idw_loo_p2 shape: {ds['idw_loo_p2'].shape}")
print(f"idw_map_p0 shape: {ds['idw_map_p0'].shape}")
print(f"idw_map_p2 shape: {ds['idw_map_p2'].shape}")