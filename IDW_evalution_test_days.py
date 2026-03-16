#!/usr/bin/env python3
# coding: utf-8

"""
Evaluate leave-one-out IDW baseline on test years only.

Method
------
1. Load gauge CSV
2. Map gauges to DEM/graph nodes
3. Aggregate duplicate gauge records by (nearest_node, DATE)
4. Restrict to test years
5. For each day:
   - use active gauge-nodes as donors
   - predict each node from the remaining donors only (leave-one-out)
6. Compute overall metrics

This gives a fair baseline against the GNN at gauge-supported locations.
"""

import rioxarray as rxr
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
DEM_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/DEM/ASTER_DEM_0p1deg_AZ_buffer.tif"
GAUGE_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/ghcn_data/ghcn_precip_2005_2024_AZ_buffer_50pct.csv"

OUT_CSV = "/xdisk/behrangi/omidzandi/GNNs/evaluation/idw_leave_one_out_test2020_2024.csv"

xmin, ymin, xmax, ymax = -115.25, 29.85, -106.45, 38.25

TEST_START = "2020-01-01"
TEST_END   = "2024-12-31"

IDW_K = 10
IDW_POWER = 2.0

# ============================================================
# METRICS
# ============================================================
def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]

def rmse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if o.size else np.nan

def mae(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(np.abs(s - o))) if o.size else np.nan

def cc(obs, sim):
    o, s = _paired(obs, sim)
    if o.size < 2:
        return np.nan
    so = np.std(o)
    ss = np.std(s)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def bias(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(s - o)) if o.size else np.nan

def kge(obs, sim):
    o, s = _paired(obs, sim)
    if o.size < 2:
        return np.nan

    mu_o = np.mean(o)
    mu_s = np.mean(s)
    sig_o = np.std(o, ddof=0)
    sig_s = np.std(s, ddof=0)

    r = cc(o, s)
    if not np.isfinite(r) or sig_o == 0 or mu_o == 0:
        return np.nan

    alpha = sig_s / sig_o
    beta = mu_s / mu_o
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

# ============================================================
# BUILD DEM NODE SET
# ============================================================
print("Loading DEM and building node coordinates...")
da = rxr.open_rasterio(DEM_PATH).squeeze()
da = da.where(~np.isnan(da), drop=True)
da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)

values = da.values
nrows, ncols = values.shape
xs, ys = da.x.values, da.y.values

G = nx.grid_2d_graph(nrows, ncols, periodic=False)

remove_nodes = []
for i in range(nrows):
    for j in range(ncols):
        elev = values[i, j]
        if np.isnan(elev):
            remove_nodes.append((i, j))
        else:
            G.nodes[(i, j)]["lon"] = float(xs[j])
            G.nodes[(i, j)]["lat"] = float(ys[i])

G.remove_nodes_from(remove_nodes)

node_ids = list(G.nodes())
node_coords = np.array([(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids], dtype=np.float32)
tree = cKDTree(node_coords, balanced_tree=False, compact_nodes=False)

print(f"DEM graph nodes available: {len(node_ids)}")

# ============================================================
# LOAD AND MAP GAUGES
# ============================================================
print("Reading gauge CSV...")
gauge_df = pd.read_csv(
    GAUGE_PATH,
    usecols=["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"]
)

gauge_df = gauge_df.dropna(subset=["LATITUDE", "LONGITUDE", "PRCP"])
gauge_df["DATE"] = pd.to_datetime(gauge_df["DATE"])
gauge_df["PRCP"] = pd.to_numeric(gauge_df["PRCP"], errors="coerce")
gauge_df = gauge_df.dropna(subset=["PRCP"])

# test period only
gauge_df = gauge_df[(gauge_df["DATE"] >= TEST_START) & (gauge_df["DATE"] <= TEST_END)].copy()

print(f"Rows in test period before mapping: {len(gauge_df):,}")

# map each gauge record to nearest DEM node
gauge_coords = gauge_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
_, idx = tree.query(gauge_coords, k=1)
gauge_df["nearest_node"] = [node_ids[i] for i in idx]

# aggregate duplicate gauges falling on same node/date
agg_df = (
    gauge_df.groupby(["nearest_node", "DATE"], as_index=False)[["PRCP", "LATITUDE", "LONGITUDE"]]
    .mean()
)

gauge_df = agg_df
gauge_df["DATE_STR"] = gauge_df["DATE"].dt.strftime("%Y-%m-%d")

print(f"Unique (nearest_node, DATE) rows after aggregation: {len(gauge_df):,}")

# attach node lon/lat
gauge_df["node_lon"] = gauge_df["nearest_node"].map(lambda n: G.nodes[n]["lon"])
gauge_df["node_lat"] = gauge_df["nearest_node"].map(lambda n: G.nodes[n]["lat"])

# ============================================================
# LEAVE-ONE-OUT IDW
# ============================================================
def idw_leave_one_out_for_day(day_df, k=10, power=2.0):
    """
    day_df: one-day dataframe at node level
    returns same dataframe with column IDW_LOO
    """
    n = len(day_df)
    out = np.full(n, np.nan, dtype=np.float32)

    if n < 2:
        tmp = day_df.copy()
        tmp["IDW_LOO"] = out
        return tmp

    coords = day_df[["node_lon", "node_lat"]].to_numpy(dtype=np.float64)
    vals = day_df["PRCP"].to_numpy(dtype=np.float64)

    donor_tree = cKDTree(coords, balanced_tree=False, compact_nodes=False)

    k_query = min(k + 1, n)
    dists, inds = donor_tree.query(coords, k=k_query)

    if k_query == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    for i in range(n):
        kept_d = []
        kept_v = []

        for dist, j in zip(dists[i], inds[i]):
            if j == i:
                continue  # leave-one-out

            kept_d.append(float(dist))
            kept_v.append(float(vals[j]))

            if len(kept_v) == k:
                break

        if len(kept_v) == 0:
            out[i] = np.nan
            continue

        kept_d = np.asarray(kept_d, dtype=np.float64)
        kept_v = np.asarray(kept_v, dtype=np.float64)

        if np.any(kept_d == 0.0):
            out[i] = np.float32(kept_v[kept_d == 0.0][0])
        else:
            w = 1.0 / np.power(kept_d, power)
            out[i] = np.float32(np.sum(w * kept_v) / np.sum(w))

    tmp = day_df.copy()
    tmp["IDW_LOO"] = out
    return tmp

print("Running leave-one-out IDW day by day...")
daily_groups = gauge_df.groupby("DATE", sort=True)

out_rows = []
for day, day_df in tqdm(daily_groups, total=gauge_df["DATE"].nunique(), desc="Daily LOO-IDW"):
    out_rows.append(idw_leave_one_out_for_day(day_df, k=IDW_K, power=IDW_POWER))

res_df = pd.concat(out_rows, ignore_index=True)

# ============================================================
# SAVE + REPORT
# ============================================================
out_path = Path(OUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)
res_df.to_csv(out_path, index=False)

obs = res_df["PRCP"].to_numpy(dtype=np.float64)
sim = res_df["IDW_LOO"].to_numpy(dtype=np.float64)

n_eval = len(_paired(obs, sim)[0])

print("\n======================================")
print("Leave-one-out IDW baseline (test years)")
print("======================================")
print(f"Test period      : {TEST_START} to {TEST_END}")
print(f"K neighbors      : {IDW_K}")
print(f"IDW power        : {IDW_POWER}")
print(f"N evaluated      : {n_eval:,}")
print(f"RMSE             : {rmse(obs, sim):.4f}")
print(f"MAE              : {mae(obs, sim):.4f}")
print(f"CC               : {cc(obs, sim):.4f}")
print(f"KGE              : {kge(obs, sim):.4f}")
print(f"Bias             : {bias(obs, sim):.4f}")
print(f"Saved CSV        : {out_path}")