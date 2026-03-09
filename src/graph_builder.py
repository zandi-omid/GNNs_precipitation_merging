#!/usr/bin/env python3
# coding: utf-8
"""
🌎 Unified Graph Builder for GNN Precipitation Project (Final Optimized)
-----------------------------------------------------------------------
1️⃣ Load DEM and create 8-neighbor grid graph
2️⃣ Add static features (lat, lon, elevation)
3️⃣ Add dynamic daily features (ERA5 & IMERG)
4️⃣ Map gauges → DEM pixels (KDTree, fast on full grid)
5️⃣ Assign precipitation targets (labels)
6️⃣ Remove non-land (NaN ERA5) nodes right before export

✅ Outputs:
  graph_with_features_labels.pkl:
    {
        "graph": networkx.Graph,
        "time_axis": list[str],
        "dynamic_feature_names": ["ERA5_precip_mm_day", "IMERG_precip_mm_day", "IDW"]
    }

Author: Omid Zandi
Date: 2025-11-04
"""

# ============================================================
# 📦 Imports
# ============================================================
import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm
from pathlib import Path
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ============================================================
# ⚙️ Configurations
# ============================================================
DEM_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/DEM/ASTER_DEM_0p1deg_AZ_buffer.tif"
ERA5_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/ERA5_daily/ERA5_DAILY_2005_2024_MERGED.nc"
IMERG_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/IMERG_daily/IMERG_DAILY_2005_2024_MERGED.nc"
GAUGE_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/ghcn_data/ghcn_precip_2005_2024_AZ_buffer_50pct.csv"

GRAPH_OUT = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels_IDW5.pkl")
xmin, ymin, xmax, ymax = -115.25, 29.85, -106.45, 38.25

IDW_K = 5
IDW_POWER = 2.0



def compute_idw_for_day(args):
    """
    Compute leakage-safe IDW values for all nodes for one day.

    Returns
    -------
    t_idx : int
    out   : np.ndarray of shape [N]
    """
    (
        t_idx,
        donor_nodes,
        donor_vals,
        donor_coords,
        node_ids,
        node_coords,
        idw_k,
        idw_power,
    ) = args

    from scipy.spatial import cKDTree
    import numpy as np

    N = len(node_ids)
    out = np.full(N, np.nan, dtype=np.float32)

    if len(donor_nodes) == 0:
        return t_idx, out

    k_use = min(idw_k + 1, len(donor_nodes))
    donor_tree = cKDTree(donor_coords, balanced_tree=False, compact_nodes=False)

    dists, inds = donor_tree.query(node_coords, k=k_use)

    if k_use == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    for qi, qnode in enumerate(node_ids):
        q_dists = dists[qi]
        q_inds = inds[qi]

        kept_d = []
        kept_v = []

        for dist, ind in zip(q_dists, q_inds):
            donor_node = donor_nodes[ind]

            # leakage prevention
            if donor_node == qnode:
                continue

            kept_d.append(float(dist))
            kept_v.append(float(donor_vals[ind]))

            if len(kept_v) == idw_k:
                break

        if len(kept_v) == 0:
            out[qi] = np.nan
        else:
            kept_d = np.array(kept_d, dtype=np.float32)
            kept_v = np.array(kept_v, dtype=np.float32)

            if np.any(kept_d == 0.0):
                out[qi] = np.float32(kept_v[kept_d == 0.0][0])
            else:
                w = 1.0 / np.power(kept_d, idw_power)
                out[qi] = np.float32(np.sum(w * kept_v) / np.sum(w))

    return t_idx, out

if __name__ == "__main__":

    # ============================================================
    # 1️⃣ Load DEM and build grid-based graph
    # ============================================================
    print("🗺️ Loading DEM and constructing base grid graph...")
    da = rxr.open_rasterio(DEM_PATH).squeeze()
    da = da.where(~np.isnan(da), drop=True)
    da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)

    values = da.values
    nrows, ncols = values.shape
    xs, ys = da.x.values, da.y.values

    G = nx.grid_2d_graph(nrows, ncols, periodic=False)
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            G.add_edge((i, j), (i + 1, j + 1))
            G.add_edge((i + 1, j), (i, j + 1))

    # Add static node attributes
    remove_nodes = []
    for i in range(nrows):
        for j in range(ncols):
            elev = values[i, j]
            if np.isnan(elev):
                remove_nodes.append((i, j))
            else:
                G.nodes[(i, j)]["elevation"] = float(elev)
                G.nodes[(i, j)]["lon"] = float(xs[j])
                G.nodes[(i, j)]["lat"] = float(ys[i])

    G.remove_nodes_from(remove_nodes)
    print(f"✅ Initial DEM graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # ============================================================
    # 2️⃣ Load dynamic datasets (ERA5 & IMERG)
    # ============================================================
    print("💧 Loading ERA5 & IMERG daily time series...")
    era5 = xr.open_dataset(ERA5_PATH)["precipitation"]
    imerg = xr.open_dataset(IMERG_PATH)["precipitation"]

    # Align grids if needed
    if not np.array_equal(era5.x, imerg.x) or not np.array_equal(era5.y, imerg.y):
        print("🔁 Interpolating ERA5 to IMERG grid...")
        era5 = era5.interp(x=imerg.x, y=imerg.y)

    common_time = np.intersect1d(era5.time.values, imerg.time.values)
    era5 = era5.sel(time=common_time)
    imerg = imerg.sel(time=common_time)

    xs_e, ys_e = imerg.x.values.astype(np.float32), imerg.y.values.astype(np.float32)
    time_axis = pd.to_datetime(common_time).strftime("%Y-%m-%d").tolist()

    # ============================================================
    # 3️⃣ Assign dynamic features to nodes
    # ============================================================
    def nearest_index(arr, val): return int(np.abs(arr - val).argmin())

    # ============================================================
    # 4️⃣ Load and map gauge data (KDTree)
    # ============================================================
    print("📊 Mapping gauge data → DEM nodes...")
    gauge_df = pd.read_csv(GAUGE_PATH, usecols=["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"])
    gauge_df = gauge_df.dropna(subset=["LATITUDE", "LONGITUDE"])
    gauge_df["DATE"] = pd.to_datetime(gauge_df["DATE"])

    node_coords = np.array([(d["lon"], d["lat"]) for _, d in G.nodes(data=True)], dtype=np.float32)
    node_ids = list(G.nodes())
    tree = cKDTree(node_coords, balanced_tree=False, compact_nodes=False)

    gauge_coords = gauge_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    _, idx = tree.query(gauge_coords, k=1)
    gauge_df["nearest_node"] = [node_ids[i] for i in idx]

    print(f"✅ Gauge mapping complete ({len(gauge_df)} total records).")

    # ============================================================
    # 5️⃣ Aggregate duplicate gauge readings and assign targets
    # ============================================================
    dup_counts = gauge_df.groupby(["nearest_node", "DATE"]).size()
    n_dups = (dup_counts > 1).sum()

    print(f"🔍 Found {n_dups} duplicate (node, date) pairs → averaging them (fast mode)...")

    # ✅ Fast numeric aggregation (vectorized)
    agg_df = (
        gauge_df.groupby(["nearest_node", "DATE"], as_index=False)
        [["PRCP", "LATITUDE", "LONGITUDE"]]
        .mean()
    )

    # (Optional) Attach station list for duplicate pairs
    # print("⚙️ Combining station names for duplicates...")
    # dup_meta = (
    #     gauge_df.loc[gauge_df.duplicated(["nearest_node", "DATE"], keep=False), ["nearest_node", "DATE", "STATION"]]
    #     .groupby(["nearest_node", "DATE"])["STATION"]
    #     .apply(lambda x: ",".join(sorted(set(x.astype(str)))))
    #     .reset_index()
    # )
    # agg_df = agg_df.merge(dup_meta, on=["nearest_node", "DATE"], how="left")

    gauge_df = agg_df

    # ============================================================
    # 5b) Build leakage-safe daily IDW feature from gauges
    # ============================================================
    # Prepare empty time series for each node
    for node in G.nodes():
        G.nodes[node]["idw5_train"] = np.full(len(time_axis), np.nan, dtype=np.float32)

    # Query-node coordinates in the same order as node_ids
    node_ids = list(G.nodes())
    node_coords = np.array([(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids], dtype=np.float32)
    node_to_idx_all = {n: i for i, n in enumerate(node_ids)}

    # Group gauges by day
    gauge_df["DATE_STR"] = pd.to_datetime(gauge_df["DATE"]).dt.strftime("%Y-%m-%d")
    daily_groups = gauge_df.groupby("DATE_STR")


    print("🧮 Computing leakage-safe IDW5 daily gauge feature for all nodes (parallel)...")

    node_ids = list(G.nodes())
    node_coords = np.array([(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids], dtype=np.float32)

    # store all IDW values in one array first: [T, N]
    IDW_all = np.full((len(time_axis), len(node_ids)), np.nan, dtype=np.float32)

    # fast lookup for time index
    time_to_idx = {d: i for i, d in enumerate(time_axis)}

    # build task list
    tasks = []
    gauge_df["DATE_STR"] = pd.to_datetime(gauge_df["DATE"]).dt.strftime("%Y-%m-%d")
    daily_groups = gauge_df.groupby("DATE_STR")

    for day_str, day_df in daily_groups:
        if day_str not in time_to_idx:
            continue

        t_idx = time_to_idx[day_str]
        donor_nodes = day_df["nearest_node"].tolist()
        donor_vals = day_df["PRCP"].to_numpy(dtype=np.float32)
        donor_coords = np.array(
            [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in donor_nodes],
            dtype=np.float32
        )

        tasks.append((
            t_idx,
            donor_nodes,
            donor_vals,
            donor_coords,
            node_ids,
            node_coords,
            IDW_K,
            IDW_POWER,
        ))

    print(f"🚀 Launching parallel IDW over {len(tasks)} days...")

    n_workers = min(os.cpu_count(), 16)  # adjust if needed
    print(f"The number of workers are {int(n_workers)}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(compute_idw_for_day, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Daily IDW"):
            t_idx, out = fut.result()
            IDW_all[t_idx, :] = out

    # assign back to graph nodes
    for qi, node in enumerate(node_ids):
        G.nodes[node]["idw5_train"] = IDW_all[:, qi]

    print("✅ IDW feature computed.")

    # ============================================================
    # Assign final dynamic features to graph nodes
    # ============================================================
    print("📡 Assigning final dynamic daily features to graph nodes...")

    for node, d in tqdm(G.nodes(data=True), total=len(G.nodes()), desc="Assigning dynamic features"):
        lat, lon = float(d["lat"]), float(d["lon"])
        yi, xi = nearest_index(ys_e, lat), nearest_index(xs_e, lon)

        era5_ts = era5[:, yi, xi].values.astype(np.float32)
        imerg_ts = imerg[:, yi, xi].values.astype(np.float32)
        idw_ts = G.nodes[node]["idw5_train"].astype(np.float32)

        G.nodes[node]["dynamic_index"] = (yi, xi)
        G.nodes[node]["dynamic"] = np.stack([era5_ts, imerg_ts, idw_ts], axis=1)

    print(f"✅ Added dynamic features ({len(time_axis)} timesteps) to {len(G.nodes())} nodes.")

    # ============================================================
    # Sanity check: count missing IDW values
    # ============================================================
    n_missing = 0
    for node in G.nodes():
        n_missing += np.isnan(G.nodes[node]["idw5_train"]).sum()

    print(f"🔎 Total missing IDW entries: {n_missing}")

    print(f"✅ Aggregated to {len(gauge_df)} unique (node, date) records (fast mode).")

    print("🎯 Assigning precipitation labels to nodes...")

    for _, row in tqdm(gauge_df.iterrows(), total=len(gauge_df), desc="Assigning labels"):
        node = row["nearest_node"]
        date = row["DATE"].strftime("%Y-%m-%d")
        prcp = float(row["PRCP"])
        if "target" not in G.nodes[node]:
            G.nodes[node]["target"] = {}
        G.nodes[node]["target"][date] = prcp

    labeled_nodes = sum("target" in G.nodes[n] for n in G.nodes())
    print(f"✅ Assigned targets to {labeled_nodes} nodes.")

    # ============================================================
    # 6️⃣ Remove ERA5-NaN nodes AFTER all assignments
    # ============================================================
    print("🌊 Re-applying ERA5 landmask (removing NaN nodes before export)...")
    first_day = era5.isel(time=0)
    era5_vals = first_day.values
    xs_e, ys_e = first_day.x.values, first_day.y.values

    remove_nodes = []
    for node, d in G.nodes(data=True):
        lat, lon = d["lat"], d["lon"]
        yi, xi = nearest_index(ys_e, lat), nearest_index(xs_e, lon)
        if np.isnan(float(era5_vals[yi, xi])):
            remove_nodes.append(node)

    G.remove_nodes_from(remove_nodes)
    print(f"✅ Removed {len(remove_nodes)} NaN ERA5 nodes → {len(G.nodes())} remain.")

    # ============================================================
    # 7️⃣ Save final graph
    # ============================================================
    payload = {
        "graph": G,
        "time_axis": time_axis,
        "dynamic_feature_names": ["ERA5_precip_mm_day", "IMERG_precip_mm_day", "IDW5_train_gauge_mm_day"],
    }

    GRAPH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_OUT, "wb") as f:
        pickle.dump(payload, f)

    print("💾 Saved graph with static, dynamic, and label data:")
    print(f"   → {GRAPH_OUT}")
    print(f"📊 Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    print("🎉 Ready for GCN training or data loader setup!")