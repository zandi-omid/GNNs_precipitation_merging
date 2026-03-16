#!/usr/bin/env python3
# coding: utf-8
"""
🌎 Unified Graph Builder for Multiple Gauge-Density Scenarios
-------------------------------------------------------------
Build graph + dynamic features + gauge targets for several gauge-density scenarios.

Scenarios:
- 100%
- 75%
- 50%
- 25%

Important design:
- We subset by STATION ID, not by daily record.
- The same selected stations are used across ALL years.
- Therefore the same reduced gauge network is used in:
    train / val / test

Outputs:
  graph_with_features_labels_100pct.pkl
  graph_with_features_labels_75pct.pkl
  graph_with_features_labels_50pct.pkl
  graph_with_features_labels_25pct.pkl

Each payload contains:
{
    "graph": networkx.Graph,
    "time_axis": list[str],
    "dynamic_feature_names": [...],
    "gauge_density": float,
    "scenario_name": str,
    "seed": int,
    "selected_stations": list[str],
    "n_selected_stations": int,
    "n_total_stations": int
}

Author: Omid Zandi (adapted)
"""

# ============================================================
# 📦 Imports
# ============================================================
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse

# ============================================================
# ⚙️ Configurations
# ============================================================
DEM_PATH   = "/xdisk/behrangi/omidzandi/GNNs/data/DEM/ASTER_DEM_0p1deg_AZ_buffer.tif"
ERA5_PATH  = "/xdisk/behrangi/omidzandi/GNNs/data/ERA5_daily/ERA5_DAILY_2005_2024_MERGED.nc"
IMERG_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/IMERG_daily/IMERG_DAILY_2005_2024_MERGED.nc"

# This is okay: it is your quality-controlled CSV with stations having enough coverage
GAUGE_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/ghcn_data/ghcn_precip_2005_2024_AZ_buffer_50pct.csv"

GRAPH_OUT_DIR_DEFAULT = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/diff_gauge_density")
GAUGE_CSV_OUT_DIR_DEFAULT = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/diff_gauge_density/gauge_csvs")

xmin, ymin, xmax, ymax = -115.25, 29.85, -106.45, 38.25

IDW_K = 10
IDW_POWER = 0.0

SEED = 42

N_WORKERS = min(os.cpu_count(), 16)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build graph(s) for one or more gauge-density scenarios."
    )

    parser.add_argument(
        "--gauge-densities",
        type=float,
        nargs="+",
        required=True,
        help="One or more gauge densities, e.g. 1.0 0.75 0.5 0.25"
    )

    parser.add_argument(
        "--graph-out-dir",
        type=str,
        default=str(GRAPH_OUT_DIR_DEFAULT),
        help="Directory to save graph pickle outputs"
    )

    parser.add_argument(
        "--gauge-csv-out-dir",
        type=str,
        default=str(GAUGE_CSV_OUT_DIR_DEFAULT),
        help="Directory to save per-scenario gauge CSVs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for nested station subsets"
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================
def scenario_tag(density: float) -> str:
    return f"{int(round(density * 100)):03d}pct"


def nearest_index(arr, val):
    return int(np.abs(arr - val).argmin())


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
                if idw_power == 0:
                    out[qi] = np.float32(np.mean(kept_v))
                else:
                    w = 1.0 / np.power(kept_d, idw_power)
                    out[qi] = np.float32(np.sum(w * kept_v) / np.sum(w))

    return t_idx, out


def build_base_graph():
    print("🗺️ Loading DEM and constructing base grid graph...")
    da = rxr.open_rasterio(DEM_PATH).squeeze()
    da = da.where(~np.isnan(da), drop=True)
    da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)

    values = da.values
    nrows, ncols = values.shape
    xs, ys = da.x.values, da.y.values

    G = nx.grid_2d_graph(nrows, ncols, periodic=False)

    # add diagonals
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            G.add_edge((i, j), (i + 1, j + 1))
            G.add_edge((i + 1, j), (i, j + 1))

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
    return G


def load_dynamic_data():
    print("💧 Loading ERA5 & IMERG daily time series...")
    era5 = xr.open_dataset(ERA5_PATH)["precipitation"]
    imerg = xr.open_dataset(IMERG_PATH)["precipitation"]

    if not np.array_equal(era5.x, imerg.x) or not np.array_equal(era5.y, imerg.y):
        print("🔁 Interpolating ERA5 to IMERG grid...")
        era5 = era5.interp(x=imerg.x, y=imerg.y)

    common_time = np.intersect1d(era5.time.values, imerg.time.values)
    era5 = era5.sel(time=common_time)
    imerg = imerg.sel(time=common_time)

    xs_e = imerg.x.values.astype(np.float32)
    ys_e = imerg.y.values.astype(np.float32)
    time_axis = pd.to_datetime(common_time).strftime("%Y-%m-%d").tolist()

    print(f"✅ Dynamic datasets ready with T={len(time_axis)}")
    return era5, imerg, xs_e, ys_e, time_axis


def load_gauge_csv():
    print("📊 Reading gauge CSV...")
    gauge_df = pd.read_csv(
        GAUGE_PATH,
        usecols=["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"]
    )
    gauge_df = gauge_df.dropna(subset=["STATION", "LATITUDE", "LONGITUDE"])
    gauge_df["DATE"] = pd.to_datetime(gauge_df["DATE"])
    print(f"✅ Gauge rows loaded: {len(gauge_df):,}")
    print(f"✅ Unique stations: {gauge_df['STATION'].nunique():,}")
    return gauge_df


def build_nested_station_subsets(all_stations, densities, seed=42):
    """
    Build nested station subsets using one shared random shuffle.
    Example:
        25% subset ⊂ 50% subset ⊂ 75% subset ⊂ 100% subset
    """
    stations = np.array(sorted(all_stations), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)

    out = {}
    for d in sorted(densities, reverse=True):
        n_keep = int(np.ceil(len(stations) * d))
        out[d] = set(stations[:n_keep].tolist())

    return out


def prepare_gauge_df_for_scenario(gauge_df_all, selected_stations):
    """
    Filter by selected stations, map to nearest nodes later.
    """
    gdf = gauge_df_all[gauge_df_all["STATION"].isin(selected_stations)].copy()
    return gdf


def map_gauges_to_nodes(gauge_df, G):
    print("📍 Mapping gauges to graph nodes...")
    node_ids = list(G.nodes())
    node_coords = np.array([(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids], dtype=np.float32)

    tree = cKDTree(node_coords, balanced_tree=False, compact_nodes=False)
    gauge_coords = gauge_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    _, idx = tree.query(gauge_coords, k=1)
    gauge_df["nearest_node"] = [node_ids[i] for i in idx]

    print(f"✅ Gauge mapping complete ({len(gauge_df):,} rows).")
    return gauge_df


def aggregate_duplicate_gauges(gauge_df):
    dup_counts = gauge_df.groupby(["nearest_node", "DATE"]).size()
    n_dups = int((dup_counts > 1).sum())
    print(f"🔍 Found {n_dups:,} duplicate (node, date) pairs → averaging them...")

    agg_df = (
        gauge_df.groupby(["nearest_node", "DATE"], as_index=False)[["PRCP", "LATITUDE", "LONGITUDE"]]
        .mean()
    )
    return agg_df


def compute_idw_timeseries_for_graph(G, gauge_df, time_axis):
    print("🧮 Computing leakage-safe daily IDW feature for all nodes...")

    # initialize
    for node in G.nodes():
        G.nodes[node]["idw5_train"] = np.full(len(time_axis), np.nan, dtype=np.float32)

    node_ids = list(G.nodes())
    node_coords = np.array([(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids], dtype=np.float32)

    IDW_all = np.full((len(time_axis), len(node_ids)), np.nan, dtype=np.float32)
    time_to_idx = {d: i for i, d in enumerate(time_axis)}

    gauge_df = gauge_df.copy()
    gauge_df["DATE_STR"] = pd.to_datetime(gauge_df["DATE"]).dt.strftime("%Y-%m-%d")
    daily_groups = gauge_df.groupby("DATE_STR")

    tasks = []
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

    print(f"🚀 Launching parallel IDW over {len(tasks)} days with {N_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(compute_idw_for_day, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Daily IDW"):
            t_idx, out = fut.result()
            IDW_all[t_idx, :] = out

    for qi, node in enumerate(node_ids):
        G.nodes[node]["idw5_train"] = IDW_all[:, qi]

    print("✅ IDW feature computed.")
    return G


def assign_dynamic_features(G, era5, imerg, xs_e, ys_e, time_axis):
    print("📡 Assigning final dynamic daily features to graph nodes...")

    for node, d in tqdm(G.nodes(data=True), total=len(G.nodes()), desc="Assigning dynamic features"):
        lat = float(d["lat"])
        lon = float(d["lon"])
        yi = nearest_index(ys_e, lat)
        xi = nearest_index(xs_e, lon)

        era5_ts = era5[:, yi, xi].values.astype(np.float32)
        imerg_ts = imerg[:, yi, xi].values.astype(np.float32)
        idw_ts = G.nodes[node]["idw5_train"].astype(np.float32)

        G.nodes[node]["dynamic_index"] = (yi, xi)
        G.nodes[node]["dynamic"] = np.stack([era5_ts, imerg_ts, idw_ts], axis=1)

    print(f"✅ Added dynamic features ({len(time_axis)} timesteps) to {len(G.nodes())} nodes.")
    return G


def count_missing_idw(G):
    n_missing = 0
    for node in G.nodes():
        n_missing += np.isnan(G.nodes[node]["idw5_train"]).sum()
    return int(n_missing)


def assign_targets(G, gauge_df):
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
    return G


def remove_nan_era5_nodes(G, era5):
    print("🌊 Re-applying ERA5 landmask (removing NaN nodes before export)...")

    first_day = era5.isel(time=0)
    era5_vals = first_day.values
    xs_e = first_day.x.values
    ys_e = first_day.y.values

    remove_nodes = []
    for node, d in G.nodes(data=True):
        lat = d["lat"]
        lon = d["lon"]
        yi = nearest_index(ys_e, lat)
        xi = nearest_index(xs_e, lon)

        if np.isnan(float(era5_vals[yi, xi])):
            remove_nodes.append(node)

    G.remove_nodes_from(remove_nodes)
    print(f"✅ Removed {len(remove_nodes)} NaN ERA5 nodes → {len(G.nodes())} remain.")
    return G


def build_one_scenario(
    density,
    selected_stations,
    gauge_df_all,
    era5,
    imerg,
    xs_e,
    ys_e,
    time_axis,
    graph_out_dir,
    gauge_csv_out_dir,
    seed,
):
    tag = scenario_tag(density)
    print("\n" + "=" * 80)
    print(f"🚧 Building scenario: {tag} ({density:.2%} of stations)")
    print("=" * 80)

    # build fresh base graph
    G = build_base_graph()

    # filter stations globally across all years
    gauge_df = prepare_gauge_df_for_scenario(gauge_df_all, selected_stations)
    print(f"✅ Scenario {tag}: selected stations = {len(selected_stations):,}")
    print(f"✅ Scenario {tag}: gauge rows after station filtering = {len(gauge_df):,}")

    gauge_csv_out_dir.mkdir(parents=True, exist_ok=True)
    gauge_csv_path = gauge_csv_out_dir / f"gauges_{tag}.csv"
    gauge_df.to_csv(gauge_csv_path, index=False)
    print(f"💾 Saved scenario gauge CSV: {gauge_csv_path}")

    gauge_df = map_gauges_to_nodes(gauge_df, G)
    gauge_df = aggregate_duplicate_gauges(gauge_df)

    G = compute_idw_timeseries_for_graph(G, gauge_df, time_axis)
    G = assign_dynamic_features(G, era5, imerg, xs_e, ys_e, time_axis)

    n_missing = count_missing_idw(G)
    print(f"🔎 Total missing IDW entries: {n_missing:,}")
    print(f"✅ Aggregated to {len(gauge_df):,} unique (node, date) records.")

    G = assign_targets(G, gauge_df)
    G = remove_nan_era5_nodes(G, era5)

    payload = {
        "graph": G,
        "time_axis": time_axis,
        "dynamic_feature_names": [
            "ERA5_precip_mm_day",
            "IMERG_precip_mm_day",
            "IDW5_train_gauge_mm_day",
        ],
        "gauge_density": float(density),
        "scenario_name": tag,
        "seed": int(seed),
        "selected_stations": sorted(list(selected_stations)),
        "n_selected_stations": int(len(selected_stations)),
        "n_total_stations": int(gauge_df_all["STATION"].nunique()),
    }

    graph_out_dir.mkdir(parents=True, exist_ok=True)
    graph_out = graph_out_dir / f"graph_with_features_labels_{tag}.pkl"

    with open(graph_out, "wb") as f:
        pickle.dump(payload, f)

    print("💾 Saved graph with static, dynamic, and label data:")
    print(f"   → {graph_out}")
    print(f"📊 Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    print(f"📊 Selected stations: {len(selected_stations):,}")
    print("🎉 Ready for TGCN sequence building!")

    return graph_out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    args = parse_args()

    gauge_densities = sorted(set(args.gauge_densities), reverse=True)
    graph_out_dir = Path(args.graph_out_dir)
    gauge_csv_out_dir = Path(args.gauge_csv_out_dir)
    seed = args.seed

    print("Starting multi-scenario graph building...")

    era5, imerg, xs_e, ys_e, time_axis = load_dynamic_data()
    gauge_df_all = load_gauge_csv()

    all_stations = sorted(gauge_df_all["STATION"].dropna().unique())
    station_subsets = build_nested_station_subsets(
        all_stations=all_stations,
        densities=gauge_densities,
        seed=seed,
    )

    print("\nStation subset summary:")
    for d in sorted(gauge_densities, reverse=True):
        print(f"  {scenario_tag(d)} -> {len(station_subsets[d]):,} stations")

    built = []
    for density in sorted(gauge_densities):
        out_graph = build_one_scenario(
            density=density,
            selected_stations=station_subsets[density],
            gauge_df_all=gauge_df_all,
            era5=era5,
            imerg=imerg,
            xs_e=xs_e,
            ys_e=ys_e,
            time_axis=time_axis,
            graph_out_dir=graph_out_dir,
            gauge_csv_out_dir=gauge_csv_out_dir,
            seed=seed,
        )
        built.append(out_graph)

    print("\n✅ All scenarios completed.")
    print("Saved graphs:")
    for p in built:
        print(f"  - {p}")