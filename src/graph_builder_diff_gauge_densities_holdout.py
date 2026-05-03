#!/usr/bin/env python3
# coding: utf-8

"""
Build graph + dynamic features + train/held-out gauge splits.

Scenarios:
    075pct, 050pct, 025pct

For each scenario:
    - train gauges = selected percentage
    - held-out gauges = remaining stations
    - IDW feature is computed ONLY from train gauges
    - IDW is leakage-safe: when estimating a node that has a train gauge,
      that same node/date is removed from the donor set
    - graph targets are assigned ONLY from train gauges
    - held-out gauges are saved separately for evaluation

Outputs:
    graph_with_features_labels_train_075pct.pkl
    gauges_train_075pct.csv
    gauges_holdout_075pct.csv
"""

from __future__ import annotations

import os
import pickle
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
DEM_PATH   = "/xdisk/behrangi/omidzandi/GNNs/data/DEM/ASTER_DEM_0p1deg_AZ_buffer.tif"
ERA5_PATH  = "/xdisk/behrangi/omidzandi/GNNs/data/ERA5_daily/ERA5_DAILY_2005_2024_MERGED.nc"
IMERG_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/IMERG_daily/IMERG_DAILY_2005_2024_MERGED.nc"

GAUGE_PATH = "/xdisk/behrangi/omidzandi/GNNs/data/ghcn_data/ghcn_precip_2005_2024_AZ_buffer_50pct.csv"

GRAPH_OUT_DIR_DEFAULT = Path(
    "/xdisk/behrangi/omidzandi/GNNs/data/graphs/train_holdout_gauge_density"
)
GAUGE_CSV_OUT_DIR_DEFAULT = Path(
    "/xdisk/behrangi/omidzandi/GNNs/data/graphs/train_holdout_gauge_density/gauge_csvs"
)

xmin, ymin, xmax, ymax = -115.25, 29.85, -106.45, 38.25

IDW_K = 10
IDW_POWER = 0.0
SEED = 42
N_WORKERS = min(os.cpu_count(), 28)


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/held-out gauge-density graph scenarios."
    )

    parser.add_argument(
        "--gauge-densities",
        type=float,
        nargs="+",
        required=True,
        help="Training gauge fractions, e.g. 0.75 0.5 0.25"
    )

    parser.add_argument(
        "--graph-out-dir",
        type=str,
        default=str(GRAPH_OUT_DIR_DEFAULT),
    )

    parser.add_argument(
        "--gauge-csv-out-dir",
        type=str,
        default=str(GAUGE_CSV_OUT_DIR_DEFAULT),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================
def scenario_tag(density: float) -> str:
    return f"{int(round(density * 100)):03d}pct"


def nearest_index(arr, val):
    return int(np.abs(arr - val).argmin())


def split_train_holdout_stations(all_stations, densities, seed=42):
    """
    Build nested train subsets and complementary holdout sets.

    Example:
        025pct train  050pct train  075pct train
        holdout = all_stations - train
    """
    stations = np.array(sorted(all_stations), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)

    station_set_all = set(stations.tolist())
    out = {}

    for d in sorted(densities, reverse=True):
        n_train = int(np.ceil(len(stations) * d))
        train_stations = set(stations[:n_train].tolist())
        holdout_stations = station_set_all - train_stations

        out[d] = {
            "train": train_stations,
            "holdout": holdout_stations,
        }

    return out


def compute_idw_for_day(args):
    """
    Leakage-safe IDW for one day.

    Important:
    - donor gauges come ONLY from train gauges
    - if query node == donor node, skip that donor
      This prevents train leakage at gauge nodes.
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

    N = len(node_ids)
    out = np.full(N, np.nan, dtype=np.float32)

    if len(donor_nodes) == 0:
        return t_idx, out

    k_query = min(idw_k + 1, len(donor_nodes))
    tree = cKDTree(donor_coords, balanced_tree=False, compact_nodes=False)

    dists, inds = tree.query(node_coords, k=k_query)

    if k_query == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    for qi, qnode in enumerate(node_ids):
        kept_d = []
        kept_v = []

        for dist, ind in zip(dists[qi], inds[qi]):
            donor_node = donor_nodes[ind]

            # ------------------------------------------------
            # CRITICAL leakage prevention:
            # do not use the same node's own gauge as IDW donor
            # ------------------------------------------------
            if donor_node == qnode:
                continue

            kept_d.append(float(dist))
            kept_v.append(float(donor_vals[ind]))

            if len(kept_v) == idw_k:
                break

        if len(kept_v) == 0:
            continue

        kept_d = np.asarray(kept_d, dtype=np.float64)
        kept_v = np.asarray(kept_v, dtype=np.float64)

        if np.any(kept_d == 0.0):
            out[qi] = np.float32(kept_v[kept_d == 0.0][0])
        else:
            if idw_power == 0:
                out[qi] = np.float32(np.mean(kept_v))
            else:
                w = 1.0 / np.power(kept_d, idw_power)
                out[qi] = np.float32(np.sum(w * kept_v) / np.sum(w))

    return t_idx, out


# ============================================================
# GRAPH / DATA LOADING
# ============================================================
def build_base_graph():
    print("Loading DEM and constructing base graph...")

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

    print(f"Base graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G


def load_dynamic_data():
    print("Loading ERA5 and IMERG...")

    era5 = xr.open_dataset(ERA5_PATH)["precipitation"]
    imerg = xr.open_dataset(IMERG_PATH)["precipitation"]

    if not np.array_equal(era5.x, imerg.x) or not np.array_equal(era5.y, imerg.y):
        print("Interpolating ERA5 to IMERG grid...")
        era5 = era5.interp(x=imerg.x, y=imerg.y)

    common_time = np.intersect1d(era5.time.values, imerg.time.values)

    era5 = era5.sel(time=common_time)
    imerg = imerg.sel(time=common_time)

    xs_e = imerg.x.values.astype(np.float32)
    ys_e = imerg.y.values.astype(np.float32)
    time_axis = pd.to_datetime(common_time).strftime("%Y-%m-%d").tolist()

    print(f"Dynamic time axis length: {len(time_axis)}")
    return era5, imerg, xs_e, ys_e, time_axis


def load_gauge_csv():
    print("Reading gauge CSV...")

    gauge_df = pd.read_csv(
        GAUGE_PATH,
        usecols=["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"]
    )

    gauge_df = gauge_df.dropna(subset=["STATION", "LATITUDE", "LONGITUDE", "PRCP"])
    gauge_df["DATE"] = pd.to_datetime(gauge_df["DATE"])

    print(f"Gauge rows: {len(gauge_df):,}")
    print(f"Unique stations: {gauge_df['STATION'].nunique():,}")

    return gauge_df


# ============================================================
# GAUGE MAPPING / FEATURES
# ============================================================
def map_gauges_to_nodes(gauge_df, G):
    node_ids = list(G.nodes())
    node_coords = np.array(
        [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids],
        dtype=np.float32,
    )

    tree = cKDTree(node_coords, balanced_tree=False, compact_nodes=False)

    gauge_coords = gauge_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    _, idx = tree.query(gauge_coords, k=1)

    gauge_df = gauge_df.copy()
    gauge_df["nearest_node"] = [node_ids[i] for i in idx]

    return gauge_df


def aggregate_duplicate_gauges(gauge_df):
    """
    Average duplicate station/node/date records after mapping to graph nodes.
    """
    dup_counts = gauge_df.groupby(["nearest_node", "DATE"]).size()
    n_dups = int((dup_counts > 1).sum())

    print(f"Duplicate (node,date) pairs: {n_dups:,}; averaging if needed.")

    agg_df = (
        gauge_df
        .groupby(["nearest_node", "DATE"], as_index=False)
        .agg({
            "PRCP": "mean",
            "LATITUDE": "mean",
            "LONGITUDE": "mean",
        })
    )

    return agg_df


def compute_idw_timeseries_for_graph(G, train_gauge_df, time_axis):
    print("Computing leakage-safe IDW using TRAIN gauges only...")

    node_ids = list(G.nodes())
    node_coords = np.array(
        [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids],
        dtype=np.float32,
    )

    IDW_all = np.full((len(time_axis), len(node_ids)), np.nan, dtype=np.float32)
    time_to_idx = {d: i for i, d in enumerate(time_axis)}

    train_gauge_df = train_gauge_df.copy()
    train_gauge_df["DATE_STR"] = pd.to_datetime(train_gauge_df["DATE"]).dt.strftime("%Y-%m-%d")

    tasks = []

    for day_str, day_df in train_gauge_df.groupby("DATE_STR"):
        if day_str not in time_to_idx:
            continue

        t_idx = time_to_idx[day_str]

        donor_nodes = day_df["nearest_node"].tolist()
        donor_vals = day_df["PRCP"].to_numpy(dtype=np.float32)
        donor_coords = np.array(
            [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in donor_nodes],
            dtype=np.float32,
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

    print(f"Running daily IDW for {len(tasks)} days with {N_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(compute_idw_for_day, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Daily IDW"):
            t_idx, out = fut.result()
            IDW_all[t_idx, :] = out

    for k, node in enumerate(node_ids):
        G.nodes[node]["idw5_train"] = IDW_all[:, k]

    return G


def assign_dynamic_features(G, era5, imerg, xs_e, ys_e, time_axis):
    print("Assigning dynamic features...")

    for node, d in tqdm(G.nodes(data=True), total=len(G.nodes()), desc="Dynamic features"):
        lat = float(d["lat"])
        lon = float(d["lon"])

        yi = nearest_index(ys_e, lat)
        xi = nearest_index(xs_e, lon)

        era5_ts = era5[:, yi, xi].values.astype(np.float32)
        imerg_ts = imerg[:, yi, xi].values.astype(np.float32)
        idw_ts = G.nodes[node]["idw5_train"].astype(np.float32)

        G.nodes[node]["dynamic_index"] = (yi, xi)
        G.nodes[node]["dynamic"] = np.stack(
            [era5_ts, imerg_ts, idw_ts],
            axis=1,
        )

    return G


def assign_train_targets(G, train_gauge_df):
    print("Assigning TRAIN gauge targets only...")

    for _, row in tqdm(train_gauge_df.iterrows(), total=len(train_gauge_df), desc="Train targets"):
        node = row["nearest_node"]
        date = row["DATE"].strftime("%Y-%m-%d")
        prcp = float(row["PRCP"])

        if "target" not in G.nodes[node]:
            G.nodes[node]["target"] = {}

        G.nodes[node]["target"][date] = prcp

    labeled_nodes = sum("target" in G.nodes[n] for n in G.nodes())
    print(f"Train target nodes: {labeled_nodes:,}")

    return G


def remove_nan_era5_nodes(G, era5):
    print("Removing nodes with NaN ERA5 landmask...")

    first_day = era5.isel(time=0)
    era5_vals = first_day.values
    xs_e = first_day.x.values
    ys_e = first_day.y.values

    remove_nodes = []

    for node, d in G.nodes(data=True):
        lat = float(d["lat"])
        lon = float(d["lon"])

        yi = nearest_index(ys_e, lat)
        xi = nearest_index(xs_e, lon)

        if np.isnan(float(era5_vals[yi, xi])):
            remove_nodes.append(node)

    G.remove_nodes_from(remove_nodes)

    print(f"Removed {len(remove_nodes):,} nodes; remaining nodes: {len(G.nodes()):,}")
    return G


def count_missing_idw(G):
    total = 0
    for node in G.nodes():
        total += np.isnan(G.nodes[node]["idw5_train"]).sum()
    return int(total)


# ============================================================
# SCENARIO BUILDER
# ============================================================
def build_one_scenario(
    density,
    train_stations,
    holdout_stations,
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

    print("\n" + "=" * 90)
    print(f"Building scenario {tag}")
    print(f"Train stations:   {len(train_stations):,}")
    print(f"Holdout stations: {len(holdout_stations):,}")
    print("=" * 90)

    G = build_base_graph()

    train_raw = gauge_df_all[gauge_df_all["STATION"].isin(train_stations)].copy()
    holdout_raw = gauge_df_all[gauge_df_all["STATION"].isin(holdout_stations)].copy()

    print(f"Train gauge rows:   {len(train_raw):,}")
    print(f"Holdout gauge rows: {len(holdout_raw):,}")

    gauge_csv_out_dir.mkdir(parents=True, exist_ok=True)

    train_raw_path = gauge_csv_out_dir / f"gauges_train_raw_{tag}.csv"
    holdout_raw_path = gauge_csv_out_dir / f"gauges_holdout_raw_{tag}.csv"

    train_raw.to_csv(train_raw_path, index=False)
    holdout_raw.to_csv(holdout_raw_path, index=False)

    print(f"Saved train raw gauges:   {train_raw_path}")
    print(f"Saved holdout raw gauges: {holdout_raw_path}")

    # Map both sets to nodes, but only train gauges are assigned to graph.
    train_mapped = map_gauges_to_nodes(train_raw, G)
    holdout_mapped = map_gauges_to_nodes(holdout_raw, G)

    train_agg = aggregate_duplicate_gauges(train_mapped)
    holdout_agg = aggregate_duplicate_gauges(holdout_mapped)

    train_agg["DATE"] = pd.to_datetime(train_agg["DATE"])
    holdout_agg["DATE"] = pd.to_datetime(holdout_agg["DATE"])

    # Save mapped/aggregated holdout gauges for later evaluation.
    train_agg_path = gauge_csv_out_dir / f"gauges_train_mapped_{tag}.csv"
    holdout_agg_path = gauge_csv_out_dir / f"gauges_holdout_mapped_{tag}.csv"

    train_agg.to_csv(train_agg_path, index=False)
    holdout_agg.to_csv(holdout_agg_path, index=False)

    print(f"Saved train mapped gauges:   {train_agg_path}")
    print(f"Saved holdout mapped gauges: {holdout_agg_path}")

    # Features and labels use TRAIN gauges only.
    G = compute_idw_timeseries_for_graph(G, train_agg, time_axis)
    G = assign_dynamic_features(G, era5, imerg, xs_e, ys_e, time_axis)

    n_missing = count_missing_idw(G)
    print(f"Missing IDW entries before ERA5 mask removal: {n_missing:,}")

    G = assign_train_targets(G, train_agg)
    G = remove_nan_era5_nodes(G, era5)

    payload = {
        "graph": G,
        "time_axis": time_axis,
        "dynamic_feature_names": [
            "ERA5_precip_mm_day",
            "IMERG_precip_mm_day",
            "IDW_train_gauge_LOO_mm_day",
        ],
        "scenario_name": tag,
        "train_gauge_density": float(density),
        "seed": int(seed),

        "train_stations": sorted(list(train_stations)),
        "holdout_stations": sorted(list(holdout_stations)),
        "n_train_stations": int(len(train_stations)),
        "n_holdout_stations": int(len(holdout_stations)),
        "n_total_stations": int(gauge_df_all["STATION"].nunique()),

        "train_gauge_csv_raw": str(train_raw_path),
        "holdout_gauge_csv_raw": str(holdout_raw_path),
        "train_gauge_csv_mapped": str(train_agg_path),
        "holdout_gauge_csv_mapped": str(holdout_agg_path),

        "idw_k": int(IDW_K),
        "idw_power": float(IDW_POWER),
        "idw_leakage_note": (
            "IDW uses only train gauges. For every query node/date, "
            "the donor at the same graph node is excluded to prevent target leakage."
        ),
        "target_note": (
            "Graph targets contain only train gauges. Holdout gauges are saved separately "
            "and are not used in IDW or training labels."
        ),
    }

    graph_out_dir.mkdir(parents=True, exist_ok=True)
    graph_out = graph_out_dir / f"graph_with_features_labels_train_{tag}.pkl"

    with open(graph_out, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved graph: {graph_out}")
    print(f"Nodes: {len(G.nodes()):,}, Edges: {len(G.edges()):,}")

    return graph_out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args = parse_args()

    gauge_densities = sorted(set(args.gauge_densities), reverse=True)
    graph_out_dir = Path(args.graph_out_dir)
    gauge_csv_out_dir = Path(args.gauge_csv_out_dir)
    seed = int(args.seed)

    print("Starting train/holdout graph building...")

    era5, imerg, xs_e, ys_e, time_axis = load_dynamic_data()
    gauge_df_all = load_gauge_csv()

    all_stations = sorted(gauge_df_all["STATION"].dropna().unique())

    station_splits = split_train_holdout_stations(
        all_stations=all_stations,
        densities=gauge_densities,
        seed=seed,
    )

    print("\nStation split summary:")
    for d in gauge_densities:
        tag = scenario_tag(d)
        print(
            f"{tag}: train={len(station_splits[d]['train']):,}, "
            f"holdout={len(station_splits[d]['holdout']):,}"
        )

    built = []

    # Build from smallest to largest, optional.
    for density in sorted(gauge_densities):
        out_graph = build_one_scenario(
            density=density,
            train_stations=station_splits[density]["train"],
            holdout_stations=station_splits[density]["holdout"],
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

    print("\nAll scenarios completed.")
    for p in built:
        print(f"  - {p}")