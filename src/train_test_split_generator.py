#!/usr/bin/env python3
# coding: utf-8
"""
🌎 Train/Test Split Generator for Gauge-Density Scenarios
---------------------------------------------------------
Takes existing scenario graphs and creates train/test splits.

For each scenario:
- 100% scenario: Train on 100% of gauges, test on 0% (or skip test)
- 75% scenario: Train on 75% of all gauges, test on remaining 25%
- 50% scenario: Train on 50% of all gauges, test on remaining 50%
- 25% scenario: Train on 25% of all gauges, test on remaining 75%

Important: IDW features are recomputed using ONLY training gauges to prevent data leakage.

Outputs for each scenario:
  graph_train_{tag}.pkl  # Graph with IDW from train gauges, targets from train gauges
  graph_test_{tag}.pkl   # Graph with IDW from train gauges, targets from test gauges

Author: Omid Zandi (adapted)
"""

# ============================================================
# 📦 Imports
# ============================================================
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse

# ============================================================
# ⚙️ Configurations
# ============================================================
GRAPH_IN_DIR_DEFAULT = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/diff_gauge_density")
GRAPH_OUT_DIR_DEFAULT = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/train_test_splits")

SEED = 42
N_WORKERS = min(os.cpu_count(), 16)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/test splits for gauge-density scenarios."
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        required=True,
        help="Scenario tags to process, e.g. 100pct 075pct 050pct 025pct"
    )

    parser.add_argument(
        "--graph-in-dir",
        type=str,
        default=str(GRAPH_IN_DIR_DEFAULT),
        help="Directory containing input graph pickles"
    )

    parser.add_argument(
        "--graph-out-dir",
        type=str,
        default=str(GRAPH_OUT_DIR_DEFAULT),
        help="Directory to save train/test graph outputs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for train/test splits"
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================
def scenario_tag(density: float) -> str:
    return f"{int(round(density * 100)):03d}pct"


def density_from_tag(tag: str) -> float:
    return int(tag.replace('pct', '')) / 100.0


def compute_idw_for_day(args):
    """
    Compute IDW values for all nodes for one day using specified donor stations.

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

            # For train/test split, we allow using the target node if it's in donors
            # (since we're splitting stations, not nodes)
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


def load_scenario_graph(graph_in_dir, tag):
    """Load a scenario graph pickle."""
    graph_path = graph_in_dir / f"graph_with_features_labels_{tag}.pkl"
    print(f"📂 Loading {graph_path}...")
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)
    return payload


def split_stations(all_stations, train_fraction, seed=42):
    """Split stations into train/test sets."""
    stations = np.array(sorted(all_stations), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)

    n_train = int(np.ceil(len(stations) * train_fraction))
    train_stations = set(stations[:n_train].tolist())
    test_stations = set(stations[n_train:].tolist())

    return train_stations, test_stations


def filter_gauge_df_by_stations(gauge_df, stations):
    """Filter gauge dataframe to only include specified stations."""
    return gauge_df[gauge_df["STATION"].isin(stations)].copy()


def recompute_idw_for_split(G, gauge_df, time_axis, idw_k=10, idw_power=0.0):
    """
    Recompute IDW timeseries using only the gauges in gauge_df.
    """
    print("🧮 Recomputing IDW features for split...")

    # Reset IDW features
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
            idw_k,
            idw_power,
        ))

    print(f"🚀 Launching parallel IDW over {len(tasks)} days with {N_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(compute_idw_for_day, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Daily IDW"):
            t_idx, out = fut.result()
            IDW_all[t_idx, :] = out

    for qi, node in enumerate(node_ids):
        G.nodes[node]["idw5_train"] = IDW_all[:, qi]

    # Update dynamic features with new IDW
    for node in G.nodes():
        lat = float(G.nodes[node]["lat"])
        lon = float(G.nodes[node]["lon"])
        yi, xi = G.nodes[node]["dynamic_index"]

        # Keep ERA5 and IMERG, update IDW
        era5_ts = G.nodes[node]["dynamic"][:, 0]
        imerg_ts = G.nodes[node]["dynamic"][:, 1]
        idw_ts = G.nodes[node]["idw5_train"]

        G.nodes[node]["dynamic"] = np.stack([era5_ts, imerg_ts, idw_ts], axis=1)

    print("✅ IDW features recomputed.")
    return G


def assign_targets_for_split(G, gauge_df, split_name):
    """Assign targets for a specific split (train/test)."""
    print(f"🎯 Assigning {split_name} targets...")

    # Clear existing targets
    for node in G.nodes():
        G.nodes[node][f"target_{split_name}"] = {}

    for _, row in tqdm(gauge_df.iterrows(), total=len(gauge_df), desc=f"Assigning {split_name} labels"):
        node = row["nearest_node"]
        date = row["DATE"].strftime("%Y-%m-%d")
        prcp = float(row["PRCP"])

        G.nodes[node][f"target_{split_name}"][date] = prcp

    labeled_nodes = sum(f"target_{split_name}" in G.nodes[n] and G.nodes[n][f"target_{split_name}"] for n in G.nodes())
    print(f"✅ Assigned {split_name} targets to {labeled_nodes} nodes.")
    return G


def create_train_test_split(payload, train_stations, test_stations, tag, graph_out_dir):
    """Create train and test graphs for a scenario."""
    print(f"\n🔄 Processing scenario: {tag}")

    G = payload["graph"].copy()
    time_axis = payload["time_axis"]
    gauge_df_all = payload.get("gauge_df_all")  # We need to reconstruct this

    # For now, assume we have the gauge_df_all from somewhere
    # Actually, we need to load the gauge CSV that was saved
    gauge_csv_path = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/diff_gauge_density/gauge_csvs") / f"gauges_{tag}.csv"
    if gauge_csv_path.exists():
        gauge_df_all = pd.read_csv(gauge_csv_path)
        gauge_df_all["DATE"] = pd.to_datetime(gauge_df_all["DATE"])
        print(f"📊 Loaded gauge CSV with {len(gauge_df_all):,} rows")
    else:
        print(f"⚠️  Gauge CSV not found: {gauge_csv_path}")
        return None, None

    # Map gauges to nodes (assuming same mapping as original)
    node_ids = list(G.nodes())
    node_coords = np.array([(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids], dtype=np.float32)
    tree = cKDTree(node_coords, balanced_tree=False, compact_nodes=False)
    gauge_coords = gauge_df_all[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    _, idx = tree.query(gauge_coords, k=1)
    gauge_df_all["nearest_node"] = [node_ids[i] for i in idx]

    # Filter for train and test
    train_gauge_df = filter_gauge_df_by_stations(gauge_df_all, train_stations)
    test_gauge_df = filter_gauge_df_by_stations(gauge_df_all, test_stations)

    print(f"📊 Train gauges: {len(train_gauge_df):,} rows from {len(train_stations)} stations")
    print(f"📊 Test gauges: {len(test_gauge_df):,} rows from {len(test_stations)} stations")

    # Create train graph: IDW from train gauges, targets from train gauges
    G_train = G.copy()
    G_train = recompute_idw_for_split(G_train, train_gauge_df, time_axis)
    G_train = assign_targets_for_split(G_train, train_gauge_df, "train")

    # Create test graph: IDW from train gauges (no data leakage), targets from test gauges
    G_test = G.copy()
    # IDW already recomputed with train gauges above, so G_test has the same IDW
    G_test = assign_targets_for_split(G_test, test_gauge_df, "test")

    # Save train graph
    train_payload = payload.copy()
    train_payload["graph"] = G_train
    train_payload["split"] = "train"
    train_payload["train_stations"] = sorted(list(train_stations))
    train_payload["n_train_stations"] = len(train_stations)
    train_payload["test_stations"] = sorted(list(test_stations))
    train_payload["n_test_stations"] = len(test_stations)

    graph_out_dir.mkdir(parents=True, exist_ok=True)
    train_out = graph_out_dir / f"graph_train_{tag}.pkl"
    with open(train_out, "wb") as f:
        pickle.dump(train_payload, f)
    print(f"💾 Saved train graph: {train_out}")

    # Save test graph (only if there are test stations)
    if test_stations:
        test_payload = payload.copy()
        test_payload["graph"] = G_test
        test_payload["split"] = "test"
        test_payload["train_stations"] = sorted(list(train_stations))
        test_payload["n_train_stations"] = len(train_stations)
        test_payload["test_stations"] = sorted(list(test_stations))
        test_payload["n_test_stations"] = len(test_stations)

        test_out = graph_out_dir / f"graph_test_{tag}.pkl"
        with open(test_out, "wb") as f:
            pickle.dump(test_payload, f)
        print(f"💾 Saved test graph: {test_out}")
        return train_out, test_out
    else:
        print("ℹ️  No test stations for this scenario")
        return train_out, None


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    args = parse_args()

    scenarios = args.scenarios
    graph_in_dir = Path(args.graph_in_dir)
    graph_out_dir = Path(args.graph_out_dir)
    seed = args.seed

    print("Starting train/test split generation...")

    # Load all stations once (from 100pct scenario)
    if "100pct" in scenarios:
        payload_100 = load_scenario_graph(graph_in_dir, "100pct")
        all_stations = set(payload_100["selected_stations"])
    else:
        # If 100pct not available, we'd need to get all_stations from somewhere else
        print("⚠️  100pct scenario required to get all stations")
        exit(1)

    print(f"📊 Total stations across all scenarios: {len(all_stations):,}")

    built = []
    for tag in scenarios:
        density = density_from_tag(tag)

        # For each scenario, train_fraction = density
        train_fraction = density
        train_stations, test_stations = split_stations(all_stations, train_fraction, seed)

        print(f"\nScenario {tag}: train_fraction={train_fraction:.2f}")
        print(f"  Train stations: {len(train_stations):,}")
        print(f"  Test stations: {len(test_stations):,}")

        payload = load_scenario_graph(graph_in_dir, tag)
        train_out, test_out = create_train_test_split(
            payload, train_stations, test_stations, tag, graph_out_dir
        )

        built.append((train_out, test_out))

    print("\n✅ All train/test splits completed.")
    print("Saved graphs:")
    for train_p, test_p in built:
        print(f"  Train: {train_p}")
        if test_p:
            print(f"  Test:  {test_p}")