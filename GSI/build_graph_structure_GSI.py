#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tomli as tomllib
import rioxarray as rxr
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm
import xarray as xr

def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(cfg, section, key, default=None):
    return cfg.get(section, {}).get(key, default)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def pct_tag(x: float, digits: int = 3) -> str:
    return f"{int(round(float(x) * 100)):0{digits}d}"


def tag_range(start, end):
    if not start and not end:
        return "all_years"
    if start and end:
        return f"{start}_to_{end}"
    if start:
        return f"from_{start}"
    return f"to_{end}"


def parse_time_axis(time_axis, start=None, end=None):
    start = start or None
    end = end or None

    dates = np.array(time_axis, dtype="U10")
    idx = np.arange(len(dates))

    if start is not None:
        keep = dates >= np.array(start, dtype="U10")
        dates, idx = dates[keep], idx[keep]

    if end is not None:
        keep = dates <= np.array(end, dtype="U10")
        dates, idx = dates[keep], idx[keep]

    return dates.tolist(), idx.tolist()


def approximate_lonlat_to_km(coords_lonlat):
    mean_lat = np.deg2rad(np.nanmean(coords_lonlat[:, 1]))
    coords_km = np.empty_like(coords_lonlat, dtype=np.float64)
    coords_km[:, 0] = coords_lonlat[:, 0] * 111.0 * np.cos(mean_lat)
    coords_km[:, 1] = coords_lonlat[:, 1] * 111.0
    return coords_km


def load_graph_payload(graph_path):
    with open(graph_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "graph" in payload:
        G = payload["graph"]
        time_axis = payload.get("time_axis", None)
    else:
        G = payload
        time_axis = None

    return G, time_axis, payload


def save_graph_payload(graph_path, payload):
    graph_path = Path(graph_path)
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    with open(graph_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved GSI graph artifact:\n  {graph_path}")


def build_base_graph(cfg):
    dem_path = cfg_get(cfg, "paths", "dem_path")
    if not dem_path:
        raise ValueError("Please provide [paths].dem_path to build the base mesh graph.")

    xmin = float(cfg_get(cfg, "domain", "xmin"))
    ymin = float(cfg_get(cfg, "domain", "ymin"))
    xmax = float(cfg_get(cfg, "domain", "xmax"))
    ymax = float(cfg_get(cfg, "domain", "ymax"))

    print("Loading DEM and building base mesh graph...")

    da = rxr.open_rasterio(dem_path).squeeze()
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

    print(f"DEM mesh graph: nodes={len(G.nodes()):,}, edges={len(G.edges()):,}")
    print("GSI A1/A2 training edges are built from [graph] settings.")

    return G


def load_gauges(gauge_path):
    df = pd.read_csv(
        gauge_path,
        usecols=["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"],
    )

    df = df.dropna(subset=["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")
    df = df.dropna(subset=["PRCP"])
    df = df[df["PRCP"] >= 0].copy()

    return df


def split_stations(stations, train_density, seed):
    stations = np.array(sorted(stations), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)

    n_train = int(np.ceil(len(stations) * train_density))

    train = set(stations[:n_train].tolist())
    holdout = set(stations[n_train:].tolist())

    return train, holdout


def map_gauges_to_nodes(gauge_df, G):
    node_ids = list(G.nodes())
    node_coords = np.array(
        [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in node_ids],
        dtype=np.float64,
    )

    tree = cKDTree(node_coords)

    gauge_coords = gauge_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float64)
    _, idx = tree.query(gauge_coords, k=1)

    gauge_df = gauge_df.copy()
    gauge_df["nearest_node"] = [node_ids[i] for i in idx]

    return gauge_df


def aggregate_duplicate_gauges(gauge_df):
    dup_counts = gauge_df.groupby(["nearest_node", "DATE"]).size()
    print(f"Duplicate node/date pairs: {int((dup_counts > 1).sum()):,}")

    return (
        gauge_df
        .groupby(["nearest_node", "DATE"], as_index=False)
        .agg({
            "PRCP": "mean",
            "LATITUDE": "mean",
            "LONGITUDE": "mean",
            "STATION": lambda x: ",".join(sorted(set(x.astype(str)))),
        })
    )


def build_edge_index_a2_mesh(
    G,
    common_nodes,
    mode="knn",
    k=8,
    radius_km=50.0,
    power=2.0,
    max_neighbors=30,
    eps=1e-6,
):
    coords_lonlat = np.array(
        [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in common_nodes],
        dtype=np.float64,
    )
    coords_km = approximate_lonlat_to_km(coords_lonlat)
    tree = cKDTree(coords_km)

    edges = []
    weights = []

    for recv_idx in tqdm(range(len(common_nodes)), desc=f"Building A2 {mode} mesh graph"):
        recv_coord = coords_km[recv_idx]

        if mode == "knn":
            dists, inds = tree.query(recv_coord, k=min(k + 1, len(common_nodes)))
            cand = []
            for d, j in zip(np.atleast_1d(dists), np.atleast_1d(inds)):
                j = int(j)
                if j == recv_idx:
                    continue
                cand.append(j)
                if len(cand) >= k:
                    break

        elif mode == "radius":
            cand = tree.query_ball_point(recv_coord, r=radius_km)
            cand = [j for j in cand if j != recv_idx]

            if max_neighbors is not None and len(cand) > max_neighbors:
                dists_tmp = np.linalg.norm(coords_km[cand] - recv_coord[None, :], axis=1)
                order = np.argsort(dists_tmp)[:max_neighbors]
                cand = [cand[q] for q in order]
        else:
            raise ValueError(f"Unknown A2 mode: {mode}")

        for sender_idx in cand:
            d_km = np.linalg.norm(coords_km[sender_idx] - recv_coord)
            d_km = max(float(d_km), eps)
            w = 1.0 / (d_km ** power)
            edges.append((recv_idx, sender_idx))
            weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    print(f"A2 mode: {mode}")
    print(f"A2 directed edges: {edge_index.shape[1]:,}")

    return edge_index, edge_weight


def build_edge_index_a1_gauge_to_mesh(
    G,
    common_nodes,
    train_gauge_nodes,
    radius_km=50.0,
    fallback_k=2,
    power=2.0,
    eps=1e-6,
):
    """
    Build A1 edges:

      receiver = every mesh node
      sender   = training gauge nodes only

    If no training gauge is within radius_km, connect receiver to nearest fallback_k
    training gauge nodes.

    Self-edge is excluded to avoid label leakage at training gauge nodes.
    """
    node_to_idx = {n: i for i, n in enumerate(common_nodes)}

    train_gauge_nodes = [n for n in train_gauge_nodes if n in node_to_idx]
    if len(train_gauge_nodes) == 0:
        raise ValueError("No training gauge nodes found in graph.")

    all_coords_lonlat = np.array(
        [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in common_nodes],
        dtype=np.float64,
    )
    gauge_coords_lonlat = np.array(
        [(G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in train_gauge_nodes],
        dtype=np.float64,
    )

    all_coords_km = approximate_lonlat_to_km(all_coords_lonlat)
    gauge_coords_km = approximate_lonlat_to_km(gauge_coords_lonlat)

    tree = cKDTree(gauge_coords_km)

    edges = []
    weights = []
    distances_km = []

    no_radius_count = 0
    isolated_after_self_count = 0

    for recv_idx, recv_node in enumerate(tqdm(common_nodes, desc="Building A1 gauge-to-mesh")):
        recv_coord = all_coords_km[recv_idx]

        cand = tree.query_ball_point(recv_coord, r=radius_km)

        # Exclude self if receiver itself is a training gauge
        cand = [
            j for j in cand
            if train_gauge_nodes[j] != recv_node
        ]

        if len(cand) == 0:
            no_radius_count += 1

            k_query = min(fallback_k + 1, len(train_gauge_nodes))
            dists, inds = tree.query(recv_coord, k=k_query)

            if np.isscalar(inds):
                inds = np.array([inds])
                dists = np.array([dists])

            cand = []
            for j in inds:
                if train_gauge_nodes[int(j)] == recv_node:
                    continue
                cand.append(int(j))
                if len(cand) >= fallback_k:
                    break

        if len(cand) == 0:
            isolated_after_self_count += 1
            continue

        sender_coords = gauge_coords_km[cand]
        dists = np.linalg.norm(sender_coords - recv_coord[None, :], axis=1)

        for local_k, sender_local_idx in enumerate(cand):
            sender_node = train_gauge_nodes[sender_local_idx]
            sender_idx = node_to_idx[sender_node]

            dist_km = max(float(dists[local_k]), eps)
            w = 1.0 / (dist_km ** power)

            # row = receiver, col = sender
            edges.append((recv_idx, sender_idx))
            weights.append(w)
            distances_km.append(dist_km)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    edge_distance_km = torch.tensor(distances_km, dtype=torch.float32)

    print(f"A1 edges: {edge_index.shape[1]:,}")
    print(
        "A1 raw edge distance km: "
        f"min={float(edge_distance_km.min()):.3f}, "
        f"max={float(edge_distance_km.max()):.3f}, "
        f"mean={float(edge_distance_km.mean()):.3f}"
    )
    print(f"Receivers without radius gauge before fallback: {no_radius_count:,}")
    print(f"Receivers still without sender after self-exclusion: {isolated_after_self_count:,}")

    return edge_index, edge_weight, edge_distance_km


def build_time_node_arrays(G, common_nodes, time_axis, train_agg, holdout_agg):
    N = len(common_nodes)
    T = len(time_axis)

    node_to_idx = {n: i for i, n in enumerate(common_nodes)}
    time_to_idx = {d: i for i, d in enumerate(time_axis)}

    X_gauge_value = np.zeros((T, N, 1), dtype=np.float32)
    X_gauge_mask = np.zeros((T, N, 1), dtype=np.float32)

    Y_all = np.full((T, N), np.nan, dtype=np.float32)
    train_available_all = np.zeros((T, N), dtype=bool)
    eval_mask_all = np.zeros((T, N), dtype=bool)

    train_agg = train_agg.copy()
    holdout_agg = holdout_agg.copy()

    train_agg["DATE_STR"] = pd.to_datetime(train_agg["DATE"]).dt.strftime("%Y-%m-%d")
    holdout_agg["DATE_STR"] = pd.to_datetime(holdout_agg["DATE"]).dt.strftime("%Y-%m-%d")

    for _, row in tqdm(train_agg.iterrows(), total=len(train_agg), desc="Stacking train gauges"):
        node = row["nearest_node"]
        day = row["DATE_STR"]

        if node not in node_to_idx or day not in time_to_idx:
            continue

        ti = time_to_idx[day]
        ni = node_to_idx[node]
        val = float(row["PRCP"])

        X_gauge_value[ti, ni, 0] = val
        X_gauge_mask[ti, ni, 0] = 1.0

        Y_all[ti, ni] = val
        train_available_all[ti, ni] = True

    for _, row in tqdm(holdout_agg.iterrows(), total=len(holdout_agg), desc="Stacking holdout gauges"):
        node = row["nearest_node"]
        day = row["DATE_STR"]

        if node not in node_to_idx or day not in time_to_idx:
            continue

        ti = time_to_idx[day]
        ni = node_to_idx[node]
        val = float(row["PRCP"])

        # Holdout gauges are labels only, never input features.
        Y_all[ti, ni] = val
        eval_mask_all[ti, ni] = True

    return X_gauge_value, X_gauge_mask, Y_all, train_available_all, eval_mask_all


def nearest_index(arr, val):
    return int(np.abs(arr - val).argmin())


def load_aux_features_for_nodes(G, common_nodes, time_axis, cfg):
    aux_specs = cfg_get(cfg, "features", "aux_inputs", None)
    if aux_specs is None:
        aux_specs = [
            {
                "name": "ERA5_precip_mm_day",
                "path": cfg_get(cfg, "paths", "era5_path"),
                "variable": "precipitation",
            },
            {
                "name": "IMERG_precip_mm_day",
                "path": cfg_get(cfg, "paths", "imerg_path"),
                "variable": "precipitation",
            },
        ]
    if not aux_specs:
        raise ValueError("[features].aux_inputs must contain at least one input.")

    arrays = []
    names = []
    for spec in aux_specs:
        name = str(spec["name"])
        path = str(spec["path"])
        variable = str(spec.get("variable", "precipitation"))
        print(f"Loading auxiliary input {name}: {path}")
        dataset = xr.open_dataset(path)
        if variable not in dataset:
            raise KeyError(f"Variable {variable!r} is missing from {path}")
        array = dataset[variable]
        if not {"time", "x", "y"}.issubset(array.dims):
            raise ValueError(
                f"Auxiliary input {name!r} must have time/y/x dimensions; "
                f"found {array.dims}."
            )
        arrays.append(array)
        names.append(name)

    reference = arrays[0]
    aligned = [reference]
    for name, array in zip(names[1:], arrays[1:]):
        if not np.array_equal(array.x, reference.x) or not np.array_equal(array.y, reference.y):
            print(f"Interpolating {name} to {names[0]} grid...")
            array = array.interp(x=reference.x, y=reference.y)
        aligned.append(array)
    arrays = aligned

    date_sets = [
        set(pd.to_datetime(array.time.values).strftime("%Y-%m-%d").tolist())
        for array in arrays
    ]
    common_dates = sorted(set(time_axis).intersection(*date_sets))
    if len(common_dates) == 0:
        raise RuntimeError("No common dates among gauges and auxiliary inputs.")

    arrays = [array.sel(time=pd.to_datetime(common_dates).values) for array in arrays]
    xs = reference.x.values
    ys = reference.y.values

    T = len(common_dates)
    N = len(common_nodes)
    X_aux = np.zeros((T, N, len(arrays)), dtype=np.float32)

    print(f"Stacking {len(arrays)} auxiliary input(s) for mesh nodes...")

    for j, node in enumerate(tqdm(common_nodes, desc="Aux features")):
        lon = G.nodes[node]["lon"]
        lat = G.nodes[node]["lat"]

        xi = nearest_index(xs, lon)
        yi = nearest_index(ys, lat)

        for feature_index, array in enumerate(arrays):
            X_aux[:, j, feature_index] = array[:, yi, xi].values.astype(np.float32)

    X_aux = np.nan_to_num(X_aux, nan=0.0, posinf=0.0, neginf=0.0)

    return common_dates, X_aux, names

def build_sequences(
    G,
    time_axis,
    train_agg,
    holdout_agg,
    scenario_name,
    cfg,
    graph_out_path=None,
    gauge_fold_id=None,
):
    t_in = int(cfg_get(cfg, "sequence", "t_in", 30))
    date_start = cfg_get(cfg, "sequence", "date_start", "") or None
    date_end = cfg_get(cfg, "sequence", "date_end", "") or None
    seq_out_root = Path(cfg_get(cfg, "paths", "seq_out_root"))

    include_aux = bool(cfg_get(cfg, "features", "include_aux", False))

    a1_radius_km = float(cfg_get(cfg, "graph", "a1_radius_km", 50.0))
    a1_fallback_k = int(cfg_get(cfg, "graph", "a1_fallback_k", 2))
    a1_power = float(cfg_get(cfg, "graph", "a1_power", 2.0))

    a2_mode = str(cfg_get(cfg, "graph", "a2_mode", "knn"))
    a2_k = int(cfg_get(cfg, "graph", "a2_k", 8))
    a2_radius_km = float(cfg_get(cfg, "graph", "a2_radius_km", 50.0))
    a2_power = float(cfg_get(cfg, "graph", "a2_power", 2.0))
    use_max_neighbors = bool(cfg_get(cfg, "graph", "use_max_neighbors", True))
    a2_max_neighbors = int(cfg_get(cfg, "graph", "max_neighbors", 30)) if use_max_neighbors else None

    training_mode = str(cfg_get(cfg, "training_mode", "mode", "global")).lower()

    adaptive_enabled = bool(cfg_get(cfg, "adaptive_adj", "enabled", False))
    adaptive_mode = str(cfg_get(cfg, "adaptive_adj", "mode", "edge")).lower()
    adaptive_nheads = int(cfg_get(cfg, "adaptive_adj", "nheads", 4))

    common_nodes = sorted(G.nodes())
    N = len(common_nodes)

    # Load aux first and update time_axis if needed
    if include_aux:
        time_axis, X_aux_all, aux_feature_names = load_aux_features_for_nodes(
            G=G,
            common_nodes=common_nodes,
            time_axis=time_axis,
            cfg=cfg,
        )
    else:
        X_aux_all = None

    time_axis_filt, orig_indices = parse_time_axis(time_axis, date_start, date_end)

    if len(time_axis_filt) < t_in:
        raise RuntimeError("Not enough timesteps for requested t_in.")

    print(f"Common nodes: {N:,}")
    print(f"Timesteps after filtering: {len(time_axis_filt):,}")

    X_gauge_value_all, X_gauge_mask_all, Y_all, train_available_all, eval_mask_all = build_time_node_arrays(
        G=G,
        common_nodes=common_nodes,
        time_axis=time_axis,
        train_agg=train_agg,
        holdout_agg=holdout_agg,
    )

    X_gauge_value = X_gauge_value_all[orig_indices, :, :]
    X_gauge_mask = X_gauge_mask_all[orig_indices, :, :]

    if include_aux:
        X_aux = X_aux_all[orig_indices, :, :]
        X_all = np.concatenate([X_gauge_value, X_gauge_mask, X_aux], axis=-1)
        feature_names = [
            "gauge_obs_full_train_only",
            "gauge_available_full_train_only",
            *aux_feature_names,
        ]

    else:
        X_all = np.concatenate([X_gauge_value, X_gauge_mask], axis=-1)
        feature_names = [
            "gauge_obs_full_train_only",
            "gauge_available_full_train_only",
        ]

    Y = Y_all[orig_indices, :]
    train_available = train_available_all[orig_indices, :]
    eval_mask = eval_mask_all[orig_indices, :]

    train_gauge_nodes = sorted(set(train_agg["nearest_node"].tolist()))
    node_to_idx = {n: i for i, n in enumerate(common_nodes)}

    source_nodes = torch.tensor(
        [node_to_idx[n] for n in train_gauge_nodes if n in node_to_idx],
        dtype=torch.long,
    )

    print(f"Static source/training gauge nodes: {len(source_nodes):,}")

    edge_index_a1, edge_weight_a1, edge_distance_a1_km = build_edge_index_a1_gauge_to_mesh(
        G=G,
        common_nodes=common_nodes,
        train_gauge_nodes=train_gauge_nodes,
        radius_km=a1_radius_km,
        fallback_k=a1_fallback_k,
        power=a1_power,
    )

    edge_index_a2, edge_weight_a2 = build_edge_index_a2_mesh(
        G=G,
        common_nodes=common_nodes,
        mode=a2_mode,
        k=a2_k,
        radius_km=a2_radius_km,
        power=a2_power,
        max_neighbors=a2_max_neighbors,
    )

    print(f"A2 mesh edges: {edge_index_a2.shape[1]:,}")

    if graph_out_path:
        graph_payload = {
            "graph": G,
            "time_axis": time_axis_filt,
            "common_nodes": common_nodes,
            "source_nodes": source_nodes,
            "edge_index_a1": edge_index_a1,
            "edge_weight_a1": edge_weight_a1,
            "edge_distance_a1_km": edge_distance_a1_km,
            "edge_index_a2": edge_index_a2,
            "edge_weight_a2": edge_weight_a2,
            "feature_names": feature_names,
            "scenario_name": scenario_name,
            "training_mode_info": {
                "mode": training_mode,
            },
            "adaptive_adj_info": {
                "enabled": adaptive_enabled,
                "mode": adaptive_mode,
                "nheads": adaptive_nheads,
            },
            "adjacency_info": {
                "A1": "static_training_gauge_to_mesh; active context filtering should happen in dataset/model",
                "A1_edge_convention": "edge_index_a1[0]=receiver, edge_index_a1[1]=sender",
                "edge_weight_a1": "inverse-distance weight",
                "edge_distance_a1_km": "raw receiver-sender distance in kilometers aligned with edge_index_a1 columns",
                "A2": "mesh_graph_no_self_loops_saved; self_loops_added_in_model",
                "a1_radius_km": a1_radius_km,
                "a1_fallback_k": a1_fallback_k,
                "a1_power": a1_power,
                "a2_mode": a2_mode,
                "a2_k": a2_k,
                "a2_radius_km": a2_radius_km,
                "a2_power": a2_power,
                "a2_max_neighbors": a2_max_neighbors,
            },
        }
        save_graph_payload(graph_out_path, graph_payload)

    out_dir = (
        seq_out_root
        / f"pyg_sequences_tgsi_T{t_in:03d}_{tag_range(date_start, date_end)}_{scenario_name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing sequences to:\n  {out_dir}")

    sample_count = 0

    for t in tqdm(range(t_in - 1, len(time_axis_filt)), desc="Saving seq_*.pt"):
        t0 = t - (t_in - 1)

        x_window = X_all[t0:t + 1]
        y_t = np.nan_to_num(Y[t].copy(), nan=0.0, posinf=0.0, neginf=0.0)

        data = {
            "x": torch.from_numpy(x_window.astype(np.float32)),
            "y": torch.from_numpy(y_t.astype(np.float32)),
            "train_available_mask": torch.from_numpy(train_available[t].copy()),
            "train_mask": torch.from_numpy(train_available[t].copy()),  # placeholder; dataset will resample this
            "eval_mask": torch.from_numpy(eval_mask[t].copy()),
            "edge_index_a1": edge_index_a1,
            "edge_weight_a1": edge_weight_a1,
            "edge_distance_a1_km": edge_distance_a1_km,
            "edge_index_a2": edge_index_a2,
            "edge_weight_a2": edge_weight_a2,
            "source_nodes": source_nodes,
            "date": time_axis_filt[t],
            "t_index": int(t),
            "t0_index": int(t0),
            "T_IN": int(t_in),
            "feature_names": feature_names,
            "training_mode_info": {
                "mode": training_mode,
            },
            "adaptive_adj_info": {
                "enabled": adaptive_enabled,
                "mode": adaptive_mode,
                "nheads": adaptive_nheads,
            },
            "adjacency_info": {
                "A1": "static_training_gauge_to_mesh; active context filtering should happen in dataset/model",
                "A1_edge_convention": "edge_index_a1[0]=receiver, edge_index_a1[1]=sender",
                "edge_weight_a1": "inverse-distance weight",
                "edge_distance_a1_km": "raw receiver-sender distance in kilometers aligned with edge_index_a1 columns",
                "A2": "mesh_graph_no_self_loops_saved; self_loops_added_in_model",
                "a1_radius_km": a1_radius_km,
                "a1_fallback_k": a1_fallback_k,
                "a1_power": a1_power,
                "a2_mode": a2_mode,
                "a2_k": a2_k,
                "a2_radius_km": a2_radius_km,
                "a2_power": a2_power,
                "a2_max_neighbors": a2_max_neighbors,
            },
        }
        if gauge_fold_id is not None:
            data["gauge_fold_id"] = torch.from_numpy(gauge_fold_id.copy()).long()

        torch.save(data, out_dir / f"seq_{sample_count:05d}.pt")
        sample_count += 1

    print(f"Saved {sample_count:,} sequence samples.")
    return out_dir

def main():
    args = parse_args()
    cfg = load_toml(args.config)

    graph_path = cfg_get(cfg, "paths", "graph_path")
    gauge_path = cfg_get(cfg, "paths", "gauge_path")

    train_density = float(cfg_get(cfg, "split", "train_density", 0.75))
    seed = int(cfg_get(cfg, "split", "seed", 42))
    split_mode = str(cfg_get(cfg, "split", "mode", "random")).lower()

    a1_radius_km = float(cfg_get(cfg, "graph", "a1_radius_km", 50.0))
    a2_mode = str(cfg_get(cfg, "graph", "a2_mode", "knn"))
    a2_k = int(cfg_get(cfg, "graph", "a2_k", 8))

    train_tag = pct_tag(train_density, 3)
    holdout_tag = pct_tag(1.0 - train_density, 2)

    if split_mode == "manual":
        split_tag = str(cfg_get(cfg, "split", "manual_name", "manual"))
    elif split_mode == "all_crossfit":
        n_folds = int(cfg_get(cfg, "split", "n_folds", 10))
        split_tag = str(cfg_get(cfg, "split", "name", f"all_gauges_crossfit{n_folds}"))
    else:
        split_tag = str(
            cfg_get(
                cfg,
                "split",
                "name",
                f"train{train_tag}_holdout{holdout_tag}_seed{seed}",
            )
        )

    scenario_name = (
        f"gauge2meshA1_{int(a1_radius_km)}km_"
        f"A2{a2_mode}{a2_k}_"
        f"{split_tag}"
    )

    print("=" * 100)
    print(f"Scenario: {scenario_name}")
    print("=" * 100)

    G = build_base_graph(cfg)
    gauge_df = load_gauges(gauge_path)
    gauge_df["STATION"] = gauge_df["STATION"].astype(str)

    time_axis = sorted(gauge_df["DATE"].dt.strftime("%Y-%m-%d").unique().tolist())

    if split_mode == "random":
        train_stations, holdout_stations = split_stations(
            gauge_df["STATION"].dropna().unique(),
            train_density=train_density,
            seed=seed,
        )

    elif split_mode == "manual":
        manual_split_csv = Path(cfg_get(cfg, "split", "manual_split_csv"))
        manual_rule = str(cfg_get(cfg, "split", "manual_train_rule", "xy_nonzero")).lower()

        split_df = pd.read_csv(manual_split_csv)
        required_cols = {"STATION", "X", "Y"}
        missing = required_cols - set(split_df.columns)
        if missing:
            raise ValueError(f"Manual split CSV missing columns: {missing}")

        split_df["STATION"] = split_df["STATION"].astype(str)

        if manual_rule == "xy_nonzero":
            train_mask = ~(
                np.isclose(split_df["X"].astype(float), 0.0)
                & np.isclose(split_df["Y"].astype(float), 0.0)
            )
        elif manual_rule == "xy_zero":
            train_mask = (
                np.isclose(split_df["X"].astype(float), 0.0)
                & np.isclose(split_df["Y"].astype(float), 0.0)
            )
        else:
            raise ValueError(f"Unknown manual_train_rule: {manual_rule}")

        train_stations = set(split_df.loc[train_mask, "STATION"].tolist())
        holdout_stations = set(split_df.loc[~train_mask, "STATION"].tolist())

        all_gauge_stations = set(gauge_df["STATION"].dropna().unique())

        missing_from_gauge = (train_stations | holdout_stations) - all_gauge_stations
        if missing_from_gauge:
            print(
                f"WARNING: {len(missing_from_gauge)} stations in manual split CSV "
                f"are not in gauge file."
            )
            print("First missing:", sorted(missing_from_gauge)[:20])

        train_stations = train_stations & all_gauge_stations
        holdout_stations = holdout_stations & all_gauge_stations

    elif split_mode == "all_crossfit":
        all_gauge_stations = set(gauge_df["STATION"].dropna().unique())
        train_stations = all_gauge_stations
        holdout_stations = all_gauge_stations

    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    print(f"Train stations: {len(train_stations):,}")
    print(f"Holdout stations: {len(holdout_stations):,}")

    train_raw = gauge_df[gauge_df["STATION"].isin(train_stations)].copy()
    holdout_raw = gauge_df[gauge_df["STATION"].isin(holdout_stations)].copy()

    print("Mapping train gauges to mesh nodes...")
    train_mapped = map_gauges_to_nodes(train_raw, G)

    if split_mode == "all_crossfit":
        holdout_mapped = train_mapped.copy()
    else:
        print("Mapping holdout gauges to mesh nodes...")
        holdout_mapped = map_gauges_to_nodes(holdout_raw, G)

    print("Aggregating train gauges...")
    train_agg = aggregate_duplicate_gauges(train_mapped)

    if split_mode == "all_crossfit":
        holdout_agg = train_agg.copy()
    else:
        print("Aggregating holdout gauges...")
        holdout_agg = aggregate_duplicate_gauges(holdout_mapped)

    train_agg["DATE"] = pd.to_datetime(train_agg["DATE"])
    holdout_agg["DATE"] = pd.to_datetime(holdout_agg["DATE"])

    gauge_fold_id = None
    if split_mode == "all_crossfit":
        n_folds = int(cfg_get(cfg, "split", "n_folds", 10))
        if n_folds < 2:
            raise ValueError("[split].n_folds must be at least 2 for all_crossfit mode.")
        fold_seed = int(cfg_get(cfg, "split", "fold_seed", seed))
        common_nodes = sorted(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(common_nodes)}
        gauge_nodes = sorted(set(train_agg["nearest_node"]).intersection(node_to_idx))
        shuffled = list(gauge_nodes)
        np.random.default_rng(fold_seed).shuffle(shuffled)
        gauge_fold_id = np.full(len(common_nodes), -1, dtype=np.int64)
        for position, node in enumerate(shuffled):
            gauge_fold_id[node_to_idx[node]] = position % n_folds
        fold_counts = np.bincount(gauge_fold_id[gauge_fold_id >= 0], minlength=n_folds)
        print(f"Cross-fit gauge-node folds: n_folds={n_folds}, seed={fold_seed}")
        print("Fold node counts:", fold_counts.tolist())

    gauge_csv_out_dir = Path(cfg_get(cfg, "paths", "gauge_csv_out_dir", "./mapped_gauges"))
    gauge_csv_out_dir.mkdir(parents=True, exist_ok=True)

    csv_split_tag = split_tag

    train_agg_path = gauge_csv_out_dir / f"tgsi_train_mapped_{csv_split_tag}.csv"
    holdout_agg_path = gauge_csv_out_dir / f"tgsi_holdout_mapped_{csv_split_tag}.csv"

    train_agg.to_csv(train_agg_path, index=False)
    holdout_agg.to_csv(holdout_agg_path, index=False)

    print(f"Saved train mapped gauges:\n  {train_agg_path}")
    print(f"Saved holdout mapped gauges:\n  {holdout_agg_path}")

    seq_dir = build_sequences(
        G=G,
        time_axis=time_axis,
        train_agg=train_agg,
        holdout_agg=holdout_agg,
        scenario_name=scenario_name,
        cfg=cfg,
        graph_out_path=graph_path,
        gauge_fold_id=gauge_fold_id,
    )

    print("\nDONE")
    print(f"Final seq_dir:\n{seq_dir}")

if __name__ == "__main__":
    main()
