#!/usr/bin/env python3
# coding: utf-8
"""
Build TGCN-ready sequence dataset for retrospective precipitation estimation
---------------------------------------------------------------------------
Each sample corresponds to a target day t and contains:
  x: [T_in, N, F]  where T_in=14 (t-13..t), F=2 (ERA5, IMERG)
  y: [N]           gauge precip at day t (NaN if missing)
  y_mask: [N]      True where y is available
  edge_index: [2, E] and edge_weight: [E] from weighted DEM graph

We do ESTIMATION (retrospective merging), not forecasting:
use (t-13..t) to estimate precip at time t.

Author: Omid Zandi (adapted)
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
T_IN = 14  # window length: t-13..t

GRAPH_FEAT_LABELS = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels.pkl")
GRAPH_WEIGHTED     = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/DEM_graph_weighted.pkl")

OUT_DIR = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T14")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: restrict time range (leave as None to use all)
DATE_START = None  # "2008-01-01"
DATE_END   = None  # "2012-12-31"

# -------------------------
# Helpers
# -------------------------
def build_edge_index_and_weight(G, node_to_idx, weight_key="weight"):
    """
    Convert nx graph edges to undirected COO edge_index + edge_weight.
    """
    edges = []
    weights = []
    for u, v, d in G.edges(data=True):
        if (u not in node_to_idx) or (v not in node_to_idx):
            continue
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        w = float(d.get(weight_key, 1.0))

        # undirected explicit (both directions)
        edges.append((iu, iv)); weights.append(w)
        edges.append((iv, iu)); weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def parse_time_axis(time_axis, start=None, end=None):
    """
    time_axis is list[str] "YYYY-MM-DD".
    Optionally filter to [start,end].
    Returns filtered axis + indices in original list.
    """
    dates = np.array(time_axis, dtype="U10")
    idx = np.arange(len(dates))

    if start is not None:
        keep = dates >= np.array(start, dtype="U10")
        dates, idx = dates[keep], idx[keep]
    if end is not None:
        keep = dates <= np.array(end, dtype="U10")
        dates, idx = dates[keep], idx[keep]
    return dates.tolist(), idx.tolist()


# -------------------------
# 1) Load graphs
# -------------------------
print(f"📦 Loading feature/label graph: {GRAPH_FEAT_LABELS}")
with open(GRAPH_FEAT_LABELS, "rb") as f:
    payload = pickle.load(f)

G_feat = payload["graph"]
time_axis = payload["time_axis"]  # list[str]
print(f"✅ G_feat nodes={len(G_feat.nodes())}, edges={len(G_feat.edges())}, T={len(time_axis)}")

print(f"📦 Loading weighted structure graph: {GRAPH_WEIGHTED}")
with open(GRAPH_WEIGHTED, "rb") as f:
    Gw_payload = pickle.load(f)

# Allow either dict payload or raw nx graph
G_w = Gw_payload["graph"] if isinstance(Gw_payload, dict) and "graph" in Gw_payload else Gw_payload
print(f"✅ G_w nodes={len(G_w.nodes())}, edges={len(G_w.edges())}")

# -------------------------
# 2) Decide node set + ordering (intersection to be safe)
# -------------------------
nodes_feat = set(G_feat.nodes())
nodes_w    = set(G_w.nodes())
common_nodes = sorted(nodes_feat.intersection(nodes_w))

if len(common_nodes) == 0:
    raise RuntimeError("No overlapping nodes between weighted graph and feature/label graph.")

print(f"🔗 Common nodes: {len(common_nodes)}")

node_to_idx = {n: i for i, n in enumerate(common_nodes)}
N = len(common_nodes)

# -------------------------
# 3) Build edge_index / edge_weight from weighted graph
# -------------------------
edge_index, edge_weight = build_edge_index_and_weight(G_w, node_to_idx, weight_key="weight")

print(f"🕸️ edge_index: {edge_index.shape} | edge_weight: {edge_weight.shape}")
print(f"    node idx range: {edge_index.min().item()}..{edge_index.max().item()}")

# -------------------------
# 4) Prepare time axis (optional filtering)
# -------------------------
time_axis_filt, orig_indices = parse_time_axis(time_axis, start=DATE_START, end=DATE_END)
T = len(time_axis_filt)
print(f"📅 Using time steps: {T} (after optional filtering)")

if T < T_IN:
    raise RuntimeError(f"Not enough timesteps ({T}) for T_IN={T_IN}")

# -------------------------
# 5) Stack dynamic features into array [T, N, F]
# -------------------------
print("📡 Building dynamic tensor X_all [T, N, 2] ...")
# G_feat stores: G.nodes[node]["dynamic"] shape [T_total, 2]
# We will slice using orig_indices to match filtered time axis.

X_all = np.empty((T, N, 2), dtype=np.float32)

for j, node in enumerate(tqdm(common_nodes, desc="Nodes")):
    dyn = G_feat.nodes[node]["dynamic"]  # [T_total, 2]
    # slice
    dyn_f = dyn[orig_indices, :]
    X_all[:, j, :] = dyn_f.astype(np.float32)

# -------------------------
# 6) Build labels matrix Y_all [T, N] with NaN where missing
# -------------------------
print("🎯 Building label tensor Y_all [T, N] (NaN if missing) ...")
Y_all = np.full((T, N), np.nan, dtype=np.float32)

# Only some nodes have "target" dict
for j, node in enumerate(tqdm(common_nodes, desc="Labels")):
    tdict = G_feat.nodes[node].get("target", None)
    if not tdict:
        continue
    # fill by matching date strings
    for ti, day in enumerate(time_axis_filt):
        val = tdict.get(day, None)
        if val is not None:
            Y_all[ti, j] = float(val)

# -------------------------
# 7) Export sequence samples: each sample uses (t-13..t) -> y at t
# -------------------------
print(f"💾 Writing sequence samples to: {OUT_DIR}")
sample_count = 0

for t in tqdm(range(T_IN - 1, T), desc="Saving seq_*.pt"):
    # window indices inclusive: t-T_IN+1 .. t
    t0 = t - (T_IN - 1)
    x_seq = X_all[t0 : t + 1, :, :]            # [T_IN, N, 2]
    y_t   = Y_all[t, :]                        # [N]
    y_m   = ~np.isnan(y_t)                     # [N] bool

    data = {
        "x": torch.from_numpy(x_seq),          # float32
        "y": torch.from_numpy(y_t),            # float32, includes NaN
        "y_mask": torch.from_numpy(y_m),       # bool
        "edge_index": edge_index,              # long
        "edge_weight": edge_weight,            # float32
        "date": time_axis_filt[t],
        "t_index": int(t),
        "t0_index": int(t0),
    }

    out_path = OUT_DIR / f"seq_{sample_count:05d}.pt"
    torch.save(data, out_path)
    sample_count += 1

print(f"✅ Done. Saved {sample_count} sequence samples (T_IN={T_IN}).")
print("📌 Each sample: x=[T_in, N, 2], y=[N], y_mask=[N], edge_index, edge_weight")