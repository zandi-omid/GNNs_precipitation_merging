#!/usr/bin/env python3
# coding: utf-8
"""
Convert labeled graph with dynamic features into PyTorch Geometric dataset.
---------------------------------------------------------------------------
Uses the *final* graph_with_features_labels.pkl, where:
  • Nodes are DEM pixels with static + dynamic + target data
  • Non-land (NaN ERA5) nodes have already been removed

Each daily snapshot is exported as torch_geometric.data.Data:
    x[node, feat]  = [lat, lon, elev, ERA5, IMERG]
    y[node]        = precipitation (label) or NaN
    edge_index     = undirected edges in integer node index space
    train/val/test masks fixed across days (only nodes with labels)
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path

# ============================================================
# 1️⃣ Load final graph (already land-masked)
# ============================================================
graph_path = Path(
    "/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels.pkl"
)
with open(graph_path, "rb") as f:
    payload = pickle.load(f)

G = payload["graph"]
time_axis = payload["time_axis"]  # list of date strings
print(f"✅ Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
print(f"⏱️ {len(time_axis)} daily timesteps")

# ============================================================
# 2️⃣ Create node index mapping + static features
# ============================================================
# Nodes are (i, j) tuples → map them to integer IDs 0..N-1
nodes = list(G.nodes())                     # list of (i, j)
node_to_idx = {n: i for i, n in enumerate(nodes)}
n = len(nodes)

# Static features
lat = np.array([G.nodes[n]["lat"] for n in nodes], dtype=np.float32)
lon = np.array([G.nodes[n]["lon"] for n in nodes], dtype=np.float32)
elev = np.array([G.nodes[n]["elevation"] for n in nodes], dtype=np.float32)

print(f"📍 Static feature arrays built for {n} nodes")

# ============================================================
# 3️⃣ Build edge_index in integer node ID space
# ============================================================
edges = []
for u, v in G.edges():
    iu = node_to_idx[u]
    iv = node_to_idx[v]
    edges.append((iu, iv))
    edges.append((iv, iu))  # make it explicitly undirected

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"🕸️ edge_index shape: {edge_index.shape}")
print(f"    min idx = {edge_index.min().item()}, max idx = {edge_index.max().item()}")

# Sanity check: all edges are valid
assert edge_index.min().item() >= 0
assert edge_index.max().item() < n, "edge_index has out-of-range node IDs!"

# ============================================================
# 4️⃣ Identify labeled nodes + build masks
# ============================================================
labeled_nodes = [n for n in nodes if "target" in G.nodes[n]]
labeled_idx = np.array([node_to_idx[n] for n in labeled_nodes], dtype=np.int64)

np.random.seed(42)
np.random.shuffle(labeled_idx)
n_lab = len(labeled_idx)
train_end = int(0.5 * n_lab)
val_end   = int(0.7 * n_lab)

train_idx = labeled_idx[:train_end]
val_idx   = labeled_idx[train_end:val_end]
test_idx  = labeled_idx[val_end:]

train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

print(f"🎯 Masks → Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

# ============================================================
# 5️⃣ Build PyG Data objects (one per day)
# ============================================================
pyg_data_list = []
print("📦 Building daily Data objects...")

for t, day in tqdm(enumerate(time_axis), total=len(time_axis), desc="Building snapshots"):
    # Dynamic features (ERA5, IMERG) for this day for all nodes
    # Each node has "dynamic" shape [T, 2] → we take [:, 0] and [:, 1]
    dynamic = np.stack(
        [G.nodes[n]["dynamic"][t] for n in nodes],  # shape [n, 2]
        axis=0,
    ).astype(np.float32)

    x_np = np.stack(
        [lat, lon, elev, dynamic[:, 0], dynamic[:, 1]], axis=1
    )  # [n, 5]
    x = torch.from_numpy(x_np)

    # Labels (targets); default NaN
    y_np = np.full(n, np.nan, dtype=np.float32)
    for idx, node in enumerate(nodes):
        target_dict = G.nodes[node].get("target", None)
        if target_dict is not None:
            if day in target_dict:
                y_np[idx] = target_dict[day]
    y = torch.from_numpy(y_np)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    # optional: store date
    data.date = day

    pyg_data_list.append(data)

print(f"✅ Created {len(pyg_data_list)} daily Data objects")

# ============================================================
# 6️⃣ Save dataset
# ============================================================
out_dir = Path(
    "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots"
)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "daily_graphs_landmask.pt"

torch.save(pyg_data_list, out_path)
print(f"💾 Saved {len(pyg_data_list)} land-only daily graphs → {out_path}")