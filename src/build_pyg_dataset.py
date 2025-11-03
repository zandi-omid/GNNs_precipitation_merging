#!/usr/bin/env python3
# coding: utf-8
"""
Convert labeled graph with dynamic features into PyTorch Geometric dataset.
----------------------------------------------------------------------------
Each daily snapshot is exported as torch_geometric.data.Data:
    x[node, feat]  = [lat, lon, elev, ERA5, IMERG]
    y[node]        = precipitation (label) or NaN
    edge_index     = from DEM adjacency
    train/val/test masks fixed across days (only nodes with labels)

Author: Omid Zandi
"""
#%% imports
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
import random

# ============================================================
# 1️⃣ Load graph
# ============================================================
graph_path = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels.pkl")
with open(graph_path, "rb") as f:
    data_dict = pickle.load(f)
G = data_dict["graph"]
time_axis = data_dict["time_axis"]
print(f"✅ Loaded graph with {len(G.nodes())} nodes and {len(G.edges())}")

# ============================================================
# 2️⃣ Prepare edges for PyG
# ============================================================
edge_index = np.array(list(G.edges())).T
edge_index = torch.tensor(edge_index, dtype=torch.long)

# ============================================================
# 3️⃣ Extract static & dynamic info
# ============================================================
nodes = list(G.nodes())
n = len(nodes)
sample_node = G.nodes[nodes[0]]
T = len(G.nodes[nodes[0]]["dynamic"])  # number of days
print(f"⏱️ {T} daily timesteps, {n} nodes")

# Static features: lat, lon, elevation
lat = np.array([G.nodes[n]["lat"] for n in nodes])
lon = np.array([G.nodes[n]["lon"] for n in nodes])
elev = np.array([G.nodes[n]["elevation"] for n in nodes])

# ============================================================
# 4️⃣ Identify labeled nodes (with any gauge data)
# ============================================================
labeled_nodes = [n for n in nodes if "target" in G.nodes[n]]
labeled_idx = np.array([nodes.index(n) for n in labeled_nodes])

# Split into train/val/test (fixed ratios)
np.random.seed(42)
np.random.shuffle(labeled_idx)
n_lab = len(labeled_idx)
train_end = int(0.5 * n_lab)
val_end   = int(0.7 * n_lab)

train_idx = labeled_idx[:train_end]
val_idx   = labeled_idx[train_end:val_end]
test_idx  = labeled_idx[val_end:]

# Masks for all nodes
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
for t, day in tqdm(enumerate(time_axis), total=len(time_axis), desc="Building daily snapshots"):
    # Dynamic features (ERA5, IMERG)
    dynamic = np.array([G.nodes[n]["dynamic"][t] for n in nodes])  # shape [nodes, 2]
    x = np.stack([lat, lon, elev, dynamic[:, 0], dynamic[:, 1]], axis=1)
    x = torch.tensor(x, dtype=torch.float32)

    # Labels (targets)
    y = np.full(n, np.nan, dtype=np.float32)
    for i, node in enumerate(nodes):
        if "target" in G.nodes[node] and day in G.nodes[node]["target"]:
            y[i] = G.nodes[node]["target"][day]
    y = torch.tensor(y, dtype=torch.float32)

    # Save snapshot
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    pyg_data_list.append(data)

print(f"✅ Created {len(pyg_data_list)} daily Data objects")

# ============================================================
# 6️⃣ Save dataset
# ============================================================
out_dir = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "daily_graphs_fixed_masks.pt"

torch.save(pyg_data_list, out_path)
print(f"💾 Saved {len(pyg_data_list)} daily graphs → {out_path}")