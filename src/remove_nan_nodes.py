#!/usr/bin/env python3
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.utils import subgraph

# ============================================================
# Paths
# ============================================================
root = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots")
in_dir = root / "daily_snapshots"
out_dir = root / "daily_snapshots_landmask"
out_dir.mkdir(exist_ok=True)
print(f"Input:  {in_dir}")
print(f"Output: {out_dir}")

# ============================================================
# Load reference NaN mask (global pattern)
# ============================================================
ERA5_idx = 3  # adjust if needed
files = sorted(in_dir.glob("day_*.pt"))

# Find the first valid snapshot to get the mask
for f in files:
    try:
        d = torch.load(f, map_location="cpu")
        break
    except Exception as e:
        print(f"⚠️ Skipping corrupted file {f.name} ({e})")

nan_mask = torch.isnan(d.x[:, ERA5_idx])
valid_mask = ~nan_mask
valid_nodes = valid_mask.nonzero(as_tuple=True)[0]
print(f"Found {nan_mask.sum().item()} NaN nodes → will remove these globally.")
print(f"Keeping {valid_nodes.numel()} valid land nodes.")

# ============================================================
# Filter and save all snapshots
# ============================================================
for f in tqdm(files, desc="Filtering snapshots", ncols=80):
    try:
        d = torch.load(f, map_location="cpu")
    except Exception as e:
        print(f"⚠️ Skipping corrupted file {f.name} ({e})")
        continue

    # Subset node features and labels
    d.x = d.x[valid_mask]
    d.y = d.y[valid_mask]
    for m in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(d, m):
            setattr(d, m, getattr(d, m)[valid_mask])

    # Filter edges to valid nodes only
    edge_index, _ = subgraph(valid_nodes, d.edge_index, relabel_nodes=True)
    d.edge_index = edge_index

    # Save cleaned snapshot
    out_path = out_dir / f.name
    torch.save(d, out_path)

print(f"\n✅ Cleaned land-only snapshots saved to {out_dir}")