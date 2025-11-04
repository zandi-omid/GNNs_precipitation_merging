#!/usr/bin/env python3
# coding: utf-8
"""
Split large .pt dataset into daily graph snapshots
---------------------------------------------------
Loads a single multi-day PyTorch Geometric dataset and
saves each day as an individual .pt file for lazy loading.

***Adds progress bar for large datasets
***Skips already existing files to allow safe reruns
"""

import torch
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
snapshots_path = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots")
input_file = snapshots_path / "daily_graphs_fixed_masks.pt"
out_dir = snapshots_path / "daily_snapshots"
out_dir.mkdir(exist_ok=True)

# ============================================================
# Load main dataset
# ============================================================
print(f"= Loading dataset  {input_file}")
dataset = torch.load(input_file, map_location="cpu")
print(f" Loaded {len(dataset)} daily Data objects")

# ============================================================
# =¾ Save individual daily snapshots
# ============================================================
for i, day_data in tqdm(
    enumerate(dataset),
    total=len(dataset),
    desc="Saving daily snapshots",
    ncols=80
):
    out_path = out_dir / f"day_{i:04d}.pt"
    if not out_path.exists():  # skip if already saved
        torch.save(day_data, out_path)

print(f"\n Saved {len(dataset)} daily .pt files to  {out_dir}")
print("= Done!")
