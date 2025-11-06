#!/usr/bin/env python3
# coding: utf-8
"""
Split large .pt dataset into daily graph snapshots
---------------------------------------------------
Loads a single multi-day PyTorch Geometric dataset and
saves each day as an individual .pt file for lazy loading.

Features:
• Progress bar for large datasets
• Safe reruns (skips already existing files)
• Optional regeneration of a single day
"""

import torch
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
snapshots_path = Path(
    "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots"
)
input_file = snapshots_path / "daily_graphs_landmask.pt"
out_dir = snapshots_path / "daily_snapshots_landmask"
out_dir.mkdir(exist_ok=True)

# ============================================================
# Load main dataset
# ============================================================
print(f"📦 Loading dataset → {input_file}")
dataset = torch.load(input_file, map_location="cpu")
print(f"✅ Loaded {len(dataset)} daily Data objects")

# ============================================================
# Optionally regenerate a single day
# ============================================================
# Change this if you want to regenerate only one day
single_day = 803  # index (0-based)
only_one = False  # set True to rewrite just one snapshot

if only_one:
    i = single_day
    out_path = out_dir / f"day_{i:04d}.pt"
    day_data = dataset[i]
    print(f"🔁 Rewriting {out_path} ...")
    torch.save(day_data, out_path)
    print("✅ Done regenerating one day.")
    raise SystemExit  # cleaner exit

# ============================================================
# Save individual daily snapshots
# ============================================================
print(f"💾 Saving individual daily snapshots to {out_dir} ...")

for i, day_data in tqdm(
    enumerate(dataset),
    total=len(dataset),
    desc="Saving daily snapshots",
    ncols=90,
):
    out_path = out_dir / f"day_{i:04d}.pt"
    if not out_path.exists():  # skip already saved
        torch.save(day_data, out_path)

print(f"\n✅ Saved {len(dataset)} daily .pt files to {out_dir}")
print("🏁 Done!")