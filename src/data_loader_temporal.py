#!/usr/bin/env python3
# coding: utf-8
"""
Temporal PyTorch-Geometric DataLoader
-------------------------------------
• Loads daily graph snapshots saved as individual .pt files
• Yields (x, y, edge_index, masks, date) for temporal training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TemporalGraphDataset(Dataset):
    """Dataset representing daily graph snapshots (one .pt per day)."""

    def __init__(self, snapshots_dir, device="cpu"):
        self.snapshots_dir = Path(snapshots_dir)
        self.device = device

        # Locate daily .pt files (e.g., day_0000.pt, day_0001.pt, ...)
        self.files = sorted(self.snapshots_dir.glob("day_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No daily .pt snapshots found in {snapshots_dir}")

        # Load one sample to inspect structure
        
        sample = torch.load(self.files[0], map_location=device)
        self.edge_index = sample.edge_index
        self.train_mask = sample.train_mask
        self.val_mask = sample.val_mask
        self.test_mask = sample.test_mask

        print(f"✅ Found {len(self.files)} daily snapshots in {snapshots_dir}")
        print(f"⏱️ Each snapshot: {sample.x.shape[0]} nodes × {sample.x.shape[1]} features")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Return data for one day."""
        data = torch.load(self.files[idx], map_location=self.device, weights_only=False)
        return {
            "x": data.x,
            "y": data.y,
            "edge_index": data.edge_index,
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
            "date": getattr(data, "date", f"day_{idx:04d}")
        }


def TemporalGraphLoader(dataset, batch_size_days=32, shuffle=True):
    """Creates a DataLoader that samples multiple days per batch."""

    def collate_fn(batch):
        xs = [b["x"].unsqueeze(1) for b in batch]  # [nodes, 1, features]
        ys = [b["y"].unsqueeze(1) for b in batch]  # [nodes, 1]
        x = torch.cat(xs, dim=1)                   # [nodes, days, features]
        y = torch.cat(ys, dim=1)                   # [nodes, days]
        dates = [b["date"] for b in batch]

        return {
            "x": x,
            "y": y,
            "edge_index": batch[0]["edge_index"],
            "train_mask": batch[0]["train_mask"],
            "val_mask": batch[0]["val_mask"],
            "test_mask": batch[0]["test_mask"],
            "dates": dates
        }

    return DataLoader(
        dataset,
        batch_size=batch_size_days,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


# ============================================================
# 🔧 Example usage
# ============================================================
if __name__ == "__main__":
    snapshots_dir = (
        "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/"
        "data/pyg_snapshots/daily_snapshots"
    )

    dataset = TemporalGraphDataset(snapshots_dir, device="cpu")
    loader = TemporalGraphLoader(dataset, batch_size_days=8)

    for batch in loader:
        x, y, edge_index = batch["x"], batch["y"], batch["edge_index"]
        print(f"🧩 Batch → x: {x.shape}, y: {y.shape}, dates: {batch['dates'][:3]} ...")
        break