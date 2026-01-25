#!/usr/bin/env python3
# coding: utf-8
"""
DDP-ready GCN training for daily graph snapshots
------------------------------------------------
Each daily .pt file = one sample.
DataLoader + DDP automatically handle batching and GPU distribution.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.data import Dataset, Batch
from pathlib import Path
from torch_geometric.loader import DataLoader
import os
os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_JOB_ID", None)

# ============================================================
# Dataset definition
# ============================================================

class DailyGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)  # ✅ initialize base Dataset
        self.files = sorted(Path(root_dir).glob("day_*.pt"))

    def __len__(self):  # ✅ standard Python method
        return len(self.files)

    def __getitem__(self, idx):  # ✅ standard Python method
        return torch.load(self.files[idx], map_location="cpu")


# ============================================================
# Model definition
# ============================================================
class GCNSpatial(LightningModule):
    def __init__(self, in_feats, hidden_feats=64, out_feats=1, lr=1e-3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv3 = GCNConv(hidden_feats, out_feats)
        self.lr = lr

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        edge_index = batch["edge_index"]
        y = batch["y"]
        train_mask = batch["train_mask"]

        pred = self(x, edge_index)
        pred = pred.squeeze(-1)  # ✅ Fix shape mismatch

        mask = train_mask & ~torch.isnan(y)
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=self.device)
        else:
            loss = F.mse_loss(pred[mask], y[mask])

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        pred = self(batch.x, batch.edge_index)
        mask = batch.val_mask & ~torch.isnan(y)
        loss = F.mse_loss(pred[mask], y[mask])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# Training launcher
# ============================================================
if __name__ == "__main__":
    seed_everything(42)
    snapshots_dir = Path(
        "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots/daily_snapshots_landmask"
    )

    dataset = DailyGraphDataset(snapshots_dir)
    loader = DataLoader(
        dataset,
        batch_size=10,          # 1 day per batch (increase if GPU can fit several)
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    # Infer input dimension
    sample = dataset[0]
    in_feats = sample.x.shape[-1]

    model = GCNSpatial(in_feats=in_feats, hidden_feats=64, out_feats=1, lr=1e-3)

    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,                     # of GPUs
        strategy=DDPStrategy(find_unused_parameters=False),
        precision="16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, loader)