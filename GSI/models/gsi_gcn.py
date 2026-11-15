# models/gsi_gcn.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import build_row_normalized_adjacency_coo, GSI_GCNLayer


class GSI_GCNRegressor(nn.Module):
    def __init__(
        self,
        edge_index_a1,
        edge_weight_a1,
        edge_index_a2,
        edge_weight_a2,
        num_nodes: int,
        in_channels: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        out_channels: int = 1,
        adaptive_edges: bool = False,
        graph_source: str = "a1a2",
    ):
        super().__init__()
        self.graph_source = str(graph_source).lower()
        if self.graph_source not in {"a1a2", "a2"}:
            raise ValueError(
                f"Unsupported GCN graph_source={graph_source!r}; use 'a1a2' or 'a2'."
            )
        A2_idx, A2_val = build_row_normalized_adjacency_coo(
            edge_index_a2, edge_weight_a2, num_nodes, add_self_loops=True
        )

        if self.graph_source == "a2":
            first_idx, first_val = A2_idx, A2_val
        else:
            first_idx, first_val = build_row_normalized_adjacency_coo(
                edge_index_a1, edge_weight_a1, num_nodes, add_self_loops=False
            )

        self.gc1 = GSI_GCNLayer(first_idx, first_val, num_nodes, in_channels, hidden_dim, adaptive_edges)
        self.gc2 = GSI_GCNLayer(A2_idx, A2_val, num_nodes, hidden_dim, hidden_dim, adaptive_edges)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        # x: [B,T,N,F], spatial-only uses last timestep
        x = x[:, -1, :, :]
        h = F.elu(self.gc1(x))
        h = self.dropout(h)
        h = F.elu(self.gc2(h))
        return self.head(h)
