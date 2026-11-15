# models/gsi_tgcn.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import (
    build_row_normalized_adjacency_coo,
    GSI_GCNLayer,
    TemporalAttention,
)


class GSI_GCNSpatialEncoder(nn.Module):
    def __init__(
        self,
        edge_index_a1,
        edge_weight_a1,
        edge_index_a2,
        edge_weight_a2,
        num_nodes: int,
        in_channels: int,
        hidden_dim: int,
        dropout: float = 0.0,
        adaptive_edges: bool = False,
    ):
        super().__init__()
        A1_idx, A1_val = build_row_normalized_adjacency_coo(
            edge_index_a1, edge_weight_a1, num_nodes, add_self_loops=False
        )
        A2_idx, A2_val = build_row_normalized_adjacency_coo(
            edge_index_a2, edge_weight_a2, num_nodes, add_self_loops=True
        )

        self.gc1 = GSI_GCNLayer(A1_idx, A1_val, num_nodes, in_channels, hidden_dim, adaptive_edges)
        self.gc2 = GSI_GCNLayer(A2_idx, A2_val, num_nodes, hidden_dim, hidden_dim, adaptive_edges)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t):
        h = F.elu(self.gc1(x_t))
        h = self.dropout(h)
        h = F.elu(self.gc2(h))
        return h


class GSI_TGCNRegressor(nn.Module):
    """
    Two-stage GSI spatial encoder + temporal GRU.

    For each timestep:
      X_t -> GCN(A1) -> GCN(A2) -> spatial embedding

    Then:
      embeddings over time -> GRU -> prediction

    A3T-TGCN:
      set use_temporal_attention=True
    """

    def __init__(
        self,
        edge_index_a1,
        edge_weight_a1,
        edge_index_a2,
        edge_weight_a2,
        num_nodes: int,
        in_channels: int = 4,
        hidden_dim: int = 64,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        out_channels: int = 1,
        adaptive_edges: bool = False,
        use_temporal_attention: bool = False,
        attn_hidden: Optional[int] = None,
        return_attention: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.use_temporal_attention = bool(use_temporal_attention)
        self.return_attention = bool(return_attention)

        self.spatial = GSI_GCNSpatialEncoder(
            edge_index_a1=edge_index_a1,
            edge_weight_a1=edge_weight_a1,
            edge_index_a2=edge_index_a2,
            edge_weight_a2=edge_weight_a2,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            adaptive_edges=adaptive_edges,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=int(rnn_layers),
            batch_first=True,
            dropout=dropout if int(rnn_layers) > 1 else 0.0,
        )

        if self.use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_dim, attn_hidden)
        else:
            self.temporal_attention = None

        self.head = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        # x: [B,T,N,F]
        B, T, N, F_in = x.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")

        spatial_seq = []
        for t in range(T):
            spatial_seq.append(self.spatial(x[:, t, :, :]))

        h_seq = torch.stack(spatial_seq, dim=1)  # [B,T,N,H]

        # GRU per node: [B*N,T,H]
        z = h_seq.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        z_out, _ = self.gru(z)
        z_out = z_out.reshape(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)

        if self.use_temporal_attention:
            h_context, alpha = self.temporal_attention(z_out)
            y_hat = self.head(h_context)
            if self.return_attention:
                return y_hat, alpha
            return y_hat

        h_last = z_out[:, -1, :, :]
        return self.head(h_last)