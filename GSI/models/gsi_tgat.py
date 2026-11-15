# models/gsi_tgat.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import GSI_GATLayer, TemporalAttention
from models.gsi_gat import add_self_loops_receiver_sender
from models.gsi_gatgcn import A1DistanceAttentionMixin


class GSI_GATSpatialEncoder(A1DistanceAttentionMixin, nn.Module):
    def __init__(
        self,
        edge_index_a1,
        edge_index_a2,
        num_nodes: int,
        in_channels: int,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        adaptive_edges: bool = False,
        edge_distance_a1_km=None,
        distance_attention: str = "none",
        distance_epsilon_km: float = 1e-3,
        distance_lambda_init: float = 1.0,
        distance_per_head: bool = False,
        distance_lambda_mode=None,
    ):
        super().__init__()

        edge_index_a2 = add_self_loops_receiver_sender(edge_index_a2, num_nodes)
        self._init_a1_distance_attention(
            edge_distance_a1_km=edge_distance_a1_km,
            num_a1_edges=edge_index_a1.shape[1],
            num_nodes=num_nodes,
            distance_attention=distance_attention,
            distance_epsilon_km=distance_epsilon_km,
            distance_lambda_init=distance_lambda_init,
            distance_per_head=distance_per_head,
            heads=heads,
            distance_lambda_mode=distance_lambda_mode,
        )

        self.gat1 = GSI_GATLayer(
            edge_index=edge_index_a1,
            num_nodes=num_nodes,
            in_dim=in_channels,
            out_dim=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            adaptive_edges=adaptive_edges,
        )

        self.gat2 = GSI_GATLayer(
            edge_index=edge_index_a2,
            num_nodes=num_nodes,
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            adaptive_edges=adaptive_edges,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t):
        h = F.elu(
            self.gat1(
                x_t,
                edge_distance_km=(
                    self.log_edge_distance_a1
                    if self.distance_attention != "none"
                    else None
                ),
                edge_mode=self.distance_attention,
                lambda_distance=self.lambda_distance,
                epsilon_km=self.distance_epsilon_km,
                edge_distance_is_log=True,
            )
        )
        h = self.dropout(h)
        h = F.elu(self.gat2(h))
        return h


class GSI_TGATRegressor(nn.Module):
    """
    Two-stage GSI spatial encoder with GAT + temporal GRU.

    A3T-TGAT:
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
        heads: int = 4,
        dropout: float = 0.0,
        out_channels: int = 1,
        adaptive_edges: bool = False,
        use_temporal_attention: bool = False,
        attn_hidden: Optional[int] = None,
        return_attention: bool = False,
        edge_distance_a1_km=None,
        distance_attention: str = "none",
        distance_epsilon_km: float = 1e-3,
        distance_lambda_init: float = 1.0,
        distance_per_head: bool = False,
        distance_lambda_mode=None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.use_temporal_attention = bool(use_temporal_attention)
        self.return_attention = bool(return_attention)

        self.spatial = GSI_GATSpatialEncoder(
            edge_index_a1=edge_index_a1,
            edge_index_a2=edge_index_a2,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            adaptive_edges=adaptive_edges,
            edge_distance_a1_km=edge_distance_a1_km,
            distance_attention=distance_attention,
            distance_epsilon_km=distance_epsilon_km,
            distance_lambda_init=distance_lambda_init,
            distance_per_head=distance_per_head,
            distance_lambda_mode=distance_lambda_mode,
        )

    @property
    def lambda_distance(self):
        return self.spatial.lambda_distance

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
        B, T, N, F_in = x.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")

        spatial_seq = []
        for t in range(T):
            spatial_seq.append(self.spatial(x[:, t, :, :]))

        h_seq = torch.stack(spatial_seq, dim=1)

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
