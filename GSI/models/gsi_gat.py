# models/gsi_gat.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import GSI_GATLayer
from models.gsi_gatgcn import A1DistanceAttentionMixin


def add_self_loops_receiver_sender(edge_index, num_nodes):
    loop = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    loop = torch.stack([loop, loop], dim=0)
    return torch.cat([edge_index.long(), loop], dim=1)


class GSI_GATRegressor(A1DistanceAttentionMixin, nn.Module):
    def __init__(
        self,
        edge_index_a1,
        edge_weight_a1,
        edge_index_a2,
        edge_weight_a2,
        num_nodes: int,
        in_channels: int = 4,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
        out_channels: int = 1,
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
        self.head = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        x = x[:, -1, :, :]
        h = F.elu(
            self.gat1(
                x,
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
        return self.head(h)
