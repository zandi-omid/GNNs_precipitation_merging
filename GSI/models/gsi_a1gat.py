# models/gsi_a1gat.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import GSI_GATLayer


class GSI_A1GATRegressor(nn.Module):
    """
    A1-only naive GAT interpolation model.

    Architecture
    ------------
    Gauge/context node features
        -> GAT over A1 gauge-to-target edges
        -> regression head

    No:
        - A2 mesh propagation
        - GCN
        - GRU
        - temporal attention
        - ERA5
        - IMERG
        - inverse-distance edge weights

    Edge convention
    ---------------
    edge_index_a1[0] = receiver
    edge_index_a1[1] = sender

    The graph builder stores edge_weight_a1 as an inverse-distance weight.
    This naive baseline deliberately ignores it. Future graph artifacts should
    also store edge_distance_a1_km so edge-aware variants do not need to
    reconstruct raw distance from inverse-distance weights.
    """

    def __init__(
        self,
        edge_index_a1: torch.Tensor,
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
        out_channels: int = 1,
        gauge_mask_channel: int = 1,
        adaptive_edges: bool = False,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.gauge_mask_channel = int(gauge_mask_channel)

        if not 0 <= self.gauge_mask_channel < self.in_channels:
            raise ValueError(
                "gauge_mask_channel must refer to a channel in the "
                f"model input. Got channel={self.gauge_mask_channel}, "
                f"in_channels={self.in_channels}"
            )

        self.gat = GSI_GATLayer(
            edge_index=edge_index_a1,
            num_nodes=self.num_nodes,
            in_dim=self.in_channels,
            out_dim=self.hidden_dim,
            heads=int(heads),
            concat=False,
            dropout=float(dropout),
            adaptive_edges=bool(adaptive_edges),
        )

        self.dropout = nn.Dropout(float(dropout))

        self.head = nn.Linear(
            self.hidden_dim,
            int(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor [B,T,N,F].

        Returns
        -------
        torch.Tensor
            Predictions [B,N,out_channels].
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x [B,T,N,F], got {tuple(x.shape)}"
            )

        B, _, N, F_in = x.shape

        if N != self.num_nodes:
            raise ValueError(
                f"N mismatch in GSI_A1GATRegressor: "
                f"got {N}, expected {self.num_nodes}"
            )

        if F_in != self.in_channels:
            raise ValueError(
                f"Feature mismatch in GSI_A1GATRegressor: "
                f"got {F_in}, expected {self.in_channels}"
            )

        # Initial experiment uses only the prediction day.
        x_t = x[:, -1, :, :]  # [B,N,F]

        # Context gauges have mask=1. Missing gauges and hidden
        # target gauges have mask=0.
        source_mask = (
            x_t[:, :, self.gauge_mask_channel] > 0.5
        )

        h = self.gat(
            x_t,
            source_mask=source_mask,
        )

        h = F.elu(h)
        h = self.dropout(h)

        return self.head(h)
