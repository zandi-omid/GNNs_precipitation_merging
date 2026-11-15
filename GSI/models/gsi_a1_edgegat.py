# models/gsi_a1_edgegat.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import GSI_GATLayer


def _inv_softplus(value: float) -> float:
    v = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(v.clamp_min(1e-6))))


class GSI_A1DistanceGATRegressor(nn.Module):
    """
    A1-only edge-aware GAT interpolation model.

    This is separate from GSI_A1GATRegressor so the naive baseline remains
    scientifically distinct and unchanged.

    Edge convention and metadata
    ----------------------------
    edge_index_a1[0] = receiver
    edge_index_a1[1] = sender
    edge_distance_a1_km = raw A1 distance in kilometers
    edge_weight_a1 = inverse-distance weight, preserved but unused here

    Supported edge modes
    --------------------
    none:
        exactly the naive GAT attention logits.

    fixed_log_distance:
        e*_ij = e^GAT_ij - log(d_ij + epsilon)

    learnable_log_distance:
        e*_ij = e^GAT_ij - lambda * log(d_ij + epsilon), lambda >= 0

    Future MLP edge-bias mode, not implemented yet:
        b_ij = MLP(log(d_ij + epsilon))
        e*_ij = e^GAT_ij + b_ij
    """

    VALID_EDGE_MODES = {"none", "fixed_log_distance", "learnable_log_distance"}

    def __init__(
        self,
        edge_index_a1: torch.Tensor,
        edge_distance_a1_km: torch.Tensor,
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
        out_channels: int = 1,
        gauge_mask_channel: int = 1,
        edge_mode: str = "learnable_log_distance",
        epsilon_km: float = 1e-3,
        lambda_init: float = 1.0,
        per_head: bool = False,
        adaptive_edges: bool = False,
    ):
        super().__init__()

        if adaptive_edges:
            raise ValueError("adaptive_edges must remain disabled for GSI_A1DistanceGATRegressor")

        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.gauge_mask_channel = int(gauge_mask_channel)
        self.edge_mode = str(edge_mode or "none").lower()
        self.epsilon_km = float(epsilon_km)
        self.per_head = bool(per_head)

        if self.edge_mode not in self.VALID_EDGE_MODES:
            raise ValueError(
                f"edge_mode must be one of {sorted(self.VALID_EDGE_MODES)}, "
                f"got {edge_mode!r}"
            )
        if not 0 <= self.gauge_mask_channel < self.in_channels:
            raise ValueError(
                "gauge_mask_channel must refer to a channel in the model input. "
                f"Got channel={self.gauge_mask_channel}, in_channels={self.in_channels}"
            )
        if edge_distance_a1_km is None and self.edge_mode != "none":
            raise ValueError(
                "edge_distance_a1_km is required for GSI_A1DistanceGATRegressor. "
                "Prefer adding raw kilometer distances to graph artifacts rather "
                "than reconstructing them from inverse-distance weights."
            )

        if edge_distance_a1_km is not None:
            edge_distance_a1_km = edge_distance_a1_km.float()
            if edge_distance_a1_km.ndim != 1 or edge_distance_a1_km.numel() != edge_index_a1.shape[1]:
                raise ValueError(
                    "edge_distance_a1_km must have shape [E], "
                    f"got {tuple(edge_distance_a1_km.shape)}, expected [{edge_index_a1.shape[1]}]"
                )

        self.gat = GSI_GATLayer(
            edge_index=edge_index_a1,
            num_nodes=self.num_nodes,
            in_dim=self.in_channels,
            out_dim=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=float(dropout),
            adaptive_edges=False,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.hidden_dim, int(out_channels))

        if edge_distance_a1_km is None:
            log_distance = torch.empty(0, dtype=torch.float32)
        else:
            log_distance = torch.log(edge_distance_a1_km.clamp_min(self.epsilon_km))
        self.register_buffer("log_edge_distance_a1", log_distance.float())

        if self.edge_mode == "learnable_log_distance":
            shape = (self.heads,) if self.per_head else ()
            raw_init = _inv_softplus(float(lambda_init))
            self.raw_lambda_distance = nn.Parameter(torch.full(shape, raw_init, dtype=torch.float32))
        else:
            self.raw_lambda_distance = None

    @property
    def lambda_distance(self) -> Optional[torch.Tensor]:
        if self.raw_lambda_distance is None:
            return None
        return F.softplus(self.raw_lambda_distance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,N,F], got {tuple(x.shape)}")

        B, _, N, F_in = x.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch in GSI_A1DistanceGATRegressor: got {N}, expected {self.num_nodes}")
        if F_in != self.in_channels:
            raise ValueError(
                f"Feature mismatch in GSI_A1DistanceGATRegressor: got {F_in}, "
                f"expected {self.in_channels}"
            )

        x_t = x[:, -1, :, :]
        source_mask = x_t[:, :, self.gauge_mask_channel] > 0.5
        h = self.gat(
            x_t,
            source_mask=source_mask,
            edge_distance_km=(
                self.log_edge_distance_a1 if self.edge_mode != "none" else None
            ),
            edge_mode=self.edge_mode,
            lambda_distance=self.lambda_distance,
            epsilon_km=self.epsilon_km,
            edge_distance_is_log=True,
        )
        h = F.elu(h)
        h = self.dropout(h)
        return self.head(h)


# Backward-compatible name used by earlier experiments/configs.
GSI_A1EdgeAwareGATRegressor = GSI_A1DistanceGATRegressor
