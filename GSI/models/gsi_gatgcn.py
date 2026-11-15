# models/gsi_gatgcn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

from models.gsi_layers import (
    GSI_GATLayer,
    GSI_GCNLayer,
    build_row_normalized_adjacency_coo,
)


def inv_softplus(value: float) -> float:
    v = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(v.clamp_min(1e-6))))


def normalize_distance_attention_mode(mode: Optional[str]) -> str:
    mode = str(mode or "none").strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "standard": "none",
        "naive": "none",
        "learnable_log": "learnable_log_distance",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "learnable_log_distance"}:
        raise ValueError(
            "distance_attention must be 'none' or 'learnable_log' "
            f"(internal alias 'learnable_log_distance'); got {mode!r}"
        )
    return mode


def normalize_distance_lambda_mode(mode: Optional[str], per_head: bool = False) -> str:
    if mode is None:
        return "per_head" if per_head else "global"
    mode = str(mode).strip().lower()
    aliases = {
        "scalar": "global",
        "shared": "global",
        "head": "per_head",
        "heads": "per_head",
        "node": "per_receiver_node",
        "receiver_node": "per_receiver_node",
        "per_node": "per_receiver_node",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"global", "per_head", "per_receiver_node"}:
        raise ValueError(
            "distance_lambda_mode must be 'global', 'per_head', "
            "or 'per_receiver_node'; "
            f"got {mode!r}"
        )
    return mode


class A1DistanceAttentionMixin:
    def _init_a1_distance_attention(
        self,
        edge_distance_a1_km: Optional[torch.Tensor],
        num_a1_edges: int,
        num_nodes: int,
        distance_attention: str,
        distance_epsilon_km: float,
        distance_lambda_init: float,
        distance_per_head: bool,
        heads: int,
        distance_lambda_mode: Optional[str] = None,
    ) -> None:
        self.distance_attention = normalize_distance_attention_mode(distance_attention)
        self.distance_epsilon_km = float(distance_epsilon_km)
        self.distance_lambda_mode = normalize_distance_lambda_mode(
            distance_lambda_mode,
            per_head=bool(distance_per_head),
        )
        self.distance_per_head = self.distance_lambda_mode == "per_head"

        if self.distance_attention == "none":
            self.register_buffer("log_edge_distance_a1", torch.empty(0, dtype=torch.float32))
            self.raw_lambda_distance = None
            return

        if edge_distance_a1_km is None:
            raise ValueError(
                "edge_distance_a1_km is required when distance_attention='learnable_log'. "
                "Distances are expected in kilometers and aligned with edge_index_a1 columns."
            )
        edge_distance_a1_km = edge_distance_a1_km.float()
        if edge_distance_a1_km.ndim != 1 or edge_distance_a1_km.numel() != int(num_a1_edges):
            raise ValueError(
                "edge_distance_a1_km must have shape [E_a1], "
                f"got {tuple(edge_distance_a1_km.shape)}, expected [{int(num_a1_edges)}]"
            )
        log_distance = torch.log(edge_distance_a1_km.clamp_min(self.distance_epsilon_km))
        self.register_buffer("log_edge_distance_a1", log_distance.float())

        if self.distance_lambda_mode == "per_head":
            shape = (int(heads),)
        elif self.distance_lambda_mode == "per_receiver_node":
            shape = (int(num_nodes),)
        else:
            shape = ()
        raw_init = inv_softplus(float(distance_lambda_init))
        self.raw_lambda_distance = nn.Parameter(torch.full(shape, raw_init, dtype=torch.float32))

    @property
    def lambda_distance(self) -> Optional[torch.Tensor]:
        if getattr(self, "raw_lambda_distance", None) is None:
            return None
        return F.softplus(self.raw_lambda_distance)


def mask_receiver_feature_channels(x, channels):
    if not channels:
        return None
    receiver_x = x.clone()
    for channel in channels:
        channel = int(channel)
        if channel < 0 or channel >= receiver_x.shape[-1]:
            raise ValueError(
                f"receiver gauge feature channel {channel} is outside "
                f"the input feature dimension {receiver_x.shape[-1]}"
            )
        receiver_x[:, :, channel] = 0.0
    return receiver_x


class GSI_GATGCNRegressor(A1DistanceAttentionMixin, nn.Module):
    def __init__(
        self,
        edge_index_a1,
        edge_weight_a1,
        edge_index_a2,
        edge_weight_a2,
        num_nodes: int,
        in_channels: int = 4,
        hidden_dim: int = 64,
        heads: int = 1,
        dropout: float = 0.0,
        out_channels: int = 1,
        adaptive_edges: bool = False,
        gauge_mask_channel: Optional[int] = None,
        receiver_gauge_feature_channels: Optional[list[int]] = None,
        edge_distance_a1_km: Optional[torch.Tensor] = None,
        distance_attention: str = "none",
        distance_epsilon_km: float = 1e-3,
        distance_lambda_init: float = 1.0,
        distance_per_head: bool = False,
        distance_lambda_mode: Optional[str] = None,
        fusion_mode: str = "legacy",
        a1_hidden_dim: Optional[int] = None,
        a1_feature_channels: Optional[list[int]] = None,
        aux_feature_channels: Optional[list[int]] = None,
        aux_hidden_dim: int = 32,
        gcn_intermediate_dim: int = 128,
    ):
        super().__init__()
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"legacy", "encoded_late"}:
            raise ValueError(
                "fusion_mode must be 'legacy' or 'encoded_late', "
                f"got {fusion_mode!r}"
            )
        self.a1_hidden_dim = int(
            hidden_dim if a1_hidden_dim is None else a1_hidden_dim
        )
        self.a1_feature_channels = [
            int(c) for c in (
                a1_feature_channels
                if a1_feature_channels is not None
                else range(int(in_channels))
            )
        ]
        self.aux_feature_channels = [
            int(c) for c in (aux_feature_channels or [2, 3])
        ]
        self.aux_hidden_dim = int(aux_hidden_dim)
        self.gcn_intermediate_dim = int(gcn_intermediate_dim)
        if self.fusion_mode == "encoded_late":
            if not self.a1_feature_channels:
                raise ValueError("encoded_late fusion requires A1 feature channels")
            for channel in self.a1_feature_channels:
                if channel < 0 or channel >= int(in_channels):
                    raise ValueError(
                        f"A1 feature channel {channel} is outside input dimension "
                        f"{in_channels}"
                    )
            if not self.aux_feature_channels:
                raise ValueError("encoded_late fusion requires auxiliary channels")
            for channel in self.aux_feature_channels:
                if channel < 0 or channel >= int(in_channels):
                    raise ValueError(
                        f"auxiliary channel {channel} is outside input dimension "
                        f"{in_channels}"
                    )
            overlap = set(self.a1_feature_channels) & set(self.aux_feature_channels)
            if overlap:
                warnings.warn(
                    "A1 and auxiliary feature channels overlap. This compatibility "
                    "mode reproduces early encoded-late checkpoints; new experiments "
                    "should use disjoint channels. "
                    f"overlap={sorted(overlap)}",
                    UserWarning,
                    stacklevel=2,
                )
        if gauge_mask_channel is not None and int(gauge_mask_channel) not in self.a1_feature_channels:
            raise ValueError("gauge_mask_channel must be included in a1_feature_channels")

        self.gauge_mask_channel = gauge_mask_channel
        self.receiver_gauge_feature_channels = (
            [int(c) for c in receiver_gauge_feature_channels]
            if receiver_gauge_feature_channels
            else []
        )
        self.a1_gauge_mask_channel = (
            self.a1_feature_channels.index(int(gauge_mask_channel))
            if gauge_mask_channel is not None
            else None
        )
        self.a1_receiver_gauge_feature_channels = [
            self.a1_feature_channels.index(channel)
            for channel in self.receiver_gauge_feature_channels
            if channel in self.a1_feature_channels
        ]
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

        preserve_attention_heads = self.fusion_mode == "encoded_late" and heads > 1
        if preserve_attention_heads and self.a1_hidden_dim % int(heads) != 0:
            raise ValueError(
                "a1_hidden_dim must be divisible by heads when encoded_late "
                "fusion preserves multiple attention summaries"
            )
        a1_head_dim = (
            self.a1_hidden_dim // int(heads)
            if preserve_attention_heads
            else self.a1_hidden_dim
        )
        self.gat1 = GSI_GATLayer(
            edge_index=edge_index_a1,
            num_nodes=num_nodes,
            in_dim=len(self.a1_feature_channels),
            out_dim=a1_head_dim,
            heads=heads,
            concat=preserve_attention_heads,
            dropout=dropout,
            adaptive_edges=adaptive_edges,
        )

        A2_idx, A2_val = build_row_normalized_adjacency_coo(
            edge_index_a2,
            edge_weight_a2,
            num_nodes,
            add_self_loops=True,
        )

        if self.fusion_mode == "encoded_late":
            self.aux_encoder = nn.Sequential(
                nn.Linear(len(self.aux_feature_channels), self.aux_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(self.aux_hidden_dim),
            )
            fusion_dim = self.a1_hidden_dim + self.aux_hidden_dim
            self.gc1 = GSI_GCNLayer(
                A2_idx,
                A2_val,
                num_nodes,
                fusion_dim,
                self.gcn_intermediate_dim,
                adaptive_edges,
            )
            self.gc2 = GSI_GCNLayer(
                A2_idx,
                A2_val,
                num_nodes,
                self.gcn_intermediate_dim,
                hidden_dim,
                adaptive_edges,
            )
        else:
            self.aux_encoder = None
            self.gc1 = None
            self.gc2 = GSI_GCNLayer(
                A2_idx,
                A2_val,
                num_nodes,
                self.a1_hidden_dim,
                hidden_dim,
                adaptive_edges,
            )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        x = x[:, -1, :, :]
        source_mask = None
        if self.gauge_mask_channel is not None:
            source_mask = x[:, :, int(self.gauge_mask_channel)] > 0.5
        x_a1 = x[:, :, self.a1_feature_channels]
        receiver_x = mask_receiver_feature_channels(
            x_a1,
            self.a1_receiver_gauge_feature_channels,
        )
        h_a1 = F.elu(
            self.gat1(
                x_a1,
                source_mask=source_mask,
                receiver_x=receiver_x,
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
        if self.fusion_mode == "encoded_late":
            aux = x[:, :, self.aux_feature_channels]
            h_aux = self.aux_encoder(aux)
            h = torch.cat([h_a1, h_aux], dim=-1)
            h = self.dropout(F.elu(self.gc1(h)))
            h = F.elu(self.gc2(h))
        else:
            h = self.dropout(h_a1)
            h = F.elu(self.gc2(h))
        return self.head(h)
