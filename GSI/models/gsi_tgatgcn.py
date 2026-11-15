# models/gsi_tgatgcn.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gsi_layers import (
    GSI_GATLayer,
    GSI_GCNLayer,
    TemporalAttention,
    build_row_normalized_adjacency_coo,
)
from models.gsi_gatgcn import A1DistanceAttentionMixin, mask_receiver_feature_channels


class GSI_GATGCNSpatialEncoder(A1DistanceAttentionMixin, nn.Module):
    """
    Two-stage spatial encoder:

        X_t
          -> GAT on A1: gauge/context-to-mesh injection
          -> GCN on A2: mesh-to-mesh propagation

    Expected edge convention:
        edge_index[0] = receiver
        edge_index[1] = sender
    """

    def __init__(
        self,
        edge_index_a1: torch.Tensor,
        edge_weight_a1: Optional[torch.Tensor],
        edge_index_a2: torch.Tensor,
        edge_weight_a2: Optional[torch.Tensor],
        num_nodes: int,
        in_channels: int,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
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
        a1_hidden_dim: int = 64,
        a1_feature_channels: Optional[list[int]] = None,
        aux_feature_channels: Optional[list[int]] = None,
        aux_hidden_dim: int = 32,
        gcn_intermediate_dim: int = 128,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"legacy", "encoded_late"}:
            raise ValueError("fusion_mode must be 'legacy' or 'encoded_late'")
        self.a1_hidden_dim = int(a1_hidden_dim)
        self.a1_feature_channels = [
            int(c) for c in (a1_feature_channels or [0, 1])
        ]
        self.aux_feature_channels = [
            int(c) for c in (aux_feature_channels or [2, 3])
        ]
        if self.fusion_mode == "encoded_late" and set(self.a1_feature_channels) & set(self.aux_feature_channels):
            raise ValueError("A1 and auxiliary feature channels must be disjoint")
        for channel in self.a1_feature_channels + self.aux_feature_channels:
            if channel < 0 or channel >= self.in_channels:
                raise ValueError(f"Feature channel {channel} is outside input dimension {self.in_channels}")
        if gauge_mask_channel is not None and int(gauge_mask_channel) not in self.a1_feature_channels:
            raise ValueError("gauge_mask_channel must be included in a1_feature_channels")
        self.gauge_mask_channel = gauge_mask_channel
        self.receiver_gauge_feature_channels = (
            [int(c) for c in receiver_gauge_feature_channels]
            if receiver_gauge_feature_channels
            else []
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

        # ----------------------------------------------------
        # Stage 1: A1 gauge/context injection using GAT
        # ----------------------------------------------------
        # A1 intentionally has no self-loops. Its purpose is to
        # inject information from available gauge/context nodes
        # into neighboring mesh nodes.
        if self.fusion_mode == "encoded_late" and self.a1_hidden_dim % int(heads) != 0:
            raise ValueError("a1_hidden_dim must be divisible by heads")
        gat_in_dim = (
            len(self.a1_feature_channels)
            if self.fusion_mode == "encoded_late"
            else self.in_channels
        )
        gat_out_dim = (
            self.a1_hidden_dim // int(heads)
            if self.fusion_mode == "encoded_late"
            else self.hidden_dim
        )
        self.gat1 = GSI_GATLayer(
            edge_index=edge_index_a1,
            num_nodes=self.num_nodes,
            in_dim=gat_in_dim,
            out_dim=gat_out_dim,
            heads=int(heads),
            concat=self.fusion_mode == "encoded_late",
            dropout=float(dropout),
            adaptive_edges=bool(adaptive_edges),
        )

        # ----------------------------------------------------
        # Stage 2: A2 mesh propagation using GCN
        # ----------------------------------------------------
        # A2 includes self-loops so that each node retains its
        # representation while receiving neighboring mesh messages.
        A2_idx, A2_val = build_row_normalized_adjacency_coo(
            edge_index=edge_index_a2,
            edge_weight=edge_weight_a2,
            num_nodes=self.num_nodes,
            add_self_loops=True,
        )

        if self.fusion_mode == "encoded_late":
            self.aux_encoder = nn.Sequential(
                nn.Linear(len(self.aux_feature_channels), int(aux_hidden_dim)),
                nn.ELU(),
                nn.LayerNorm(int(aux_hidden_dim)),
            )
            self.gc1 = GSI_GCNLayer(
                A_idx=A2_idx,
                A_val=A2_val,
                num_nodes=self.num_nodes,
                in_dim=self.a1_hidden_dim + int(aux_hidden_dim),
                out_dim=int(gcn_intermediate_dim),
                adaptive_edges=bool(adaptive_edges),
            )
            gc2_in_dim = int(gcn_intermediate_dim)
        else:
            self.aux_encoder = None
            self.gc1 = None
            gc2_in_dim = self.hidden_dim
        self.gc2 = GSI_GCNLayer(
            A_idx=A2_idx,
            A_val=A2_val,
            num_nodes=self.num_nodes,
            in_dim=gc2_in_dim,
            out_dim=self.hidden_dim,
            adaptive_edges=bool(adaptive_edges),
        )

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t
            Tensor with shape [B, N, F].

        Returns
        -------
        Tensor
            Spatial representation with shape [B, N, H].
        """
        if x_t.ndim != 3:
            raise ValueError(
                f"Expected x_t [B,N,F], got shape {tuple(x_t.shape)}"
            )

        B, N, F_in = x_t.shape

        if N != self.num_nodes:
            raise ValueError(
                f"N mismatch in GSI_GATGCNSpatialEncoder: "
                f"got {N}, expected {self.num_nodes}"
            )

        if F_in != self.in_channels:
            raise ValueError(
                f"Feature mismatch in GSI_GATGCNSpatialEncoder: "
                f"got {F_in}, expected {self.in_channels}"
            )

        # A1: learned attention-based gauge/context injection.
        source_mask = None
        if self.gauge_mask_channel is not None:
            source_mask = x_t[:, :, int(self.gauge_mask_channel)] > 0.5
        x_a1 = (
            x_t[:, :, self.a1_feature_channels]
            if self.fusion_mode == "encoded_late"
            else x_t
        )
        receiver_channels = (
            self.a1_receiver_gauge_feature_channels
            if self.fusion_mode == "encoded_late"
            else self.receiver_gauge_feature_channels
        )
        receiver_x = mask_receiver_feature_channels(
            x_a1,
            receiver_channels,
        )
        h_a1 = self.gat1(
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
        h_a1 = F.elu(h_a1)
        if self.fusion_mode == "encoded_late":
            assert self.aux_encoder is not None and self.gc1 is not None
            h_aux = self.aux_encoder(x_t[:, :, self.aux_feature_channels])
            h = torch.cat([h_a1, h_aux], dim=-1)
            h = self.dropout(F.elu(self.gc1(h)))
        else:
            h = self.dropout(h_a1)

        # A2: distance-weighted mesh propagation.
        h = self.gc2(h)
        h = F.elu(h)

        return h


class GSI_TGATGCNRegressor(nn.Module):
    """
    Temporal GAT-GCN hybrid for sparse-gauge precipitation reconstruction.

    Spatial processing at every timestep:

        X_t
          -> GAT(A1)
          -> GCN(A2)
          -> spatial embedding H_t

    Temporal processing:

        [H_1, ..., H_T]
          -> node-wise GRU
          -> final hidden state or temporal attention
          -> regression/quantile head

    Model variants
    --------------
    TGATGCN:
        use_temporal_attention=False

    A3T-GATGCN:
        use_temporal_attention=True
    """

    def __init__(
        self,
        edge_index_a1: torch.Tensor,
        edge_weight_a1: Optional[torch.Tensor],
        edge_index_a2: torch.Tensor,
        edge_weight_a2: Optional[torch.Tensor],
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
        gauge_mask_channel: Optional[int] = None,
        receiver_gauge_feature_channels: Optional[list[int]] = None,
        edge_distance_a1_km: Optional[torch.Tensor] = None,
        distance_attention: str = "none",
        distance_epsilon_km: float = 1e-3,
        distance_lambda_init: float = 1.0,
        distance_per_head: bool = False,
        distance_lambda_mode: Optional[str] = None,
        fusion_mode: str = "legacy",
        a1_hidden_dim: int = 64,
        a1_feature_channels: Optional[list[int]] = None,
        aux_feature_channels: Optional[list[int]] = None,
        aux_hidden_dim: int = 32,
        gcn_intermediate_dim: int = 128,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.rnn_layers = int(rnn_layers)

        self.use_temporal_attention = bool(use_temporal_attention)
        self.return_attention = bool(return_attention)

        if self.rnn_layers < 1:
            raise ValueError(
                f"rnn_layers must be >= 1, got {self.rnn_layers}"
            )

        self.spatial = GSI_GATGCNSpatialEncoder(
            edge_index_a1=edge_index_a1,
            edge_weight_a1=edge_weight_a1,
            edge_index_a2=edge_index_a2,
            edge_weight_a2=edge_weight_a2,
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            heads=int(heads),
            dropout=float(dropout),
            adaptive_edges=bool(adaptive_edges),
            gauge_mask_channel=gauge_mask_channel,
            receiver_gauge_feature_channels=receiver_gauge_feature_channels,
            edge_distance_a1_km=edge_distance_a1_km,
            distance_attention=distance_attention,
            distance_epsilon_km=distance_epsilon_km,
            distance_lambda_init=distance_lambda_init,
            distance_per_head=distance_per_head,
            distance_lambda_mode=distance_lambda_mode,
            fusion_mode=fusion_mode,
            a1_hidden_dim=a1_hidden_dim,
            a1_feature_channels=a1_feature_channels,
            aux_feature_channels=aux_feature_channels,
            aux_hidden_dim=aux_hidden_dim,
            gcn_intermediate_dim=gcn_intermediate_dim,
        )
        # The GRU is independently applied to each node's time series.
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=float(dropout) if self.rnn_layers > 1 else 0.0,
        )

        if self.use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=self.hidden_dim,
                attn_hidden=attn_hidden,
            )
        else:
            self.temporal_attention = None

        self.head = nn.Linear(
            self.hidden_dim,
            int(out_channels),
        )

    @property
    def lambda_distance(self) -> Optional[torch.Tensor]:
        return self.spatial.lambda_distance

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x
            Input tensor [B, T, N, F].

        Returns
        -------
        Tensor
            Predictions [B, N, out_channels].

        If return_attention=True and temporal attention is enabled,
        returns:
            predictions, alpha

        where:
            alpha has shape [B, N, T].
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x [B,T,N,F], got shape {tuple(x.shape)}"
            )

        B, T, N, F_in = x.shape

        if N != self.num_nodes:
            raise ValueError(
                f"N mismatch in GSI_TGATGCNRegressor: "
                f"got {N}, expected {self.num_nodes}"
            )

        if F_in != self.in_channels:
            raise ValueError(
                f"Feature mismatch in GSI_TGATGCNRegressor: "
                f"got {F_in}, expected {self.in_channels}"
            )

        # ----------------------------------------------------
        # Spatial encoding for every timestep
        # ----------------------------------------------------
        spatial_seq = []

        for t in range(T):
            h_t = self.spatial(x[:, t, :, :])  # [B,N,H]
            spatial_seq.append(h_t)

        h_seq = torch.stack(
            spatial_seq,
            dim=1,
        )  # [B,T,N,H]

        # ----------------------------------------------------
        # Node-wise temporal GRU
        # ----------------------------------------------------
        z = (
            h_seq
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B * N, T, self.hidden_dim)
        )  # [B*N,T,H]

        z_out, _ = self.gru(z)  # [B*N,T,H]

        z_out = (
            z_out
            .reshape(B, N, T, self.hidden_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B,T,N,H]

        # ----------------------------------------------------
        # A3T-style temporal attention
        # ----------------------------------------------------
        if self.use_temporal_attention:
            assert self.temporal_attention is not None

            h_context, alpha = self.temporal_attention(z_out)
            # h_context: [B,N,H]
            # alpha:     [B,N,T]

            y_hat = self.head(h_context)

            if self.return_attention:
                return y_hat, alpha

            return y_hat

        # Standard TGATGCN uses the final GRU state.
        h_last = z_out[:, -1, :, :]  # [B,N,H]
        y_hat = self.head(h_last)

        return y_hat
