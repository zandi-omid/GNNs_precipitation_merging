# models/gsi_layers.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_row_normalized_adjacency_coo(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    add_self_loops: bool = False,
    eps: float = 1e-12,
):
    row = edge_index[0].long()
    col = edge_index[1].long()

    if edge_weight is None:
        w = torch.ones(row.numel(), device=edge_index.device, dtype=torch.float32)
    else:
        w = edge_weight.to(device=edge_index.device, dtype=torch.float32)

    if add_self_loops:
        loop = torch.arange(num_nodes, device=edge_index.device)
        row = torch.cat([row, loop])
        col = torch.cat([col, loop])
        w = torch.cat([w, torch.ones(num_nodes, device=edge_index.device)])

    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, w)
    w = w / deg[row].clamp_min(eps)

    A = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        w,
        size=(num_nodes, num_nodes),
    ).coalesce()

    return A.indices().long(), A.values().float()


def sparse_mm_batch(A_idx, A_val, num_nodes: int, x: torch.Tensor):
    B, N, C = x.shape
    if N != num_nodes:
        raise ValueError(f"N mismatch: got {N}, expected {num_nodes}")

    A = torch.sparse_coo_tensor(
        A_idx.to(x.device),
        A_val.to(x.device),
        size=(num_nodes, num_nodes),
        device=x.device,
        dtype=torch.float32,
    ).coalesce()

    x2 = x.permute(1, 0, 2).reshape(num_nodes, B * C)

    with torch.autocast(device_type=x.device.type, enabled=False):
        y2 = torch.sparse.mm(A, x2.float())

    y = y2.reshape(num_nodes, B, C).permute(1, 0, 2)
    return y.to(dtype=x.dtype)


class GSI_GCNLayer(nn.Module):
    def __init__(self, A_idx, A_val, num_nodes, in_dim, out_dim, adaptive_edges=False):
        super().__init__()
        self.register_buffer("A_idx", A_idx.long())
        self.register_buffer("A_val_base", A_val.float())
        self.num_nodes = int(num_nodes)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

        self.adaptive_edges = bool(adaptive_edges)
        if self.adaptive_edges:
            self.edge_mask = nn.Parameter(torch.ones_like(A_val.float()))
        else:
            self.edge_mask = None

    def forward(self, x):
        h = self.linear(x)

        if self.edge_mask is not None:
            # ELU + 1 keeps multiplier positive and starts near 1.
            A_val = self.A_val_base * (F.elu(self.edge_mask) + 1.0)
        else:
            A_val = self.A_val_base

        return sparse_mm_batch(self.A_idx, A_val, self.num_nodes, h)


class GSI_GATLayer(nn.Module):
    """
    GAT layer using edge convention:
      edge_index[0] = receiver
      edge_index[1] = sender
    """

    def __init__(
        self,
        edge_index,
        num_nodes,
        in_dim,
        out_dim,
        heads=4,
        concat=True,
        dropout=0.0,
        adaptive_edges=False,
    ):
        super().__init__()
        self.register_buffer("edge_index", edge_index.long())
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = nn.Dropout(dropout)

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_recv = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_send = nn.Parameter(torch.empty(1, heads, out_dim))

        self.adaptive_edges = bool(adaptive_edges)
        if self.adaptive_edges:
            self.edge_mask = nn.Parameter(torch.ones(edge_index.shape[1], heads))
        else:
            self.edge_mask = None

        out_total = heads * out_dim if concat else out_dim
        self.skip = nn.Linear(in_dim, out_total, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_total))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_recv)
        nn.init.xavier_uniform_(self.att_send)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        receiver_x: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        edge_distance_km: Optional[torch.Tensor] = None,
        edge_mode: str = "none",
        distance_mode: Optional[str] = None,
        lambda_distance: Optional[torch.Tensor] = None,
        epsilon_km: float = 1e-3,
        distance_epsilon_km: Optional[float] = None,
        edge_distance_is_log: bool = False,
    ):
        """
        Parameters
        ----------
        x
            Node features [B, N, F].

        source_mask
            Optional Boolean mask [B, N].

            source_mask[b, j] = True means node j is an active sender
            for sample b.

            Edges from inactive senders are completely excluded from
            the attention softmax and message aggregation.

        receiver_x
            Optional receiver-side node features [B, N, F]. When provided,
            receiver attention scores and the residual/skip projection are
            computed from receiver_x, while sender messages still use x. This
            lets A1 sender gauges provide observations without letting a
            receiver node shortcut through its own revealed gauge value.

        edge_attr
            Reserved extension point for future MLP edge-bias attention. The
            current implemented edge-aware modes do not use this argument.

        edge_distance_km
            Optional A1 edge distances [E], in kilometers. For efficiency,
            callers may pass precomputed log distances and set
            edge_distance_is_log=True.

        edge_mode
            "none", "fixed_log_distance", or "learnable_log_distance".

        distance_mode
            Alias for edge_mode. Kept so distance-aware callers can use a
            distance-specific name without changing older edge_mode callers.

        lambda_distance
            Positive distance scale for learnable_log_distance. Shape may be
            scalar, [heads], or [num_nodes]. A [num_nodes] vector is gathered
            by receiver node so each target node learns its own distance
            sensitivity.

        Returns
        -------
        torch.Tensor
            Output node representations.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected x [B,N,F], got shape {tuple(x.shape)}"
            )

        B, N, _ = x.shape

        if receiver_x is None:
            receiver_x = x
        else:
            if receiver_x.shape != x.shape:
                raise ValueError(
                    "receiver_x must have the same shape as x; "
                    f"got {tuple(receiver_x.shape)} and {tuple(x.shape)}"
                )
            receiver_x = receiver_x.to(device=x.device, dtype=x.dtype)

        if N != self.num_nodes:
            raise ValueError(
                f"N mismatch in GSI_GATLayer: got {N}, "
                f"expected {self.num_nodes}"
            )

        recv = self.edge_index[0].long()
        send = self.edge_index[1].long()
        E = recv.numel()

        x0 = x

        h_send = self.lin(self.dropout(x)).view(
            B,
            N,
            self.heads,
            self.out_dim,
        )

        if receiver_x is x:
            h_recv = h_send
        else:
            h_recv = self.lin(self.dropout(receiver_x)).view(
                B,
                N,
                self.heads,
                self.out_dim,
            )
            x0 = receiver_x

        score_recv = (h_recv * self.att_recv).sum(dim=-1)
        score_send = (h_send * self.att_send).sum(dim=-1)

        e = F.leaky_relu(
            score_recv[:, recv, :] +
            score_send[:, send, :],
            negative_slope=0.2,
        )  # [B,E,H]

        if distance_mode is not None:
            edge_mode = distance_mode
        if distance_epsilon_km is not None:
            epsilon_km = distance_epsilon_km

        edge_mode = str(edge_mode or "none").lower()
        if edge_mode not in {"none", "fixed_log_distance", "learnable_log_distance"}:
            raise ValueError(
                "edge_mode must be one of 'none', 'fixed_log_distance', "
                f"'learnable_log_distance'; got {edge_mode!r}"
            )

        if edge_mode != "none":
            if edge_distance_km is None:
                raise ValueError(f"edge_distance_km is required for edge_mode={edge_mode!r}")
            if edge_distance_km.ndim != 1 or edge_distance_km.numel() != E:
                raise ValueError(
                    "edge_distance_km must have shape [E], "
                    f"got {tuple(edge_distance_km.shape)}, expected [{E}]"
                )
            edge_distance_km = edge_distance_km.to(device=x.device, dtype=e.dtype)
            if edge_distance_is_log:
                log_d = edge_distance_km
            else:
                log_d = torch.log(edge_distance_km.clamp_min(float(epsilon_km)))
            log_d = log_d.view(1, E, 1)

            if edge_mode == "fixed_log_distance":
                e = e - log_d
            else:
                if lambda_distance is None:
                    raise ValueError("lambda_distance is required for learnable_log_distance")
                lam = lambda_distance.to(device=x.device, dtype=e.dtype)
                if lam.ndim == 0 or lam.numel() == 1:
                    lam = lam.reshape(1, 1, 1)
                elif lam.shape == (self.heads,):
                    lam = lam.reshape(1, 1, self.heads)
                elif lam.shape == (self.num_nodes,):
                    lam = lam[recv].reshape(1, E, 1)
                else:
                    raise ValueError(
                        "lambda_distance must be scalar, [heads], or [num_nodes], "
                        f"got {tuple(lam.shape)}"
                    )
                e = e - lam * log_d

        if self.edge_mask is not None:
            positive_edge_multiplier = (
                F.elu(self.edge_mask.to(x.device)) + 1.0
            ).clamp_min(1e-6)

            e = e + torch.log(
                positive_edge_multiplier
            ).unsqueeze(0)

        # --------------------------------------------------------
        # Exclude unavailable or hidden gauges from attention
        # --------------------------------------------------------
        if source_mask is None:
            edge_active = torch.ones(
                B,
                E,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            if source_mask.ndim != 2:
                raise ValueError(
                    "source_mask must have shape [B,N], "
                    f"got {tuple(source_mask.shape)}"
                )

            if source_mask.shape != (B, N):
                raise ValueError(
                    "source_mask shape mismatch: "
                    f"got {tuple(source_mask.shape)}, expected {(B, N)}"
                )

            source_mask = source_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

            # An edge is active only if its sender is an active gauge.
            edge_active = source_mask[:, send]  # [B,E]

        e_masked = e.masked_fill(~edge_active.unsqueeze(-1), -torch.inf)

        max_e = torch.full(
            (B, N, self.heads),
            -torch.inf,
            device=x.device,
            dtype=e.dtype,
        )
        recv_b = recv.view(1, E, 1).expand(B, E, self.heads)
        if hasattr(max_e, "scatter_reduce_"):
            max_e.scatter_reduce_(
                dim=1,
                index=recv_b,
                src=e_masked,
                reduce="amax",
                include_self=True,
            )
        else:
            for edge_pos in range(E):
                r = int(recv[edge_pos])
                max_e[:, r, :] = torch.maximum(max_e[:, r, :], e_masked[:, edge_pos, :])

        edge_max = max_e[:, recv, :]
        edge_max = torch.where(
            torch.isfinite(edge_max),
            edge_max,
            torch.zeros_like(edge_max),
        )

        exp_e = torch.exp(e_masked - edge_max)
        exp_e = torch.where(torch.isfinite(exp_e), exp_e, torch.zeros_like(exp_e))
        exp_e = exp_e * edge_active.unsqueeze(-1).to(exp_e.dtype)

        denom = torch.zeros(
            B,
            N,
            self.heads,
            device=x.device,
            dtype=exp_e.dtype,
        )

        denom.scatter_add_(
            dim=1,
            index=recv_b,
            src=exp_e,
        )

        alpha = exp_e / (
            denom[:, recv, :] + 1e-12
        )

        alpha = self.dropout(alpha)

        msg = h_send[:, send, :, :] * alpha.unsqueeze(-1)

        out = torch.zeros(
            B,
            N,
            self.heads,
            self.out_dim,
            device=x.device,
            dtype=msg.dtype,
        )

        recv_msg = recv.view(1, E, 1, 1).expand(
            B,
            E,
            self.heads,
            self.out_dim,
        )

        out.scatter_add_(
            dim=1,
            index=recv_msg,
            src=msg,
        )

        if self.concat:
            out = out.reshape(
                B,
                N,
                self.heads * self.out_dim,
            )
        else:
            out = out.mean(dim=2)

        out = out + self.skip(x0) + self.bias

        return out


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_hidden: Optional[int] = None):
        super().__init__()
        attn_hidden = int(attn_hidden or hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, h_seq):
        # h_seq: [B,T,N,H]
        e = self.score(h_seq).squeeze(-1).permute(0, 2, 1)  # [B,N,T]
        alpha = torch.softmax(e, dim=-1)
        context = torch.sum(alpha.permute(0, 2, 1).unsqueeze(-1) * h_seq, dim=1)
        return context, alpha
