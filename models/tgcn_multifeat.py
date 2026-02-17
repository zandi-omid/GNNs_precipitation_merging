# models/tgcn_multifeat.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Sparse normalized adjacency builder (COO buffers for DDP-safe broadcast)
# ============================================================
def build_normalized_adjacency_coo(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    add_self_loops: bool = True,
    improved: bool = False,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build normalized adjacency A_hat = D^{-1/2} (A + I) D^{-1/2} in COO form.

    Returns:
      A_idx: [2, nnz] long
      A_val: [nnz] float32

    NOTE: We return COO (idx, val) instead of a sparse tensor to avoid
    PyTorch DDP "No support for sparse tensors" when broadcasting buffers.
    """
    device = edge_index.device
    dtype = torch.float32

    row, col = edge_index[0].long(), edge_index[1].long()
    if edge_weight is None:
        w = torch.ones(row.numel(), device=device, dtype=dtype)
    else:
        w = edge_weight.to(device=device, dtype=dtype)

    if add_self_loops:
        loop_idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        loop_weight = torch.full(
            (num_nodes,),
            2.0 if improved else 1.0,
            device=device,
            dtype=dtype,
        )
        row = torch.cat([row, loop_idx], dim=0)
        col = torch.cat([col, loop_idx], dim=0)
        w = torch.cat([w, loop_weight], dim=0)

    # degree
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, row, w)
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)

    # normalized weights
    w_norm = w * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    A = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=w_norm,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    return A.indices().to(torch.long), A.values().to(torch.float32)


# ============================================================
# Sparse graph mix: A_hat @ X for batched node features
# ============================================================
def sparse_graph_mix(A_idx, A_val, num_nodes, x):
    B, N, C = x.shape
    if N != num_nodes:
        raise ValueError(f"N mismatch: got {N}, expected {num_nodes}")

    A = torch.sparse_coo_tensor(
        A_idx.to(device=x.device),
        A_val.to(device=x.device),
        (num_nodes, num_nodes),
        device=x.device,
        dtype=torch.float32,
    ).coalesce()

    x2 = x.permute(1, 0, 2).reshape(num_nodes, B * C)

    # --- IMPORTANT: disable autocast for sparse.mm and force fp32 ---
    # This avoids: addmm_sparse_cuda not implemented for 'Half'
    with torch.cuda.amp.autocast(enabled=False):
        y2 = torch.sparse.mm(A, x2.float())  # [N, B*C] fp32

    y = y2.reshape(num_nodes, B, C).permute(1, 0, 2)
    return y.to(dtype=x.dtype)


# ============================================================
# 2-layer "GCN" stack (graph mix + MLP)
# ============================================================
class GCNStack(nn.Module):
    """
    Graph mixing (A_hat @ x) followed by an MLP.
    If gcn_layers=2 => Linear -> ReLU -> Linear.

    Input:  x [B, N, Cin]
    Output: y [B, N, Cout]
    """

    def __init__(
        self,
        A_idx: torch.Tensor,
        A_val: torch.Tensor,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        gcn_layers: int = 2,
    ):
        super().__init__()
        if gcn_layers < 1:
            raise ValueError("gcn_layers must be >= 1")

        # DDP-safe buffers (dense)
        self.register_buffer("A_idx", A_idx.to(torch.long))
        self.register_buffer("A_val", A_val.to(torch.float32))
        self.num_nodes = int(num_nodes)

        layers = []
        d_in = in_dim
        for _ in range(gcn_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # graph mix
        x_mix = sparse_graph_mix(self.A_idx, self.A_val, self.num_nodes, x)  # [B,N,Cin]
        return self.mlp(x_mix)  # [B,N,Cout]


# ============================================================
# One TGCN cell (GRU-like) using GCNStack inside gates
# ============================================================
class TGCNCell(nn.Module):
    """
    One recurrent layer (one GRU layer) using graph convs inside gates.

    x_t:   [B, N, F]
    hprev: [B, N, H] or None
    returns h: [B, N, H]
    """

    def __init__(
        self,
        A_idx: torch.Tensor,
        A_val: torch.Tensor,
        num_nodes: int,
        in_channels: int,
        hidden_dim: int,
        gcn_layers: int = 2,          # "2 GCN layers"
        gcn_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        gcn_hidden = int(gcn_hidden or hidden_dim)

        gate_in = self.in_channels + self.hidden_dim

        # update gate z, reset gate r, candidate h~
        self.gcn_z = GCNStack(A_idx, A_val, num_nodes, gate_in, gcn_hidden, self.hidden_dim, gcn_layers=gcn_layers)
        self.gcn_r = GCNStack(A_idx, A_val, num_nodes, gate_in, gcn_hidden, self.hidden_dim, gcn_layers=gcn_layers)
        self.gcn_h = GCNStack(A_idx, A_val, num_nodes, gate_in, gcn_hidden, self.hidden_dim, gcn_layers=gcn_layers)

    def forward(self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, F = x_t.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch in TGCNCell: got {N}, expected {self.num_nodes}")
        if F != self.in_channels:
            raise ValueError(f"F mismatch in TGCNCell: got {F}, expected {self.in_channels}")

        if h_prev is None:
            h_prev = torch.zeros(B, N, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)

        xh = torch.cat([x_t, h_prev], dim=-1)            # [B,N,F+H]
        z = torch.sigmoid(self.gcn_z(xh))                # [B,N,H]
        r = torch.sigmoid(self.gcn_r(xh))                # [B,N,H]

        xh_r = torch.cat([x_t, r * h_prev], dim=-1)      # [B,N,F+H]
        h_tilde = torch.tanh(self.gcn_h(xh_r))           # [B,N,H]

        h = (1.0 - z) * h_prev + z * h_tilde
        return h


# ============================================================
# Stacked recurrent layers across time (this is "2 GRU layers")
# ============================================================
class TGCNBackbone(nn.Module):
    """
    Processes window x [B,T,N,F] and returns last hidden [B,N,H]
    using stacked recurrent layers (rnn_layers).

    - rnn_layers=2 means "2 GRU layers"
    - gcn_layers=2 means "2 GCN layers per gate"
    """

    def __init__(
        self,
        A_idx: torch.Tensor,
        A_val: torch.Tensor,
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        rnn_layers: int = 2,      # "2 GRU layers"
        gcn_layers: int = 2,      # "2 GCN layers"
        gcn_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.rnn_layers = int(rnn_layers)

        self.cells = nn.ModuleList()
        for i in range(self.rnn_layers):
            self.cells.append(
                TGCNCell(
                    A_idx=A_idx,
                    A_val=A_val,
                    num_nodes=num_nodes,
                    in_channels=self.in_channels if i == 0 else self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    gcn_layers=gcn_layers,
                    gcn_hidden=gcn_hidden,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,N,F]
        return: [B,N,H] last hidden state of top layer
        """
        B, T, N, F = x.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch in TGCNBackbone: got {N}, expected {self.num_nodes}")
        if F != self.in_channels:
            raise ValueError(f"F mismatch in TGCNBackbone: got {F}, expected {self.in_channels}")

        hs: list[Optional[torch.Tensor]] = [None] * self.rnn_layers  # each -> [B,N,H]

        for t in range(T):
            inp = x[:, t, :, :]  # [B,N,F] then [B,N,H]
            for l in range(self.rnn_layers):
                hs[l] = self.cells[l](inp, hs[l])  # [B,N,H]
                inp = hs[l]
        return hs[-1]  # [B,N,H]


# ============================================================
# Regressor head: per-node output
# ============================================================
class TGCNRegressor(nn.Module):
    """
    TGCN backbone (stacked GRU layers, with stacked GCN inside gates)
    + per-node regression head.

    Your target architecture:
      - rnn_layers=2 (2 GRU layers)
      - gcn_layers=2 (2 GCN layers per gate)
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        rnn_layers: int = 2,
        gcn_layers: int = 2,
        gcn_hidden: Optional[int] = None,
        add_self_loops: bool = True,
        improved: bool = False,
        out_channels: int = 1,   # NEW: Q
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.out_channels = int(out_channels)

        A_idx, A_val = build_normalized_adjacency_coo(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            add_self_loops=add_self_loops,
            improved=improved,
        )

        # store COO buffers (DDP-safe)
        self.register_buffer("A_idx", A_idx)
        self.register_buffer("A_val", A_val)

        self.backbone = TGCNBackbone(
            A_idx=self.A_idx,
            A_val=self.A_val,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            gcn_layers=gcn_layers,
            gcn_hidden=gcn_hidden,
        )
        # output Q channels per node
        self.head = nn.Linear(hidden_dim, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,N,F]
        returns:
          - if out_channels == 1: [B,N,1]
          - else:                [B,N,Q]
        """
        h_last = self.backbone(x)          # [B,N,H]
        y_hat = self.head(h_last)          # [B,N,Q]
        return y_hat