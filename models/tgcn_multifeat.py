# models/tgcn_multifeat.py
from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Adjacency builder (your version, but keep dense output)
# ============================================================
def build_normalized_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    add_self_loops: bool = True,
    improved: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build A_hat = D^{-1/2} (A + I) D^{-1/2}.

    edge_index: [2, E]
    edge_weight: [E] or None (=> all ones)
    returns: dense [N, N] float32
    """
    device = edge_index.device
    dtype = torch.float32

    row, col = edge_index[0], edge_index[1]
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

    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, row, w)
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)

    w_norm = w * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    A_hat = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=w_norm,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    return A_hat.to_dense()


# ============================================================
# 2-layer GCN block (this is your "2 GCN layers")
# ============================================================
class GCNStack(nn.Module):
    """
    A simple stacked GCN using precomputed dense A_hat.

    Input:  x  [B, N, C]
    Output: y  [B, N, out_dim]
    """

    def __init__(
        self,
        A_hat: torch.Tensor,   # [N, N] dense
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        gcn_layers: int = 2,
    ):
        super().__init__()
        if gcn_layers < 1:
            raise ValueError("gcn_layers must be >= 1")

        self.register_buffer("A_hat", A_hat)

        layers = []
        d_in = in_dim
        for _ in range(gcn_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Graph mixing
        # A_hat: [N,N], x: [B,N,C] -> [B,N,C]
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_mix = torch.matmul(self.A_hat.float(), x.float())
        x_mix = x_mix.to(x.dtype)
        return self.mlp(x_mix)


# ============================================================
# One TGCN cell (GRU-like) where gates use GCNStack
# ============================================================
class TGCNCell(nn.Module):
    """
    One recurrent layer (one GRU layer) using graph convs inside gates.

    x_t:   [B, N, F]
    hprev: [B, N, H]
    returns h: [B, N, H]
    """

    def __init__(
        self,
        A_hat: torch.Tensor,          # [N,N]
        in_channels: int,
        hidden_dim: int,
        gcn_layers: int = 2,          # <-- "2 GCN layers"
        gcn_hidden: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        gcn_hidden = gcn_hidden or hidden_dim

        gate_in = in_channels + hidden_dim

        # update gate z, reset gate r, candidate h~
        self.gcn_z = GCNStack(A_hat, gate_in, gcn_hidden, hidden_dim, gcn_layers=gcn_layers)
        self.gcn_r = GCNStack(A_hat, gate_in, gcn_hidden, hidden_dim, gcn_layers=gcn_layers)
        self.gcn_h = GCNStack(A_hat, gate_in, gcn_hidden, hidden_dim, gcn_layers=gcn_layers)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        B, N, _ = x_t.shape

        if h_prev is None:
            h_prev = torch.zeros(B, N, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)

        xh = torch.cat([x_t, h_prev], dim=-1)            # [B,N,F+H]
        z = torch.sigmoid(self.gcn_z(xh))                # [B,N,H]
        r = torch.sigmoid(self.gcn_r(xh))                # [B,N,H]

        xh_r = torch.cat([x_t, r * h_prev], dim=-1)      # [B,N,F+H]
        h_tilde = torch.tanh(self.gcn_h(xh_r))            # [B,N,H]

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
        A_hat: torch.Tensor,
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        rnn_layers: int = 2,      # <-- "2 GRU layers"
        gcn_layers: int = 2,      # <-- "2 GCN layers"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers

        self.cells = nn.ModuleList()
        for i in range(rnn_layers):
            self.cells.append(
                TGCNCell(
                    A_hat=A_hat,
                    in_channels=in_channels if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    gcn_layers=gcn_layers,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,N,F]
        return: [B,N,H] last hidden state of top layer
        """
        B, T, N, F = x.shape
        if N != self.num_nodes:
            raise ValueError(f"N mismatch: got {N}, expected {self.num_nodes}")
        if F != self.in_channels:
            raise ValueError(f"F mismatch: got {F}, expected {self.in_channels}")

        hs = [None] * self.rnn_layers  # each will become [B,N,H]

        for t in range(T):
            inp = x[:, t, :, :]  # [B,N,F]
            for l in range(self.rnn_layers):
                hs[l] = self.cells[l](inp, hs[l])  # [B,N,H]
                inp = hs[l]
        return hs[-1]  # top layer last hidden: [B,N,H]


# ============================================================
# Regressor head: per-node output
# ============================================================
class TGCNRegressor(nn.Module):
    """
    TGCN backbone (stacked GRU layers, with stacked GCN inside gates)
    + per-node regression head.

    This matches your desired: 2 layers of GRU + 2 layers of GCN.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        rnn_layers: int = 2,      # <-- set 2
        gcn_layers: int = 2,      # <-- set 2
        add_self_loops: bool = True,
        improved: bool = False,
    ):
        super().__init__()

        A_hat = build_normalized_adjacency(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            add_self_loops=add_self_loops,
            improved=improved,
        )
        self.register_buffer("A_hat", A_hat)

        self.backbone = TGCNBackbone(
            A_hat=self.A_hat,
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            rnn_layers=rnn_layers,
            gcn_layers=gcn_layers,
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,N,F]
        returns: [B,N]
        """
        h_last = self.backbone(x)                 # [B,N,H]
        y_hat = self.head(h_last).squeeze(-1)     # [B,N]
        return y_hat