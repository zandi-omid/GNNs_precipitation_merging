from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Sparse normalized adjacency builder
# ============================================================
def build_normalized_adjacency_coo(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    num_nodes: int,
    add_self_loops: bool = True,
    improved: bool = False,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, row, w)
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)

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
# Sparse graph mix
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

    with torch.cuda.amp.autocast(enabled=False):
        y2 = torch.sparse.mm(A, x2.float())

    y = y2.reshape(num_nodes, B, C).permute(1, 0, 2)
    return y.to(dtype=x.dtype)


# ============================================================
# GCN stack
# ============================================================
class GCNStack(nn.Module):
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
        x_mix = sparse_graph_mix(self.A_idx, self.A_val, self.num_nodes, x)
        return self.mlp(x_mix)


# ============================================================
# Spatial-only GCN regressor
# ============================================================
class GCNRegressor(nn.Module):
    """
    Spatial-only baseline:
      x [B,T,N,F] -> use last timestep x[:, -1, :, :] -> GCN -> head

    This removes recurrence entirely and isolates the value of temporal modeling.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
        in_channels: int = 3,
        hidden_dim: int = 64,
        gcn_layers: int = 2,
        gcn_hidden: Optional[int] = None,
        add_self_loops: bool = True,
        improved: bool = False,
        out_channels: int = 1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.out_channels = int(out_channels)
        gcn_hidden = int(gcn_hidden or hidden_dim)

        A_idx, A_val = build_normalized_adjacency_coo(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            add_self_loops=add_self_loops,
            improved=improved,
        )

        self.register_buffer("A_idx", A_idx)
        self.register_buffer("A_val", A_val)

        self.gcn = GCNStack(
            A_idx=self.A_idx,
            A_val=self.A_val,
            num_nodes=num_nodes,
            in_dim=in_channels,
            hidden_dim=gcn_hidden,
            out_dim=hidden_dim,
            gcn_layers=gcn_layers,
        )

        self.head = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,N,F]
        returns:
          [B,N,1] or [B,N,Q]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,N,F], got {tuple(x.shape)}")

        x_last = x[:, -1, :, :]     # [B,N,F]
        h = self.gcn(x_last)        # [B,N,H]
        y_hat = self.head(h)        # [B,N,Q]
        return y_hat