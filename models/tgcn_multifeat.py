# models/tgcn_multifeat.py
import torch
import torch.nn as nn


def build_normalized_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    add_self_loops: bool = True,
    improved: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build A_hat = D^{-1/2} (A + I) D^{-1/2} as a torch.sparse COO tensor.
    Works without torch_sparse.

    edge_index: [2, E]
    edge_weight: [E] or None (=> all ones)
    returns: sparse COO [N, N]
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
        loop_weight = torch.full((num_nodes,), 2.0 if improved else 1.0, device=device, dtype=dtype)

        row = torch.cat([row, loop_idx], dim=0)
        col = torch.cat([col, loop_idx], dim=0)
        w = torch.cat([w, loop_weight], dim=0)

    # Degree
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.scatter_add_(0, row, w)
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)

    # Normalize weights: w_ij * d_i^{-1/2} * d_j^{-1/2}
    w_norm = w * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    A_hat = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=w_norm,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    return A_hat.to_dense()

class TGCNGraphConvolution(nn.Module):
    """
    Graph convolution used inside the GRU gates:
      Z = A_hat @ [X, H] @ W + b
    """

    def __init__(
        self,
        A_hat: torch.Tensor,   # sparse COO [N, N] (or dense, but you use sparse.mm)
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("A_hat", A_hat)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # W: (F + H) -> out_dim
        self.weights = nn.Parameter(torch.empty(in_channels + hidden_dim, out_dim))
        self.biases = nn.Parameter(torch.empty(out_dim))
        self.reset_parameters(bias_init)

    def reset_parameters(self, bias_init: float):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, bias_init)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        inputs:      [B, N, F]
        hidden_state:[B, N*H]
        returns:     [B, N*out_dim]
        """
        B, N, F = inputs.shape
        H = self.hidden_dim  # ✅ FIX: was self.num_gru_units

        # reshape hidden state
        hidden_state = hidden_state.reshape(B, N, H)  # [B, N, H]

        # concat input + hidden
        concat = torch.cat((inputs, hidden_state), dim=2)  # [B, N, F+H]

        # prepare for A_hat @ concat
        concat = concat.transpose(0, 1).transpose(1, 2)     # [N, F+H, B]
        concat = concat.reshape(N, (F + H) * B)             # [N, (F+H)B]

        # concat is [N, (F+H)B]
        A_hat = self.A_hat  # [N, N] dense

        with torch.amp.autocast(device_type="cuda", enabled=False):
            a_times = A_hat.float() @ concat.float()

        # cast back to model dtype
        a_times = a_times.to(inputs.dtype)

        # reshape back
        a_times = a_times.reshape(N, F + H, B).transpose(0, 2).transpose(1, 2)  # [B, N, F+H]
        a_times = a_times.reshape(B * N, F + H)                                  # [BN, F+H]

        out = a_times @ self.weights + self.biases                               # [BN, out_dim]
        out = out.reshape(B, N, self.out_dim)                                    # ✅ FIX: was self.output_dim
        out = out.reshape(B, N * self.out_dim)                                   # ✅ FIX

        return out


class TGCNCell(nn.Module):
    """
    GRU cell with graph convolutions instead of linear layers.
    """

    def __init__(self, A_hat: torch.Tensor, in_channels: int, hidden_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # gate conv: produces 2H (reset r and update u)
        self.gc_ru = TGCNGraphConvolution(A_hat, in_channels, hidden_dim, out_dim=2 * hidden_dim, bias_init=1.0)
        # candidate conv: produces H
        self.gc_c  = TGCNGraphConvolution(A_hat, in_channels, hidden_dim, out_dim=hidden_dim, bias_init=0.0)

    def forward(self, x_t: torch.Tensor, h_flat: torch.Tensor) -> torch.Tensor:
        """
        x_t:   [B, N, F]
        h_flat:[B, N*H]
        returns new_h_flat: [B, N*H]
        """
        B, N, F = x_t.shape
        H = self.hidden_dim

        ru = torch.sigmoid(self.gc_ru(x_t, h_flat))       # [B, N*(2H)]
        r, u = torch.chunk(ru, chunks=2, dim=1)           # each [B, N*H]

        c = torch.tanh(self.gc_c(x_t, r * h_flat))        # [B, N*H]
        new_h = u * h_flat + (1.0 - u) * c                # [B, N*H]
        return new_h


class TGCNBackbone(nn.Module):
    """
    Processes a window of length T and returns per-node hidden states [B, N, H].
    """

    def __init__(self, A_hat: torch.Tensor, num_nodes: int, in_channels: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.cell = TGCNCell(A_hat, in_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, F]
        return: [B, N, H]
        """
        B, T, N, F = x.shape
        assert N == self.num_nodes
        assert F == self.in_channels

        h = torch.zeros(B, N * self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(T):
            h = self.cell(x[:, t, :, :], h)               # [B, N*H]
        return h.view(B, N, self.hidden_dim)              # [B, N, H]

class TGCNRegressor(nn.Module):
    """
    Stacked TGCN backbone + per-node regression head.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        num_nodes: int,
        in_channels: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
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

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 🔹 Stack TGCN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TGCNBackbone(
                    A_hat=self.A_hat,
                    num_nodes=num_nodes,
                    in_channels=in_channels if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                )
            )

        # Final regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, F]
        returns: [B, N]
        """

        out = x
        for layer in self.layers:
            out = layer(out)                  # [B, N, H]
            out = out.unsqueeze(1)            # [B, 1, N, H] → fake T=1 for next layer

        y_hat = self.head(out.squeeze(1)).squeeze(-1)  # [B, N]
        return y_hat
