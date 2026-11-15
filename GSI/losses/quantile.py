from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


def make_quantiles(n: int, *, device=None, dtype=None) -> torch.Tensor:
    if n < 1:
        raise ValueError(f"n_quantiles must be >= 1, got {n}")

    i = torch.arange(1, n + 1, device=device, dtype=dtype)
    return i / (n + 1)


@dataclass
class QuantileLossConfig:
    n_quantiles: int = 32
    quantiles: Optional[list[float]] = None
    quantile_weights: Optional[list[float]] = None
    higher_q_weight: Optional[float] = None
    crossing_weight: float = 0.0


class QuantileLoss(nn.Module):
    """
    Masked weighted multi-quantile pinball loss.

    Supports two modes:

    1) Old uniform mode:
       QuantileLoss(n_quantiles=32)

       uses:
       q = 1/(Q+1), ..., Q/(Q+1)

    2) Explicit paper-style mode:
       QuantileLoss(
           quantiles=[0.5, 0.9, 0.95],
           higher_q_weight=0.5,
       )

       uses:
       q = [0.5, 0.9, 0.95]
       weights = [1.0, 0.5, 0.5]

    preds:  [B, N, Q]
    target: [B, N]
    mask:   [B, N] bool
    """

    def __init__(
        self,
        n_quantiles: int = 32,
        quantiles: Optional[Sequence[float]] = None,
        quantile_weights: Optional[Sequence[float]] = None,
        higher_q_weight: Optional[float] = None,
        crossing_weight: float = 0.0,
    ):
        super().__init__()

        if quantiles is None:
            q_tensor = make_quantiles(
                int(n_quantiles),
                device=None,
                dtype=torch.float32,
            )
        else:
            if len(quantiles) == 0:
                raise ValueError("quantiles must not be empty")

            q_tensor = torch.tensor(list(quantiles), dtype=torch.float32)

            if torch.any(q_tensor <= 0.0) or torch.any(q_tensor >= 1.0):
                raise ValueError(f"All quantiles must be in (0,1), got {quantiles}")

            # Keep quantiles sorted for stable crossing penalty.
            q_tensor, sort_idx = torch.sort(q_tensor)

            if quantile_weights is not None:
                qw = torch.tensor(list(quantile_weights), dtype=torch.float32)

                if len(qw) != len(q_tensor):
                    raise ValueError(
                        "quantile_weights must have same length as quantiles"
                    )

                qw = qw[sort_idx]
                quantile_weights = qw.tolist()

        self.n_quantiles = int(q_tensor.numel())
        self.crossing_weight = float(crossing_weight)

        if quantile_weights is not None:
            w_tensor = torch.tensor(list(quantile_weights), dtype=torch.float32)

            if len(w_tensor) != self.n_quantiles:
                raise ValueError(
                    f"quantile_weights length={len(w_tensor)} does not match "
                    f"n_quantiles={self.n_quantiles}"
                )

        elif higher_q_weight is not None:
            # Paper-style:
            # first quantile gets weight 1.0, all others get higher_q_weight.
            w_tensor = torch.ones(self.n_quantiles, dtype=torch.float32)

            if self.n_quantiles > 1:
                w_tensor[1:] = float(higher_q_weight)

        else:
            # Normal unweighted quantile loss.
            w_tensor = torch.ones(self.n_quantiles, dtype=torch.float32)

        if torch.any(w_tensor < 0):
            raise ValueError(f"quantile weights must be non-negative, got {w_tensor}")

        self.register_buffer("quantiles", q_tensor)
        self.register_buffer("quantile_weights", w_tensor)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if preds.ndim != 3:
            raise ValueError(
                f"QuantileLoss expects preds [B,N,Q], got {tuple(preds.shape)}"
            )

        if target.ndim != 2:
            raise ValueError(
                f"QuantileLoss expects target [B,N], got {tuple(target.shape)}"
            )

        B, N, Q = preds.shape

        if Q != self.n_quantiles:
            raise ValueError(
                f"Q mismatch: preds has Q={Q}, expected {self.n_quantiles}. "
                f"quantiles={self.quantiles.detach().cpu().tolist()}"
            )

        finite_t = torch.isfinite(target)

        if mask is None:
            mask_b = finite_t
        else:
            mask_b = mask.to(dtype=torch.bool) & finite_t

        valid = mask_b.sum()

        if valid == 0:
            return preds.new_zeros(())

        preds32 = preds.float()
        target32 = target.float()

        qs = self.quantiles.to(device=preds.device, dtype=torch.float32).view(1, 1, Q)
        ws = self.quantile_weights.to(device=preds.device, dtype=torch.float32).view(1, 1, Q)

        y = target32.unsqueeze(-1)
        e = y - preds32

        loss = torch.maximum(qs * e, (qs - 1.0) * e)
        loss = loss * ws

        m = mask_b.unsqueeze(-1)
        loss = loss.masked_fill(~m, 0.0)

        # Normalize by valid nodes and total quantile weight.
        # If all weights are 1, this becomes the standard average pinball loss.
        weight_sum = self.quantile_weights.sum().to(device=preds.device, dtype=torch.float32)
        denom = valid.to(loss.dtype) * weight_sum.clamp_min(1e-12)

        base = loss.sum() / denom

        if self.crossing_weight > 0.0 and Q > 1:
            diffs = preds32[:, :, 1:] - preds32[:, :, :-1]
            crossing = torch.relu(-diffs)
            crossing = crossing.masked_fill(~mask_b.unsqueeze(-1), 0.0)

            denom2 = valid.to(loss.dtype) * float(Q - 1)
            cross_pen = crossing.sum() / denom2.clamp_min(1e-12)

            base = base + float(self.crossing_weight) * cross_pen

        return base

    def extra_repr(self) -> str:
        return (
            f"quantiles={self.quantiles.detach().cpu().tolist()}, "
            f"weights={self.quantile_weights.detach().cpu().tolist()}, "
            f"crossing_weight={self.crossing_weight}"
        )