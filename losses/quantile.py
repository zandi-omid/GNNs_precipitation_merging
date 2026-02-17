from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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
    crossing_weight: float = 0.0


class QuantileLoss(nn.Module):
    """
    Pinball loss for multiple quantiles.

    preds:  [B, N, Q]
    target: [B, N]
    mask:   [B, N] bool (optional)
    """
    def __init__(self, n_quantiles: int = 32, crossing_weight: float = 0.0):
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.crossing_weight = float(crossing_weight)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if preds.ndim != 3:
            raise ValueError(f"QuantileLoss expects preds [B,N,Q], got {tuple(preds.shape)}")
        if target.ndim != 2:
            raise ValueError(f"QuantileLoss expects target [B,N], got {tuple(target.shape)}")

        B, N, Q = preds.shape
        if Q != self.n_quantiles:
            raise ValueError(f"Q mismatch: preds has Q={Q}, expected {self.n_quantiles}")

        # --- Build mask (bool) ---
        finite_t = torch.isfinite(target)
        if mask is None:
            mask_b = finite_t
        else:
            mask_b = mask.to(dtype=torch.bool) & finite_t  # robust to int/float masks

        # If nothing is valid, return 0 without NaNs / syncs
        valid = mask_b.sum()  # tensor on device
        if valid == 0:
            return preds.new_zeros(())

        # --- AMP-safe math in float32 ---
        preds32 = preds.float()
        target32 = target.float()

        qs = make_quantiles(self.n_quantiles, device=preds.device, dtype=torch.float32)  # [Q]
        qs_ = qs.view(1, 1, Q)  # [1,1,Q]

        y = target32.unsqueeze(-1)            # [B,N,1]
        e = y - preds32                       # [B,N,Q]
        loss = torch.maximum(qs_ * e, (qs_ - 1.0) * e)  # [B,N,Q]

        m = mask_b.unsqueeze(-1)              # [B,N,1] bool
        loss = loss.masked_fill(~m, 0.0)

        denom = valid.to(loss.dtype) * float(Q)
        base = loss.sum() / denom

        if self.crossing_weight > 0.0 and Q > 1:
            diffs = preds32[:, :, 1:] - preds32[:, :, :-1]   # [B,N,Q-1]
            crossing = torch.relu(-diffs)                    # penalize decreases
            crossing = crossing.masked_fill(~mask_b.unsqueeze(-1), 0.0)

            denom2 = valid.to(loss.dtype) * float(Q - 1)
            cross_pen = crossing.sum() / denom2
            base = base + float(self.crossing_weight) * cross_pen

        # return in original dtype (optional; base is fp32 scalar anyway)
        return base