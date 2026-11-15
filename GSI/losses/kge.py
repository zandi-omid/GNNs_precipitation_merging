from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .common import masked_select


class MaskedKGELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred, obs = masked_select(preds, target, mask)
        if pred is None:
            return preds.new_zeros(())

        if pred.numel() < 2:
            return preds.new_zeros(())

        pred = pred.float()
        obs = obs.float()

        pred_c = pred - pred.mean()
        obs_c = obs - obs.mean()

        denom = torch.sqrt(torch.sum(pred_c ** 2) * torch.sum(obs_c ** 2))
        cc = torch.sum(pred_c * obs_c) / denom.clamp_min(self.eps)

        mu_p = pred.mean()
        mu_o = obs.mean()

        std_p = torch.std(pred, unbiased=False)
        std_o = torch.std(obs, unbiased=False)

        alpha = std_p / (std_o + self.eps)
        beta = mu_p / (mu_o + self.eps)

        kge = 1.0 - torch.sqrt(
            (cc - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )

        return 1.0 - kge
