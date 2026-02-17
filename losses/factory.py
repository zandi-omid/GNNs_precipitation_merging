from __future__ import annotations
from typing import Any
import torch.nn as nn

from .masked_mse import MaskedMSELoss
from .huber import MaskedHuberLoss
from .quantile import QuantileLoss

def build_loss(name: str, **kwargs: Any) -> nn.Module:
    name = (name or "mse").lower()

    if name == "mse":
        return MaskedMSELoss()

    if name == "huber":
        return MaskedHuberLoss(delta=float(kwargs.get("delta", 1.0)))

    if name == "quantile":
        return QuantileLoss(
            n_quantiles=int(kwargs.get("n_quantiles", 32)),
            crossing_weight=float(kwargs.get("crossing_weight", 0.0)),
        )

    raise ValueError(f"Unknown loss: {name}")