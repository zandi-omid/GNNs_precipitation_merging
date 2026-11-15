from __future__ import annotations
from .masked_mse import MaskedMSELoss
from .huber import MaskedHuberLoss
from .quantile import QuantileLoss
from .kge import MaskedKGELoss

def build_loss(
    name: str,
    n_quantiles: int = 32,
    quantiles=None,
    quantile_weights=None,
    higher_q_weight=None,
    crossing_weight: float = 0.0,
    delta: float = 1.0,
):

    name = (name or "mse").lower()
    if name == "quantile":
        return QuantileLoss(
            n_quantiles=n_quantiles,
            quantiles=quantiles,
            quantile_weights=quantile_weights,
            higher_q_weight=higher_q_weight,
            crossing_weight=crossing_weight,
        )

    if name == "huber":
        return MaskedHuberLoss(delta=float(delta))

    if name == "mse":
        return MaskedMSELoss()

    if name == "kge":
        return MaskedKGELoss()

    raise ValueError(f"Unknown loss: {name}")
