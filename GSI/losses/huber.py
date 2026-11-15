import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import masked_select

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, preds, target, mask=None):
        pred, obs = masked_select(preds, target, mask)
        if pred is None:
            return torch.tensor(0.0, device=preds.device)
        return F.huber_loss(pred, obs, delta=self.delta, reduction="mean")