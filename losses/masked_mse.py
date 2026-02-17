import torch
import torch.nn as nn
from .common import masked_select

class MaskedMSELoss(nn.Module):
    def forward(self, preds, target, mask=None):
        pred, obs = masked_select(preds, target, mask)
        if pred is None:
            return torch.tensor(0.0, device=preds.device)
        return ((pred - obs) ** 2).mean()