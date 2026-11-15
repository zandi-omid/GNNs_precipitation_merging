from __future__ import annotations
from typing import Optional, Tuple
import torch

def masked_select(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Supports:
      preds:   [B,N] or [B,N,1]
      target:  [B,N]
      mask:    [B,N] bool (optional)

    Returns:
      (pred_1d, targ_1d) or (None, None) if no valid points.
    """
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)

    if mask is None:
        m = torch.isfinite(target)
    else:
        m = mask & torch.isfinite(target)

    if m.sum().item() == 0:
        return None, None

    return preds[m].float(), target[m].float()