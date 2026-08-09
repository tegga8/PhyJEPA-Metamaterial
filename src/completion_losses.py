"""BCE losses for Phase 3 completion."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def full_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (errors * mask).sum() / mask.sum().clamp_min(1.0)


def completion_loss_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"full_bce": full_bce_loss(logits, target), "masked_bce": masked_bce_loss(logits, target, mask)}
