"""
Anticipatory Focal Loss for driver maneuver anticipation.

Combines:
  1. Focal Loss (Lin et al., 2017) -- down-weights easy majority class
     (normal_forward), focuses gradient on hard rare maneuvers.
  2. Temporal Anticipation Penalty (Jain et al., Brain4Cars, 2016) --
     w_i = clamp(exp(-ttm_i), min=min_w). Grows exponentially as the
     true maneuver approaches, forcing early lock-in.

Per-sample loss:  L_i = w_i * focal(logits_i, y_i)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnticipatoryFocalLoss(nn.Module):
    """
    Inputs
    ------
    logits           : (B, num_classes) raw class logits
    target           : (B,) long, class index
    time_to_maneuver : (B,) float, seconds (>= 0) until the physical
                       maneuver. For samples with target == normal_class_idx,
                       the TTM is overridden internally (normal_forward has
                       no genuine TTM), so the dataset may pass any value.
    """

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        min_anticipation_weight: float = 0.1,
        max_ttm_clip: float = 10.0,
        normal_class_idx: int = 0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.min_weight = min_anticipation_weight
        self.max_ttm_clip = max_ttm_clip
        self.normal_class_idx = normal_class_idx
        self.reduction = reduction

        if alpha is None:
            alpha = torch.tensor([0.1] + [1.0] * (num_classes - 1))
        assert alpha.numel() == num_classes, (
            f"alpha must have {num_classes} elements, got {alpha.numel()}"
        )
        self.register_buffer("alpha", alpha.float())

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        time_to_maneuver: torch.Tensor,
    ) -> torch.Tensor:
        # normal_forward has no genuine TTM; force to max so anticip weight
        # collapses to `min_anticipation_weight` for those samples.
        ttm = time_to_maneuver.to(logits.device).float()
        normal_mask = (target == self.normal_class_idx)
        if normal_mask.any():
            ttm = ttm.masked_fill(normal_mask, self.max_ttm_clip)
        ttm = ttm.clamp(0.0, self.max_ttm_clip)

        # Focal loss per sample
        log_p = F.log_softmax(logits, dim=-1)                      # (B, C)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)   # (B,)
        pt = log_pt.exp()
        alpha_t = self.alpha[target]                               # (B,)
        focal = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt     # (B,)

        # Temporal anticipation weight
        anticip_w = torch.exp(-ttm).clamp(min=self.min_weight)     # (B,)

        loss = focal * anticip_w                                    # (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def compute_alpha_from_counts(
        class_counts,
        scheme: str = "sqrt_inv_freq",
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Derive per-class alpha from training-set counts.

        scheme:
            "inv_freq"      -> alpha_c proportional to 1 / freq_c   (aggressive)
            "sqrt_inv_freq" -> proportional to 1 / sqrt(freq_c)     (gentler, recommended)

        Normalized to mean 1.0 for gradient-scale stability, then scaled.
        """
        counts = torch.as_tensor(class_counts, dtype=torch.float)
        C = counts.numel()
        N = counts.sum()
        inv = N / (C * counts.clamp(min=1.0))
        alpha = inv.sqrt() if scheme == "sqrt_inv_freq" else inv
        alpha = alpha / alpha.mean()
        return (alpha * scale).float()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Linear warmup from ~0 -> base_lr over `num_warmup_steps`, then cosine
    decay to `base_lr * min_lr_ratio` over the remaining steps.

    STEP PER OPTIMIZER STEP (not per epoch).

    Recipe (CaTFormer-style for imbalanced driving data):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        total_steps  = len(train_loader) * num_epochs
        warmup_steps = int(0.05 * total_steps)  # 5%
        scheduler = build_warmup_cosine_scheduler(
            optimizer, warmup_steps, total_steps, min_lr_ratio=0.01,
        )
    """
    from torch.optim.lr_scheduler import (
        LambdaLR, CosineAnnealingLR, SequentialLR,
    )

    base_lr = optimizer.param_groups[0]["lr"]

    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda step: float(step + 1) / float(max(1, num_warmup_steps)),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_training_steps - num_warmup_steps),
        eta_min=base_lr * min_lr_ratio,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[num_warmup_steps],
    )
