"""Losses for NCCT->CTA residual synthesis.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import vessel_weight_map

try:
    from monai.losses import SSIMLoss
    _HAS_SSIM = True
except Exception:  # pragma: no cover
    _HAS_SSIM = False


class SynthesisLoss(nn.Module):
    """Composite loss. Inputs are scaled tensors in [-1, 1].

    Parameters
    ----------
    spatial_dims : 3 for volumetric models.
    w_vessel     : loss weight multiplier inside enhanced vessels.
    lambda_ssim  : weight of the (1 - SSIM) structural term (0 disables it). (optional, not used)
    """

    def __init__(
        self, spatial_dims: int = 3, w_vessel: float = 20.0, w_brain: float = 1.0,
        w_bg: float = 0.1, lambda_ssim: float = 0.0,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.w_vessel, self.w_brain, self.w_bg = w_vessel, w_brain, w_bg
        self.lambda_ssim = lambda_ssim
        self.ssim = (
            SSIMLoss(spatial_dims=spatial_dims, data_range=2.0)
            if (lambda_ssim > 0 and _HAS_SSIM)
            else None
        )

    def forward(
        self, pred_res: torch.Tensor, ncct: torch.Tensor, cta: torch.Tensor,
        brain: torch.Tensor | None = None,
    ) -> dict:
        target_res = cta - ncct
        weight = vessel_weight_map(ncct, cta, brain, self.w_vessel, self.w_brain, self.w_bg)
        l1 = (weight * (pred_res - target_res).abs()).sum() / weight.sum().clamp_min(1.0)
        out = {"l1": l1, "loss": l1}
        if self.ssim is not None:
            recon = (ncct + pred_res).clamp(-1.0, 1.0)
            ssim = self.ssim(recon, cta)  # already 1 - SSIM
            out["ssim"] = ssim
            out["loss"] = l1 + self.lambda_ssim * ssim
        return out
