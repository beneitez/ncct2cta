"""Model factory. Models is a regressor that output a single *residual* channel
(CTA - NCCT in scaled space); no final activation.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

* unet3d    : 3D U-Net on anisotropic patches. Used on the cloud GPU.
"""
from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import BasicUNet, SegResNet


def build_model(arch: str = "unet3d", in_channels: int = 1, features=None) -> nn.Module:
    """`in_channels` = n_windows for volumetric models."""
    arch = arch.lower()
    if arch == "unet3d":
        return BasicUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=1,
            features=features or (16, 16, 32, 64, 128, 16),
        )
    raise ValueError(f"Unknown arch: {arch!r}")
