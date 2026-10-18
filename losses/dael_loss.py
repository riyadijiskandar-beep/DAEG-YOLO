"""
DAELoss — Detection-Aware Enhancement Loss.

No-reference quality loss designed for ExDark, which has no clean paired
reference images. Combines:

    L_DAEL = w_tv * L_TV(enhanced)
           + w_bright * L_brightness(enhanced, x_low)

    L_TV        : anisotropic total variation (spatial smoothness)
    L_brightness: ReLU(mean(x_low) - mean(enhanced) + margin)
                  → zero when enhanced is sufficiently brighter than input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TVLoss(nn.Module):
    """Anisotropic total variation for spatial smoothness."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_diff = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        w_diff = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return h_diff + w_diff


class BrightnessLoss(nn.Module):
    """
    Brightness enhancement loss.

    Penalises the case where the enhanced image is not brighter than the
    low-light input by at least `margin` (in mean pixel value).
    """

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, enhanced: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        delta = x_low.mean() - enhanced.mean() + self.margin
        return F.relu(delta)


class DAELoss(nn.Module):
    """
    Detection-Aware Enhancement Loss for ExDark (no-reference).

    Args:
        w_tv    : Weight for total variation loss   (default 0.1).
        w_bright: Weight for brightness loss        (default 1.0).
        margin  : Minimum required brightness gain  (default 0.05).
    """

    def __init__(self, w_tv: float = 0.1, w_bright: float = 1.0, margin: float = 0.05):
        super().__init__()
        self.tv     = TVLoss()
        self.bright = BrightnessLoss(margin=margin)
        self.w_tv     = w_tv
        self.w_bright = w_bright

    def forward(self, enhanced: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced : (B, 3, H, W) enhanced image in [0, 1]
            x_low    : (B, 3, H, W) low-light input in [0, 1]
        Returns:
            scalar loss
        """
        l_tv    = self.tv(enhanced)
        l_bright = self.bright(enhanced, x_low)
        return self.w_tv * l_tv + self.w_bright * l_bright
