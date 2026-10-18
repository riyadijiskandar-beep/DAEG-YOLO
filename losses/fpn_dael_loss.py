"""
fpn_dael_loss.py — FPN-guided Detection-Aware Enhancement Loss

Truly detection-aware DAEL: uses YOLOv8's own P3 FPN features as a
spatial importance map to guide where DCENet9 should focus enhancement.

Standard DAEL (TV + global brightness) treats all pixels equally — it is
enhancement regularisation, NOT detection-aware.

FPN-DAEL closes the loop:
  enhanced → YOLOv8 backbone → FPN-P3 (detached) → importance map
  → region-weighted TV + brightness → gradient → DCENet9

DCENet9 is explicitly guided to enhance detection-critical regions
(where YOLOv8's P3 FPN is most active) rather than the full background.

Gradient flow (safe, no darkening equilibrium):
  L_fpndael → enhanced  ← gradient flows to DCENet9
  importance (P3 .detach()) ← NO gradient to YOLO
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNAwareDAEL(nn.Module):
    """
    FPN-guided Detection-Aware Enhancement Loss.

    Args:
        w_tv      : weight for region-aware TV smoothness   (default 0.1)
        w_bright  : weight for region-aware brightness push (default 1.0)
        margin    : minimum required brightness gain over x_low (default 0.05)
    """

    def __init__(
        self,
        w_tv: float    = 0.1,
        w_bright: float = 1.0,
        margin: float  = 0.05,
    ):
        super().__init__()
        self.w_tv    = w_tv
        self.w_bright = w_bright
        self.margin  = margin

    @staticmethod
    def build_importance(
        fpn_p3: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """
        Build spatial detection-importance map from P3 FPN features.

        Steps:
          1. Channel-mean of absolute activations  (B, 1, H/8, W/8)
          2. Per-image min-max normalisation        → [0, 1]
          3. Bilinear upsample to image resolution  (B, 1, H, W)
          4. .detach() — NO gradient back to YOLOv8

        High values = where YOLOv8 backbone is most active = likely objects.
        Low values  = background regions.

        Returns:
            (B, 1, H, W) float32 tensor in [0, 1], detached from graph.
        """
        imp = fpn_p3.detach().float().abs().mean(dim=1, keepdim=True)  # (B,1,h,w)

        B = imp.shape[0]
        flat = imp.view(B, -1)
        mn   = flat.min(dim=1)[0].view(B, 1, 1, 1)
        mx   = flat.max(dim=1)[0].view(B, 1, 1, 1)
        imp  = (imp - mn) / (mx - mn + 1e-8)          # per-image [0, 1]

        imp = F.interpolate(
            imp, size=target_hw, mode="bilinear", align_corners=False
        )
        return imp   # already detached (fpn_p3.detach() above)

    def forward(
        self,
        enhanced:       torch.Tensor,
        x_low:          torch.Tensor,
        fpn_p3:         torch.Tensor,
        ext_importance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            enhanced       : (B, 3, H, W)  DCENet9 output,        [0, 1]
            x_low          : (B, 3, H, W)  original dark image,   [0, 1]
            fpn_p3         : (B, C, H/8, W/8)  YOLOv8 P3 FPN features
                             (used only when ext_importance is None)
            ext_importance : (B, 1, H_low, W_low) optional pre-computed
                             importance in [0, 1]; from DAFM module (V20).
                             When provided, replaces the heuristic abs().mean()
                             computation. Must be detached already.
        Returns:
            Scalar loss tensor (with gradient w.r.t. enhanced / DCENet9).
        """
        if ext_importance is not None:
            # DAFM-predicted importance: already in [0,1] via Sigmoid; upsample to image res
            imp = F.interpolate(
                ext_importance.detach().float(),
                size=enhanced.shape[-2:], mode="bilinear", align_corners=False,
            )
        else:
            imp = self.build_importance(fpn_p3, enhanced.shape[-2:])
        # imp: (B, 1, H, W), [0,1], detached — no gradient back to YOLO

        # Weight: detection regions [1→2], background [1→1]
        w = 1.0 + imp   # (B, 1, H, W)

        # ── 1. Region-aware TV ─────────────────────────────────────────────────
        # Artifacts in detection regions are more harmful → penalise more there.
        tv_h = (enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :]).abs()   # (B,3,H-1,W)
        tv_w = (enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1]).abs()   # (B,3,H,W-1)
        # Align importance spatially to each TV diff dimension
        w_h = w[:, :, 1:, :]    # (B, 1, H-1, W)
        w_w = w[:, :, :, 1:]    # (B, 1, H, W-1)
        L_tv = (tv_h * w_h).mean() + (tv_w * w_w).mean()

        # ── 2. Region-aware brightness ─────────────────────────────────────────
        # Push enhanced to be brighter than x_low MORE in detection regions.
        # delta > 0 means enhanced is NOT bright enough yet → penalise.
        x_low_f = x_low.to(dtype=enhanced.dtype)
        delta   = x_low_f - enhanced + self.margin   # (B, 3, H, W)
        L_bright = F.relu(delta * w).mean()

        return self.w_tv * L_tv + self.w_bright * L_bright


class StandardDAEL(nn.Module):
    """
    Original enhancement regulariser (TV + global brightness).

    Used during warmup when FPN features are unstable, and as fallback
    when FPN capture fails. Kept separate to preserve the original baseline behaviour.
    """

    def __init__(
        self,
        w_tv: float    = 0.1,
        w_bright: float = 1.0,
        margin: float  = 0.05,
    ):
        super().__init__()
        self.w_tv    = w_tv
        self.w_bright = w_bright
        self.margin  = margin

    def forward(
        self,
        enhanced: torch.Tensor,
        x_low:    torch.Tensor,
    ) -> torch.Tensor:
        tv_h  = (enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :]).abs().mean()
        tv_w  = (enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1]).abs().mean()
        L_tv  = tv_h + tv_w

        delta    = x_low.to(dtype=enhanced.dtype).mean() - enhanced.mean() + self.margin
        L_bright = F.relu(delta)

        return self.w_tv * L_tv + self.w_bright * L_bright
