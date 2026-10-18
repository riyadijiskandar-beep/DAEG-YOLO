"""
train_daeg_yolo_llvip.py — DAEG-YOLO on LLVIP Dataset
==========================================================
Identical architecture to train_daeg_yolo.py (ExDark).
Dataset: LLVIP_Official — 15488 visible-light nighttime images, 1 class (person).
Per-lighting eval disabled (LLVIP does not have lighting condition labels).

Components:
  DCENet9   : 9-iteration zero-reference curve estimation (~0.081M params)
  DAFMC2f   : DAFM@Layer15 — feature modulation + learned importance export
  FPN-DAEL  : FPN-guided detection-aware enhancement loss (warmup=5ep)
  CondRecon : Conditional reconstruction loss (bright-image identity constraint)

Usage:
  python train_daeg_yolo_llvip.py \\
      --model yolov8s.pt \\
      --yaml models/yolov8s_daeg_yolo.yaml \\
      --data datasets/LLVIP_Official/LLVIP.yaml \\
      --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \\
      --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \\
      --patience 50 --lambda-dael 0.01 --w-tv 0.1 --w-bright 1.0 \\
      --dael-mode fpn --dael-warmup-epochs 5 --fpn-layer 15 \\
      --dafm-mode full \\
      --lambda-recon 1.0 --recon-threshold 0.10 --recon-k 50.0 \\
      --workers 0 --device 0 --amp --seed 42 --project results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from copy import copy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path = [p for p in sys.path if p != str(ROOT)]
_ul_user = [p for p in sys.path if "AppData\\Roaming" in p]
_ul_rest  = [p for p in sys.path if "AppData\\Roaming" not in p]
sys.path   = _ul_rest + _ul_user
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml as _yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cv2 as _cv2_pre
    _REAL_CV2_IMREAD = _cv2_pre.imread
except ImportError:
    _REAL_CV2_IMREAD = None

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C2f, DAFMC2f, DAFMBlock  # noqa: F401
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import strip_optimizer

from losses.fpn_dael_loss import FPNAwareDAEL, StandardDAEL


# ══════════════════════════════════════════════════════════════════════════════
#  DCENet9 
# ══════════════════════════════════════════════════════════════════════════════

class DCENet9(nn.Module):
    """Zero-reference curve estimation, 9 iterations, ~0.081M params."""

    def __init__(self, n_iterations: int = 9, n_filters: int = 32):
        super().__init__()
        self.n_iterations = n_iterations
        nf = n_filters
        self.conv1 = nn.Conv2d(3,      nf,               3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nf,     nf,               3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nf,     nf,               3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nf,     nf,               3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nf * 2, nf,               3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(nf * 2, nf,               3, padding=1, bias=True)
        self.conv7 = nn.Conv2d(nf * 2, n_iterations * 3, 3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)
        self.tanh  = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_low: torch.Tensor):
        x1 = self.relu(self.conv1(x_low))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x4, x3], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x5, x2], dim=1)))
        alpha = self.tanh(self.conv7(torch.cat([x6, x1], dim=1)))
        B, _, H, W = x_low.shape
        alpha = alpha.view(B, self.n_iterations, 3, H, W)
        enhanced = x_low
        for i in range(self.n_iterations):
            enhanced = enhanced + alpha[:, i] * enhanced * (1.0 - enhanced)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced, {}


# ══════════════════════════════════════════════════════════════════════════════
#  FPN capture + epoch state
# ══════════════════════════════════════════════════════════════════════════════

class _FPNCapture:
    def __init__(self, layer_idx: int = 15):
        self.layer_idx      = layer_idx
        self.p3:              torch.Tensor | None = None
        self.dafm_importance: torch.Tensor | None = None
        self._handle = None

    def _hook_fn(self, module, inp, out):
        self.p3 = out
        if hasattr(module, "dafm") and module.dafm._importance is not None:
            self.dafm_importance = module.dafm._importance
        else:
            self.dafm_importance = None

    def attach(self, det_model: DetectionModel) -> None:
        if self._handle is not None:
            self._handle.remove()
        layer = det_model.model[self.layer_idx]
        self._handle = layer.register_forward_hook(self._hook_fn)
        LOGGER.info(
            f"[DAEG FPN] Capture hook attached to layer {self.layer_idx} "
            f"({type(layer).__name__}) — P3+importance → FPN-DAEL"
        )

    def detach_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        self.p3              = None
        self.dafm_importance = None


class _TrainState:
    def __init__(self):
        self.epoch: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  CondRecon stats accumulator
# ══════════════════════════════════════════════════════════════════════════════

class _CondReconStats:
    """Per-epoch accumulator for CondRecon diagnostics."""
    def __init__(self):
        self.recon_loss_sum: float = 0.0
        self.bright_wt_sum:  float = 0.0
        self.n_batches:      int   = 0

    def update(self, recon_loss_val: float, bright_wt_mean: float):
        self.recon_loss_sum += recon_loss_val
        self.bright_wt_sum  += bright_wt_mean
        self.n_batches      += 1

    def summary(self) -> tuple[float, float]:
        if self.n_batches == 0:
            return 0.0, 0.0
        return self.recon_loss_sum / self.n_batches, self.bright_wt_sum / self.n_batches

    def reset(self):
        self.recon_loss_sum = 0.0
        self.bright_wt_sum  = 0.0
        self.n_batches      = 0


# ══════════════════════════════════════════════════════════════════════════════
#  Patched loss with DAEL + CondRecon
# ══════════════════════════════════════════════════════════════════════════════

class _PatchedLoss:
    """YOLO loss + FPN-DAEL + Conditional Reconstruction Loss.

    CondRecon loss:
      L_recon = mean_i [ w_i * L1(enhanced_i, x_low_i) ]
      w_i     = sigmoid(recon_k * (brightness_i − recon_threshold))
      Gradient flows only through enhanced → DCENet9 weights.
      For bright images (w≈1): DCENet9 pushed toward identity output.
      For dark images (w≈0): DCENet9 freely enhances.
    """

    def __init__(
        self,
        original_loss,
        dael_mode:       str,
        dafm_mode:       str,
        fpn_dael_fn:     FPNAwareDAEL,
        std_dael_fn:     StandardDAEL,
        capture:         _FPNCapture,
        state:           _TrainState,
        lam:             float,
        warmup_epochs:   int,
        lambda_recon:    float,
        recon_threshold: float,
        recon_k:         float,
        recon_stats:     _CondReconStats,
    ):
        self._original        = original_loss
        self._dael_mode       = dael_mode
        self._dafm_mode       = dafm_mode
        self._fpn_dael        = fpn_dael_fn
        self._std_dael        = std_dael_fn
        self._capture         = capture
        self._state           = state
        self._lam             = lam
        self._warmup_epochs   = warmup_epochs
        self._lambda_recon    = lambda_recon
        self._recon_threshold = recon_threshold
        self._recon_k         = recon_k
        self._recon_stats     = recon_stats

    def __call__(self, batch, preds=None):
        det_loss, det_items = self._original(batch, preds)

        if "x_low" not in batch:
            return det_loss, det_items

        enhanced = batch["img"]
        x_low    = batch["x_low"]

        # ── FPN-DAEL ──────────────────────────────────────────────────────────
        if self._lam > 0.0:
            if self._dael_mode == "fpn":
                p3          = self._capture.p3
                past_warmup = self._state.epoch >= self._warmup_epochs
                p3_valid    = p3 is not None and p3.size(0) == enhanced.size(0)
                if past_warmup and p3_valid:
                    ext_imp = None
                    if self._dafm_mode == "full":
                        ext_imp = self._capture.dafm_importance
                        if ext_imp is not None and ext_imp.size(0) != enhanced.size(0):
                            ext_imp = None
                    dael_val = self._fpn_dael(enhanced, x_low, p3, ext_importance=ext_imp)
                else:
                    dael_val = self._std_dael(enhanced, x_low)
            else:
                dael_val = self._std_dael(enhanced, x_low)
            det_loss = det_loss + self._lam * dael_val

        # ── Conditional Reconstruction Loss ───────────────────────────────────
        if self._lambda_recon > 0.0:
            with torch.no_grad():
                brightness   = x_low.mean(dim=[1, 2, 3])            # (B,)
                bright_weight = torch.sigmoid(
                    self._recon_k * (brightness - self._recon_threshold)
                )                                                     # (B,) in [0,1]

            # L1 per image, then weight by brightness mask
            # Gradient flows: recon_loss → enhanced → DCENet9 weights only
            per_img_l1 = F.l1_loss(enhanced, x_low.detach(),
                                   reduction="none").mean(dim=[1, 2, 3])  # (B,)
            recon_loss = (bright_weight * per_img_l1).mean()
            det_loss   = det_loss + self._lambda_recon * recon_loss

            # Accumulate diagnostics (detached scalars)
            self._recon_stats.update(
                recon_loss_val=float(recon_loss.detach()),
                bright_wt_mean=float(bright_weight.mean().detach()),
            )

        return det_loss, det_items


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

LIGHT_NAMES = {
    1: "Low",        2: "Ambient",     3: "Object",   4: "Single",
    5: "Weak",       6: "Strong",      7: "Window",   8: "Shadow",
    9: "Twilight",  10: "Fluorescent",
}
ANNNO_TXT    = ROOT / "ExDark_Annno" / "imageclasslist.txt"
DAFM_LAYER   = 15
DEFAULT_YAML = str(ROOT / "models" / "yolov8s_daeg_yolo.yaml")


# ══════════════════════════════════════════════════════════════════════════════
#  Validator
# ══════════════════════════════════════════════════════════════════════════════

class DAFMValidator(DetectionValidator):
    def __init__(self, *args, enhancer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._enhancer = enhancer

    def preprocess(self, batch: dict) -> dict:
        batch = super().preprocess(batch)
        if self._enhancer is not None:
            x_low = batch["img"]
            with torch.no_grad():
                enhanced, _ = self._enhancer(x_low.float())
            batch["img"] = enhanced.to(x_low.dtype)
        return batch


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class DAFMTrainer(DetectionTrainer):
    """DAEG-YOLO: DCENet9 + DAFMC2f@L15 + FPN-DAEL + CondRecon loss."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
        enhancer:        DCENet9 | None      = None,
        yaml_cfg:        str | None          = None,
        fpn_dael_fn:     FPNAwareDAEL | None = None,
        std_dael_fn:     StandardDAEL | None = None,
        lambda_dael:     float = 0.01,
        dael_mode:       str   = "fpn",
        dafm_mode:       str   = "full",
        warmup_epochs:   int   = 5,
        fpn_layer:       int   = 15,
        lr_dce_factor:   float = 1.0,
        lambda_recon:    float = 1.0,
        recon_threshold: float = 0.10,
        recon_k:         float = 50.0,
    ):
        self._enhancer        = enhancer
        self._yaml_cfg        = yaml_cfg or DEFAULT_YAML
        self._fpn_dael_fn     = fpn_dael_fn if fpn_dael_fn is not None else FPNAwareDAEL()
        self._std_dael_fn     = std_dael_fn if std_dael_fn is not None else StandardDAEL()
        self._lambda_dael     = lambda_dael
        self._dael_mode       = dael_mode
        self._dafm_mode       = dafm_mode
        self._warmup_epochs   = warmup_epochs
        self._fpn_layer       = fpn_layer
        self._lr_dce_factor   = lr_dce_factor
        self._lambda_recon    = lambda_recon
        self._recon_threshold = recon_threshold
        self._recon_k         = recon_k
        self._capture         = _FPNCapture(fpn_layer)
        self._state           = _TrainState()
        self._recon_stats     = _CondReconStats()
        self._recon_epoch_logged: int = -1
        self._best_map50:     float = 0.0
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True) -> DetectionModel:
        with open(self._yaml_cfg, encoding="utf-8") as f:
            d = _yaml.safe_load(f)
        d["nc"] = self.data["nc"]

        if "scale" not in d:
            _model_stem = Path(str(self.args.model)).stem
            for _sc in d.get("scales", {}):
                if _model_stem.endswith(_sc):
                    d["scale"] = _sc
                    break

        model = DetectionModel(
            d,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
        )
        model.enhancer = self._enhancer

        if weights:
            model.load(weights)

        layer15 = model.model[DAFM_LAYER]
        if not isinstance(layer15, DAFMC2f):
            raise TypeError(
                f"Expected DAFMC2f at layer {DAFM_LAYER}, got {type(layer15).__name__}. "
                "Ensure yolov8s_daeg_yolo.yaml declares DAFMC2f at layer 15."
            )

        if self._dafm_mode == "none":
            # True no-DAFM ablation: replace DAFMC2f with standard C2f.
            # Copy all C2f weights (cv1/cv2/m.*); discard dafm.* keys.
            old = layer15
            c2f = C2f(
                old.cv1.conv.in_channels,
                old.cv2.conv.out_channels,
                len(old.m),
                shortcut=old.m[0].add if len(old.m) > 0 else False,
            ).to(next(old.parameters()).device)
            c2f_sd = {k: v for k, v in old.state_dict().items()
                      if not k.startswith("dafm.")}
            c2f.load_state_dict(c2f_sd, strict=True)
            for attr in ("f", "i", "type", "np"):
                if hasattr(old, attr):
                    setattr(c2f, attr, getattr(old, attr))
            model.model[DAFM_LAYER] = c2f
            LOGGER.info(
                f"[DAEG] dafm_mode=none: DAFMC2f → C2f at layer {DAFM_LAYER} "
                f"(feature modulation disabled, FPN-DAEL uses heuristic importance)"
            )
        else:
            dafm_params = sum(p.numel() for p in layer15.dafm.parameters())
            c2_actual   = layer15.cv2.conv.out_channels
            LOGGER.info(
                f"[DAEG] DAFMC2f @ Layer {DAFM_LAYER} ({c2_actual}ch), "
                f"dafm_mode={self._dafm_mode}, new DAFM params: {dafm_params}"
            )

        self._capture.attach(model)

        model.loss = _PatchedLoss(
            original_loss    = model.loss,
            dael_mode        = self._dael_mode,
            dafm_mode        = self._dafm_mode,
            fpn_dael_fn      = self._fpn_dael_fn,
            std_dael_fn      = self._std_dael_fn,
            capture          = self._capture,
            state            = self._state,
            lam              = self._lambda_dael,
            warmup_epochs    = self._warmup_epochs,
            lambda_recon     = self._lambda_recon,
            recon_threshold  = self._recon_threshold,
            recon_k          = self._recon_k,
            recon_stats      = self._recon_stats,
        )
        return model

    def get_validator(self) -> DAFMValidator:
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DAFMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            enhancer=self._enhancer,
        )

    def _setup_train(self, *args, **kwargs):
        import sys as _sys
        _tm = _sys.modules.get("ultralytics.engine.trainer")
        _orig_check_amp = getattr(_tm, "check_amp", None) if _tm else None
        if _orig_check_amp is not None:
            _tm.check_amp = lambda m: True
        try:
            super()._setup_train(*args, **kwargs)
        finally:
            if _tm is not None and _orig_check_amp is not None:
                _tm.check_amp = _orig_check_amp
        if abs(self._lr_dce_factor - 1.0) > 1e-6:
            self._rebuild_differential_optimizer()

        def _w(b):
            return torch.sigmoid(
                torch.tensor(self._recon_k * (b - self._recon_threshold))
            ).item()

        LOGGER.info(
            f"[DAEG-YOLO] dafm_mode={self._dafm_mode}, dael_mode={self._dael_mode}, "
            f"warmup={self._warmup_epochs}ep, lambda_dael={self._lambda_dael}"
        )
        LOGGER.info(
            f"[DAEG-YOLO] lambda={self._lambda_recon}, "
            f"threshold={self._recon_threshold}, k={self._recon_k}"
        )


    def _rebuild_differential_optimizer(self):
        from torch import optim
        dce_ids = {
            id(p) for p in self.model.enhancer.parameters()
            if p.requires_grad and p.dtype.is_floating_point
        }
        g_other, g_dce = [], []
        for p in self.model.parameters():
            if not (p.requires_grad and p.dtype.is_floating_point):
                continue
            (g_dce if id(p) in dce_ids else g_other).append(p)
        base_lr = self.args.lr0
        dce_lr  = base_lr * self._lr_dce_factor
        groups = [{"params": g_other, "lr": base_lr, "initial_lr": base_lr,
                   "weight_decay": self.args.weight_decay}]
        if g_dce:
            groups.append({"params": g_dce, "lr": dce_lr, "initial_lr": dce_lr,
                           "weight_decay": self.args.weight_decay})
        opt_name = self.args.optimizer.upper()
        if opt_name == "SGD":
            self.optimizer = optim.SGD(groups, momentum=self.args.momentum, nesterov=True)
        elif opt_name == "ADAMW":
            self.optimizer = optim.AdamW(groups)
        else:
            self.optimizer = optim.Adam(groups)
        lrf = self.args.lrf
        lf  = lambda x: max(1 - x / self.epochs, 0) * (1.0 - lrf) + lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        self.scheduler.last_epoch = -1

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        x_low = batch["img"]

        curr_epoch = self.epoch
        # Log CondRecon stats once per epoch (at epoch boundary)
        if curr_epoch != self._recon_epoch_logged and self._recon_epoch_logged >= 0:
            recon_avg, wt_avg = self._recon_stats.summary()
            LOGGER.info(
                f"[DAEG-YOLO] Epoch {self._recon_epoch_logged}: "
                f"mean_recon_loss={recon_avg:.5f}  mean_bright_weight={wt_avg:.4f}"
            )
            self._recon_stats.reset()
        self._recon_epoch_logged = curr_epoch

        self._state.epoch = curr_epoch
        self._capture.clear()

        with torch.set_grad_enabled(self.model.training):
            enhanced, _ = self.model.enhancer(x_low)

        batch["img"]   = enhanced
        batch["x_low"] = x_low.detach()
        return batch

    def validate(self):
        result = super().validate()
        if isinstance(result, tuple) and len(result) == 2:
            metrics, _ = result
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            if map50 > self._best_map50:
                self._best_map50 = map50
            self.best_fitness = self._best_map50
            return metrics, map50
        if isinstance(result, dict):
            map50 = result.get("metrics/mAP50(B)", result.get("fitness", 0.0))
            if map50 > self._best_map50:
                self._best_map50 = map50
            self.best_fitness = self._best_map50
            result["fitness"] = map50
        return result

    def plot_training_samples(self, batch, ni):
        super().plot_training_samples({**batch, "img": batch["img"].detach()}, ni)

    def final_eval(self):
        for f in (self.last, self.best):
            if f.exists():
                strip_optimizer(f)
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.validator._enhancer  = self._enhancer
                    ckpt = torch.load(str(f), map_location=self.device,
                                      weights_only=False)
                    loaded_model = (ckpt.get("ema") or ckpt["model"])\
                                   .float().to(self.device).eval()
                    self.metrics = self.validator(model=loaded_model)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DAEG-YOLO: DCENet9 + DAFM + FPN-DAEL + CondRecon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  default="yolov8s.pt")
    p.add_argument("--yaml",   default=DEFAULT_YAML)
    p.add_argument("--data",   default="datasets/LLVIP_Official/LLVIP.yaml")

    p.add_argument("--dce-n-iter",    type=int, default=9)
    p.add_argument("--dce-n-filters", type=int, default=32)

    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch",         type=int,   default=12)
    p.add_argument("--imgsz",         type=int,   default=640)
    p.add_argument("--patience",      type=int,   default=50)
    p.add_argument("--optimizer",     default="SGD")
    p.add_argument("--lr0",           type=float, default=0.001)
    p.add_argument("--lrf",           type=float, default=0.01)
    p.add_argument("--lr-dce-factor", type=float, default=1.0)
    p.add_argument("--momentum",      type=float, default=0.937)
    p.add_argument("--weight-decay",  type=float, default=0.0005)

    p.add_argument("--lambda-dael",        type=float, default=0.01)
    p.add_argument("--w-tv",               type=float, default=0.1)
    p.add_argument("--w-bright",           type=float, default=1.0)
    p.add_argument("--dael-mode",          default="fpn",
                   choices=["standard", "fpn"])
    p.add_argument("--dael-warmup-epochs", type=int, default=5)
    p.add_argument("--fpn-layer",          type=int, default=15)
    p.add_argument("--dafm-mode",          default="full",
                   choices=["full", "heuristic", "none"])

    # ── CondRecon ──────────────────────────────────────────────────────────────
    p.add_argument("--lambda-recon",    type=float, default=1.0,
                   help=(
                       "Weight for conditional reconstruction loss. "
                       "0.0 = disabled (reproduces V20). "
                       "Default 1.0. Try 0.5/1.0/2.0."
                   ))
    p.add_argument("--recon-threshold", type=float, default=0.10,
                   help=(
                       "Brightness pivot for soft mask. Images with mean brightness "
                       "above this get penalized for deviation from input. "
                   ))
    p.add_argument("--recon-k",         type=float, default=50.0,
                   help=(
                       "Sigmoid sharpness for bright_weight mask. "
                       "Higher = sharper transition at threshold. Default 50."
                   ))

    p.add_argument("--device",  default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--amp",     action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--project", default="results")
    p.add_argument("--name",    default="")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def fix_yaml_path(yaml_path: str) -> str:
    import re
    yaml_file = Path(yaml_path).resolve()
    if not yaml_file.exists():
        return str(yaml_file)
    dataset_root = str(yaml_file.parent).replace("\\", "/")
    text = yaml_file.read_text(encoding="utf-8")
    text = re.sub(r"^path\s*:.*$", f"path: {dataset_root}", text, flags=re.MULTILINE)
    yaml_file.write_text(text, encoding="utf-8")
    return str(yaml_file)


def build_exp_name(args) -> str:
    if args.name:
        return args.name
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem         = Path(args.model).stem
    dataset_name = Path(args.data).stem
    dael_tag     = "FPNDAEL" if args.dael_mode == "fpn" else "StdDAEL"
    dafm_tag     = f"DAFM_{args.dafm_mode}" if args.dafm_mode != "none" else "noDAFM"
    recon_tag    = (
        f"CondRecon_thr{args.recon_threshold:.2f}_lam{args.lambda_recon:.1f}"
        if args.lambda_recon > 0.0 else "noCondRecon"
    )
    return (
        f"DAEG_YOLO_DCE{args.dce_n_iter}_{recon_tag}_"
        f"{dafm_tag}_{dael_tag}_{stem}_{dataset_name}_{ts}"
    )


def parse_imageclasslist(annno_txt: Path) -> dict:
    mapping = {}
    with open(annno_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3 or parts[0].startswith("Name"):
                continue
            stem  = Path(parts[0]).stem.lower()
            light = int(parts[2])
            mapping[stem] = light
    return mapping


def save_enhanced_samples(
    enhancer: DCENet9,
    loader,
    out_dir: Path,
    device: torch.device,
    n_samples: int = 8,
    dafm_mode: str = "full",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    enhancer.eval()
    collected_low, collected_enh = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].float() / 255.0
            imgs = imgs.to(device)
            enhanced, _ = enhancer(imgs)
            for i in range(imgs.shape[0]):
                collected_low.append(imgs[i].cpu())
                collected_enh.append(enhanced[i].cpu())
                if len(collected_low) >= n_samples:
                    break
            if len(collected_low) >= n_samples:
                break
    n = min(n_samples, len(collected_low))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 2.8))
    if n == 1:
        axes = axes[None, :]

    def to_np(t):
        return t.permute(1, 2, 0).numpy().clip(0, 1)

    for i in range(n):
        axes[i, 0].imshow(to_np(collected_low[i]))
        axes[i, 0].set_title("Low-Light Input", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(to_np(collected_enh[i]))
        axes[i, 1].set_title("Enhanced (DCENet9)", fontsize=8)
        axes[i, 1].axis("off")
    plt.suptitle(
        f"Enhancement — DAEG-YOLO (CondRecon, DAFM-{dafm_mode})",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_path = out_dir / "enhanced_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Qualitative] Saved {n} samples -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Full test evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _load_best(trainer) -> "torch.nn.Module":
    ckpt   = torch.load(str(trainer.best), map_location=trainer.device,
                        weights_only=False)
    loaded = (ckpt.get("ema") or ckpt["model"]).float().to(trainer.device).eval()
    return loaded


def run_full_test_eval(
    trainer,
    enhancer: DCENet9,
    data_yaml: Path,
    out_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
) -> tuple:
    from ultralytics.data.utils import check_det_dataset

    print("\n" + "=" * 60)
    print("  FULL TEST SET EVALUATION")
    print("=" * 60)

    try:
        check_data = check_det_dataset(str(data_yaml))

        val_args         = copy(trainer.args)
        val_args.plots   = False
        val_args.verbose = False
        val_args.save    = False

        validator = DAFMValidator(
            dataloader=None,
            save_dir=out_dir / "_test_eval_tmp",
            args=val_args,
            enhancer=enhancer,
        )
        validator.data       = check_data
        validator.dataloader = validator.get_dataloader(
            check_data.get("test"), val_args.batch
        )

        loaded_model = _load_best(trainer)
        validator(model=loaded_model)

        rd = validator.metrics.results_dict
        results = {
            "timestamp": timestamp,
            "split":     "test",
            "mAP50":     round(float(rd.get("metrics/mAP50(B)",     0)), 4),
            "mAP50_95":  round(float(rd.get("metrics/mAP50-95(B)",  0)), 4),
            "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
            "recall":    round(float(rd.get("metrics/recall(B)",    0)), 4),
        }

        print(f"  mAP@0.5      : {results['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95 : {results['mAP50_95']:.4f}")
        print(f"  Precision    : {results['precision']:.4f}")
        print(f"  Recall       : {results['recall']:.4f}")

        json_path = out_dir / f"test_results_{timestamp}.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  [RESULTS] Saved: {json_path.name}")

        per_class_rows = []
        try:
            box         = validator.metrics.box
            names       = validator.names
            cls_indices = box.ap_class_index
            ap50_arr    = box.ap[:, 0]
            ap5095_arr  = box.ap.mean(axis=1)
            for i, ci in enumerate(cls_indices):
                per_class_rows.append({
                    "class_id":   int(ci),
                    "class_name": names.get(int(ci), f"cls{int(ci)}"),
                    "ap50":       round(float(ap50_arr[i]),   4),
                    "ap50_95":    round(float(ap5095_arr[i]), 4),
                })
            if per_class_rows:
                csv_pc = out_dir / f"per_class_{timestamp}.csv"
                with open(csv_pc, "w", newline="", encoding="utf-8") as _f:
                    _w = csv.DictWriter(_f, fieldnames=["class_id", "class_name", "ap50", "ap50_95"])
                    _w.writeheader()
                    _w.writerows(per_class_rows)
                print(f"  [RESULTS] Per-class CSV: {csv_pc.name}")
                for row in per_class_rows:
                    print(f"    {row['class_name']:<12s}  AP50={row['ap50']:.4f}  AP50-95={row['ap50_95']:.4f}")
        except Exception as _exc:
            print(f"  [WARN] Per-class AP extraction failed: {_exc}")

        shutil.rmtree(out_dir / "_test_eval_tmp", ignore_errors=True)
        return results, per_class_rows

    except Exception as exc:
        print(f"[WARN] Full test evaluation failed: {exc}")
        return {}, []


# ══════════════════════════════════════════════════════════════════════════════
#  Per-lighting evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_by_lighting(
    trainer,
    enhancer: DCENet9,
    data_yaml: Path,
    test_img_dir: Path,
    annno_txt: Path,
    out_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
) -> list:
    import yaml as _yaml2
    from ultralytics.data.utils import check_det_dataset

    if not annno_txt.exists():
        print(f"  [SKIP] imageclasslist.txt not found: {annno_txt}")
        return []

    img_light   = parse_imageclasslist(annno_txt)
    suffixes    = {".jpg", ".jpeg", ".png", ".bmp"}
    test_images = [p for p in sorted(test_img_dir.iterdir())
                   if p.suffix.lower() in suffixes]

    groups: dict = {}
    n_unknown = 0
    for img_path in test_images:
        code = img_light.get(img_path.stem.lower())
        if code is None:
            n_unknown += 1
        else:
            groups.setdefault(code, []).append(img_path)
    if n_unknown:
        print(f"  [WARN] {n_unknown} test images not matched in imageclasslist.txt")

    with open(data_yaml, encoding="utf-8") as f:
        base_cfg = _yaml2.safe_load(f)

    dataset_root = str(data_yaml.resolve().parent).replace("\\", "/")
    tmp_dir      = out_dir / "_tmp_lighting"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for code in sorted(groups.keys()):
        img_paths  = groups[code]
        light_name = LIGHT_NAMES.get(code, f"Cond_{code}")
        n          = len(img_paths)
        print(f"  [{code:2d}] {light_name:12s} ({n:4d} images) ...", end="", flush=True)

        txt_path = tmp_dir / f"light_{code}.txt"
        txt_path.write_text(
            "\n".join(str(p.resolve()).replace("\\", "/") for p in img_paths),
            encoding="utf-8",
        )
        tmp_cfg = {
            "path":  dataset_root,
            "train": str(txt_path.resolve()).replace("\\", "/"),
            "val":   str(txt_path.resolve()).replace("\\", "/"),
            "test":  str(txt_path.resolve()).replace("\\", "/"),
            "nc":    base_cfg["nc"],
            "names": base_cfg["names"],
        }
        tmp_yaml = tmp_dir / f"light_{code}.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            import yaml as _yaml3
            _yaml3.dump(tmp_cfg, f, allow_unicode=True, default_flow_style=False)

        try:
            check_data = check_det_dataset(str(tmp_yaml))
            val_args         = copy(trainer.args)
            val_args.plots   = False
            val_args.verbose = False
            val_args.save    = False

            validator = DAFMValidator(
                dataloader=None,
                save_dir=tmp_dir / f"val_{code}",
                args=val_args,
                enhancer=enhancer,
            )
            validator.data       = check_data
            validator.dataloader = validator.get_dataloader(
                check_data.get("test"), val_args.batch
            )

            loaded_model = _load_best(trainer)
            validator(model=loaded_model)

            rd      = validator.metrics.results_dict
            map50   = round(float(rd.get("metrics/mAP50(B)",     0)), 4)
            map5095 = round(float(rd.get("metrics/mAP50-95(B)",  0)), 4)
            prec    = round(float(rd.get("metrics/precision(B)", 0)), 4)
            rec     = round(float(rd.get("metrics/recall(B)",    0)), 4)
            print(f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            map50 = map5095 = prec = rec = 0.0

        rows.append({
            "light_code": code, "light_name": light_name, "n_images": n,
            "mAP50": map50, "mAP50_95": map5095,
            "precision": prec, "recall": rec,
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)

    csv_path = out_dir / f"per_lighting_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "light_code", "light_name", "n_images",
            "mAP50", "mAP50_95", "precision", "recall",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [METRICS] Per-lighting CSV: {csv_path.name}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Computational cost measurement
# ══════════════════════════════════════════════════════════════════════════════

def measure_compute(
    model,
    enhancer: DCENet9,
    device: torch.device,
    imgsz: int = 640,
    n_warmup: int = 10,
    n_measure: int = 50,
) -> dict:
    result = {"yolo_gflops": 0.0, "dce_gflops": 0.0, "total_gflops": 0.0,
              "fps": 0.0, "latency_ms": 0.0}
    try:
        from thop import profile as _thop_profile
        _dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        _macs, _ = _thop_profile(enhancer, inputs=(_dummy,), verbose=False)
        result["dce_gflops"] = _macs * 2 / 1e9
    except Exception as _e:
        print(f"  [INFO] thop unavailable ({_e}); skipping DCENet9 GFLOPs")
    try:
        _info = model.info(verbose=False, imgsz=imgsz)
        result["yolo_gflops"] = float(_info[3])
    except Exception as _e:
        print(f"  [INFO] model.info() failed ({_e}); skipping YOLO GFLOPs")
    result["total_gflops"] = result["yolo_gflops"] + result["dce_gflops"]
    try:
        model.eval()
        enhancer.eval()
        _dark = torch.rand(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _enh, _ = enhancer(_dark)
                _ = model(_enh)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            for _ in range(n_measure):
                _enh, _ = enhancer(_dark)
                _ = model(_enh)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _elapsed = time.perf_counter() - _t0
        result["fps"]        = n_measure / _elapsed
        result["latency_ms"] = _elapsed / n_measure * 1000.0
    except Exception as _e:
        print(f"  [WARN] FPS measurement failed: {_e}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Report writer
# ══════════════════════════════════════════════════════════════════════════════

def write_report(
    out_dir: Path,
    exp_name: str,
    args: argparse.Namespace,
    metrics: dict,
    lighting_rows: list | None = None,
    timing: dict | None = None,
    compute_cost: dict | None = None,
    test_results: dict | None = None,
    per_class_rows: list | None = None,
):
    report_path = out_dir / "report_daeg_yolo.txt"

    def fmt(key):
        v = metrics.get(key)
        return f"{v:.5f}" if isinstance(v, float) else str(v) if v is not None else "N/A"

    dael_str = (
        f"FPN-guided (layer {args.fpn_layer}), warmup={args.dael_warmup_epochs}ep, "
        f"lambda={args.lambda_dael}, w_tv={args.w_tv}, w_bright={args.w_bright}"
        if args.dael_mode == "fpn"
        else f"Standard (TV+brightness), lambda={args.lambda_dael}, "
             f"w_tv={args.w_tv}, w_bright={args.w_bright}"
    )
    dafm_str = {
        "full":      f"DAFM@Layer{args.fpn_layer} (channel-SE + spatial-attention) + learned importance [MAIN]",
        "heuristic": f"DAFM@Layer{args.fpn_layer} + heuristic importance",
        "none":      "No DAFM (ablation, no DAFM block)",
    }.get(args.dafm_mode, args.dafm_mode)

    def _w(b):
        import math
        x = args.recon_k * (b - args.recon_threshold)
        return 1.0 / (1.0 + math.exp(-x))

    recon_str = (
        f"ENABLED — lambda={args.lambda_recon}, threshold={args.recon_threshold}, "
        f"k={args.recon_k} "
        f"[Low=0.037→w={_w(0.037):.3f}, Amb=0.128→w={_w(0.128):.3f}, "
        f"Str=0.169→w={_w(0.169):.3f}]"
        if args.lambda_recon > 0.0 else "disabled (lambda_recon=0)"
    )

    lines = [
        "=" * 60,
        "  DAEG-YOLO: DCENet9 + DAFM + FPN-DAEL + CondRecon — TRAINING REPORT",
        "=" * 60,
        f"  Experiment    : {exp_name}",
        f"  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Base model    : {args.model}",
        f"  DAFM mode     : {dafm_str}",
        f"  CondRecon     : {recon_str}",
        f"  YAML          : {args.yaml}",
        f"  DCE config    : {args.dce_n_iter} iter, {args.dce_n_filters} filters",
        f"  DAEL          : {dael_str}",
        f"  Dataset       : {args.data}",
        f"  Batch/ImgSz   : {args.batch} / {args.imgsz}",
        f"  Seed          : {args.seed}",
        "",
        "-- Training -------------------------------------------------------",
        f"  Epochs   : {args.epochs}   Patience : {args.patience}",
        f"  LR       : {args.lr0} -> x{args.lrf}  [{args.optimizer}]",
        f"  mAP@0.5      : {fmt('metrics/mAP50(B)')}",
        f"  mAP@0.5:95   : {fmt('metrics/mAP50-95(B)')}",
        f"  Precision    : {fmt('metrics/precision(B)')}",
        f"  Recall       : {fmt('metrics/recall(B)')}",
        "",
    ]

    if test_results:
        lines += [
            "-- Test Set Results ------------------------------------------------",
            f"  mAP@0.5       : {test_results.get('mAP50', 'N/A')}",
            f"  mAP@0.5:0.95  : {test_results.get('mAP50_95', 'N/A')}",
            f"  Precision     : {test_results.get('precision', 'N/A')}",
            f"  Recall        : {test_results.get('recall', 'N/A')}",
            "",
            "-- Comparison vs Baselines ----------------------------------------",
        ]
        m50 = test_results.get("mAP50", 0)
        delta = m50 - 0.9106
        sign  = "+" if delta >= 0 else ""
        lines.append(f"  vs YOLOv8s Baseline (LLVIP)   : {sign}{delta:.4f}")
        lines.append("")

    if lighting_rows:
        lines.append("-- Per-Lighting Condition AP@0.5 ---------------------------")
        for row in lighting_rows:
            lines.append(
                f"  [{row['light_code']:2d}] {row['light_name']:<12s} "
                f"n={row['n_images']:4d}  mAP50={row['mAP50']:.4f}  "
                f"mAP50-95={row['mAP50_95']:.4f}  P={row.get('precision',0):.4f}  "
                f"R={row.get('recall',0):.4f}"
            )
        lines.append("")

    if per_class_rows:
        lines.append("-- Per-Class AP@0.5 ----------------------------------------")
        for row in per_class_rows:
            lines.append(
                f"  {row['class_name']:<12s}  AP50={row['ap50']:.4f}  AP50-95={row['ap50_95']:.4f}"
            )
        lines.append("")

    if timing:
        lines += [
            "-- Time ------------------------------------------------------------",
            f"  Training  : {timing.get('train_min', 0):.2f} menit",
            f"  Test eval : {timing.get('test_min', 0):.2f} menit",
            f"  Total     : {timing.get('total_min', 0):.2f} menit",
            "",
        ]

    if compute_cost:
        _cc = compute_cost
        lines += [
            "-- Computational Cost ----------------------------------------------",
            f"  Params total   : {_cc.get('total_params_M', 0):.3f} M",
            f"    |- YOLO+DAFM : {_cc.get('yolo_params_M', 0):.4f} M",
            f"    |- DCENet9   : {_cc.get('dce_params_M', 0):.4f} M",
            f"    |- DAFM new  : {_cc.get('dafm_params', 0)} params",
            f"  CondRecon params : 0 (no new parameters — loss term only)",
        ]
        if _cc.get("total_gflops", 0) > 0:
            lines += [
                f"  GFLOPs YOLO  : {_cc.get('yolo_gflops', 0):.1f} G",
                f"  GFLOPs DCE   : {_cc.get('dce_gflops', 0):.3f} G",
                f"  GFLOPs total : {_cc.get('total_gflops', 0):.1f} G",
            ]
        if _cc.get("fps", 0) > 0:
            lines += [
                f"  FPS (batch=1): {_cc.get('fps', 0):.1f}",
                f"  Latency      : {_cc.get('latency_ms', 0):.1f} ms",
            ]
        lines.append("")

    lines.append("=" * 60)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [REPORT] Saved: {report_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  cv2/numpy ABI patch
# ══════════════════════════════════════════════════════════════════════════════

def _patch_cv2_imread_compat():
    if _REAL_CV2_IMREAD is None:
        return
    try:
        import cv2 as _cv2
        import ultralytics.utils.patches as _ul_patches
        import ultralytics.data.base as _ul_base

        _real = _REAL_CV2_IMREAD

        def _imread_compat(filename, flags=_cv2.IMREAD_COLOR):
            img = _real(str(filename), flags)
            if img is not None and img.ndim == 2:
                img = img[..., None]
            return img

        _ul_patches.imread = _imread_compat
        _ul_base.imread    = _imread_compat
        try:
            import ultralytics.data.augment as _ul_aug
            if hasattr(_ul_aug, "imread"):
                _ul_aug.imread = _imread_compat
        except Exception:
            pass
        LOGGER.info("[DAEG-YOLO] Patched imread for cv2/numpy ABI compatibility")
    except Exception as _e:
        LOGGER.warning(f"[DAEG-YOLO] imread patch skipped: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    _patch_cv2_imread_compat()

    data_yaml = Path(fix_yaml_path(args.data))
    exp_name  = build_exp_name(args)
    out_dir   = ROOT / args.project / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _w(b):
        import math
        x = args.recon_k * (b - args.recon_threshold)
        return 1.0 / (1.0 + math.exp(-x))

    print("=" * 60)
    print("  DAEG-YOLO: DCENet9 + DAFM@L15 + FPN-DAEL + CondRecon — LLVIP")
    print(f"  Experiment  : {exp_name}")
    print(f"  YAML        : {args.yaml}")
    print(f"  DAFM mode   : {args.dafm_mode}")
    print(f"  DAEL mode   : {args.dael_mode}")
    if args.dael_mode == "fpn":
        print(f"  FPN layer   : {args.fpn_layer}  Warmup: {args.dael_warmup_epochs} epochs")
    if args.lambda_recon > 0.0:
        print(f"  CondRecon   : λ={args.lambda_recon}, threshold={args.recon_threshold}, k={args.recon_k}")
    else:
        print("  CondRecon   : DISABLED (lambda_recon=0)")
    print("=" * 60)

    enhancer  = DCENet9(n_iterations=args.dce_n_iter, n_filters=args.dce_n_filters)
    dce_total = sum(p.numel() for p in enhancer.parameters())
    print(f"  DCENet9 params: {dce_total:,} ({dce_total/1e6:.4f}M)  [no new params from CondRecon]")

    fpn_dael_fn = FPNAwareDAEL(w_tv=args.w_tv, w_bright=args.w_bright)
    std_dael_fn = StandardDAEL(w_tv=args.w_tv, w_bright=args.w_bright)

    overrides = dict(
        model        = args.model,
        data         = str(data_yaml),
        epochs       = args.epochs,
        batch        = args.batch,
        imgsz        = args.imgsz,
        patience     = args.patience,
        optimizer    = args.optimizer,
        lr0          = args.lr0,
        lrf          = args.lrf,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
        device       = args.device,
        workers      = args.workers,
        amp          = args.amp,
        seed         = args.seed,
        project      = str(ROOT / args.project),
        name         = exp_name,
        exist_ok     = True,
        plots        = True,
        verbose      = True,
    )

    trainer = DAFMTrainer(
        overrides        = overrides,
        enhancer         = enhancer,
        yaml_cfg         = args.yaml,
        fpn_dael_fn      = fpn_dael_fn,
        std_dael_fn      = std_dael_fn,
        lambda_dael      = args.lambda_dael,
        dael_mode        = args.dael_mode,
        dafm_mode        = args.dafm_mode,
        warmup_epochs    = args.dael_warmup_epochs,
        fpn_layer        = args.fpn_layer,
        lr_dce_factor    = args.lr_dce_factor,
        lambda_recon     = args.lambda_recon,
        recon_threshold  = args.recon_threshold,
        recon_k          = args.recon_k,
    )

    t_start = time.time()
    try:
        trainer.train()
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        raise
    train_min = (time.time() - t_start) / 60.0

    metrics = {}
    if hasattr(trainer, "metrics") and trainer.metrics:
        metrics = trainer.metrics if isinstance(trainer.metrics, dict) else {}

    dafm_total_params = 0
    if args.dafm_mode != "none":
        l15 = trainer.model.model[DAFM_LAYER]
        if isinstance(l15, DAFMC2f):
            dafm_total_params = sum(p.numel() for p in l15.dafm.parameters())
    print(f"  DAFM new params: {dafm_total_params}")

    t_test    = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_results, per_class_rows = run_full_test_eval(
        trainer, enhancer, data_yaml, out_dir, timestamp, args,
    )
    test_min = (time.time() - t_test) / 60.0

    
    print("[INFO] Per-lighting eval skipped (LLVIP has no lighting condition labels)")
    lighting_rows = []

    try:
        device = next(enhancer.parameters()).device
        save_enhanced_samples(
            enhancer, trainer.get_dataloader(str(data_yaml), batch_size=8),
            out_dir / "qualitative", device, dafm_mode=args.dafm_mode,
        )
    except Exception:
        pass

    try:
        yolo_params  = sum(
            p.numel() for n_, p in trainer.model.named_parameters()
            if "enhancer" not in n_
        )
        total_params = sum(p.numel() for p in trainer.model.parameters())
    except Exception:
        yolo_params  = 0
        total_params = dce_total

    print("\nMeasuring GFLOPs & FPS (batch=1)...")
    _dev  = next(enhancer.parameters()).device
    _perf = measure_compute(trainer.model, enhancer, _dev, imgsz=args.imgsz)
    print(f"  YOLO GFLOPs   : {_perf['yolo_gflops']:.1f} G")
    print(f"  DCENet9 GFLOPs: {_perf['dce_gflops']:.3f} G")
    print(f"  Total GFLOPs  : {_perf['total_gflops']:.1f} G")
    print(f"  FPS (batch=1) : {_perf['fps']:.1f}")
    print(f"  Latency       : {_perf['latency_ms']:.1f} ms")

    compute_cost = {
        "total_params_M": total_params / 1e6,
        "yolo_params_M":  yolo_params  / 1e6,
        "dce_params_M":   dce_total    / 1e6,
        "dafm_params":    dafm_total_params,
        **_perf,
    }
    timing = {"train_min": train_min, "test_min": test_min,
              "total_min": train_min + test_min}

    write_report(
        out_dir=out_dir, exp_name=exp_name, args=args, metrics=metrics,
        lighting_rows=lighting_rows, timing=timing, compute_cost=compute_cost,
        test_results=test_results, per_class_rows=per_class_rows,
    )

    print("\n" + "=" * 60)
    print(f"  DAEG-YOLO COMPLETE: {exp_name}")
    if test_results:
        m50 = test_results.get("mAP50", 0)
        print(f"  Test mAP50 : {m50}")
        delta = m50 - 0.9106
        sign  = "+" if delta >= 0 else ""
        status = "BEAT" if delta >= 0 else "below"
        print(f"  Delta vs YOLOv8s Baseline (LLVIP): {sign}{delta:.4f}  ({status})")
    print(f"  Training time  : {train_min:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
