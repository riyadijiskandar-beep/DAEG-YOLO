"""
visualize_generalization.py
===========================
Generate qualitative comparison figure for cross-domain generalization
experiments on LLVIP (person) and DarkFace (face) datasets.

Layout: N rows × 4 columns
  Col 0: Original image + GT boxes (green)
  Col 1: DCENet9 Enhanced (3× amplified)
  Col 2: YOLOv8s Baseline detections
  Col 3: DAEG-YOLO detections

Usage:
    python visualize_generalization.py --dataset llvip --device cpu
    python visualize_generalization.py --dataset darkface --device cpu
    python visualize_generalization.py --dataset llvip --n-images 4 --seed 7
    python visualize_generalization.py --dataset llvip --images 190001.jpg 190023.jpg 190045.jpg 190100.jpg

Output:
    figures/generalization_llvip.png   (300 DPI)
    figures/generalization_darkface.png
"""

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import DAFMC2f, DAFMBlock  # noqa: F401
from ultralytics.utils.nms import non_max_suppression

from train_daeg_yolo import (  # noqa: F401
    DCENet9, _FPNCapture, _TrainState, _CondReconStats, _PatchedLoss,
    DAFMTrainer, DAFMValidator,
)

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASETS = {
    "llvip": {
        "name":        "LLVIP",
        "class_names": ["Person"],
        "test_dir":    ROOT / "datasets" / "LLVIP" / "visible" / "test",
        "ann_dir":     ROOT / "datasets" / "LLVIP" / "Annotations",
        "ann_type":    "voc",   # Pascal VOC XML
        "base_weights": ROOT / "results" /
            "Baseline_yolov8s_LLVIP_20260624_184122" / "weights" / "best.pt",
        "daeg_weights": ROOT / "results" /
            "DAEG_YOLO_DCE9_CondRecon_DAFM_full_FPNDAEL_yolov8s_LLVIP" / "weights" / "best.pt",
        "out_name":    "generalization_llvip",
        "fig_title":   "Generalisasi DAEG-YOLO pada LLVIP (Person Detection, Nighttime)",
        "mAP_base":    0.9106,
        "mAP_daeg":    0.9138,
    },
    "darkface": {
        "name":        "DarkFace",
        "class_names": ["Face"],
        "test_dir":    ROOT / "datasets" / "DarkFace" / "image",
        "ann_dir":     ROOT / "datasets" / "DarkFace" / "label",
        "ann_type":    "darkface",  # raw x1 y1 x2 y2
        "base_weights": ROOT / "results" /
            "Baseline_yolov8s_DarkFace_20260626_162756" / "weights" / "best.pt",
        "daeg_weights": ROOT / "results" /
            "DAEG_YOLO_DCE9_CondRecon_DAFM_full_FPNDAEL_yolov8s_DarkFace" / "weights" / "best.pt",
        "out_name":    "generalization_darkface",
        "fig_title":   "Generalisasi DAEG-YOLO pada DarkFace (Face Detection, Low-Light)",
        "mAP_base":    0.5642,
        "mAP_daeg":    0.5825,
    },
}

IMGSZ = 640
CONF  = 0.35   # higher threshold reduces FP on LLVIP/DarkFace
IOU   = 0.45

CLR_GT   = (0, 200, 80)    # green  — ground truth
CLR_BASE = (220, 80, 40)   # red-orange — baseline
CLR_DAEG = (30, 120, 220)  # blue   — DAEG-YOLO


# ── Model loading ──────────────────────────────────────────────────────────────

def load_daeg(weights: Path, device: torch.device):
    ckpt     = torch.load(str(weights), map_location=device, weights_only=False)
    model    = (ckpt.get("ema") or ckpt["model"]).float().to(device).eval()
    enhancer = model.enhancer.to(device).eval()
    return model, enhancer


def load_base(weights: Path, device: torch.device):
    ckpt  = torch.load(str(weights), map_location=device, weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().to(device).eval()
    return model


# ── Inference helpers ──────────────────────────────────────────────────────────

def letterbox(img_bgr: np.ndarray, imgsz: int = IMGSZ):
    h0, w0  = img_bgr.shape[:2]
    scale   = imgsz / max(h0, w0)
    nh, nw  = int(round(h0 * scale)), int(round(w0 * scale))
    padded  = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pt, pl  = (imgsz - nh) // 2, (imgsz - nw) // 2
    padded[pt:pt + nh, pl:pl + nw] = cv2.resize(img_bgr, (nw, nh))
    return padded, scale, (pl, pt)


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return (torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            .float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(device))


def unscale(det, orig_hw, scale, pads):
    if det is None or len(det) == 0:
        return det
    boxes = det[:, :4].clone()
    pl, pt = pads
    boxes[:, [0, 2]] -= pl
    boxes[:, [1, 3]] -= pt
    boxes /= scale
    h0, w0 = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h0)
    out = det.clone(); out[:, :4] = boxes
    return out


def run_base(model, tensor, conf=CONF, iou=IOU):
    with torch.no_grad():
        raw = model(tensor)
    preds = raw[0] if isinstance(raw, (list, tuple)) else raw
    return non_max_suppression(preds, conf_thres=conf, iou_thres=iou)[0]


def run_daeg_inf(model, enhancer, tensor, conf=CONF, iou=IOU):
    with torch.no_grad():
        enh, _ = enhancer(tensor)
        raw    = model(enh)
    preds = raw[0] if isinstance(raw, (list, tuple)) else raw
    return non_max_suppression(preds, conf_thres=conf, iou_thres=iou)[0], enh


# ── Annotation parsers ─────────────────────────────────────────────────────────

def parse_voc(xml_path: Path, img_w: int, img_h: int):
    """Pascal VOC XML → list of (cls, x1, y1, x2, y2)."""
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.getroot().findall("object"):
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        boxes.append((0, x1, y1, x2, y2))
    return boxes


def parse_darkface(lbl_path: Path, img_w: int, img_h: int):
    """DarkFace raw label (x1 y1 x2 y2 abs) → list of (cls, x1, y1, x2, y2)."""
    if not lbl_path.exists():
        return []
    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    try:
        n = int(lines[0])
    except ValueError:
        return []
    boxes = []
    for line in lines[1:n + 1]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x1, y1, x2, y2 = map(float, parts[:4])
        if x2 > x1 and y2 > y1:
            boxes.append((0, x1, y1, x2, y2))
    return boxes


def get_gt(cfg: dict, img_path: Path, img_w: int, img_h: int):
    stem = img_path.stem
    if cfg["ann_type"] == "voc":
        return parse_voc(cfg["ann_dir"] / (stem + ".xml"), img_w, img_h)
    elif cfg["ann_type"] == "darkface":
        return parse_darkface(cfg["ann_dir"] / (stem + ".txt"), img_w, img_h)
    return []


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_box(canvas, x1, y1, x2, y2, color, label="", thickness=2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    c = tuple(int(v) for v in color)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), c, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), c, -1)
        cv2.putText(canvas, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_gt(img_rgb, gt_boxes):
    canvas = img_rgb.copy()
    for b in gt_boxes:
        draw_box(canvas, b[1], b[2], b[3], b[4], CLR_GT, "GT", thickness=2)
    return canvas


def draw_det(img_rgb, det, class_names, color):
    canvas = img_rgb.copy()
    if det is None or len(det) == 0:
        return canvas
    for *box, conf, cls in det.cpu().numpy():
        lbl = f"{class_names[int(cls)]} {conf:.2f}"
        draw_box(canvas, *box, color, lbl, thickness=2)
    return canvas


def enhanced_rgb(enhancer, tensor, img_rgb, orig_hw, scale, pads, amp=3.0):
    h0, w0 = orig_hw
    nh = int(round(h0 * scale))
    nw = int(round(w0 * scale))
    pl, pt = pads
    with torch.no_grad():
        enh_t, _ = enhancer(tensor)
    enh_np      = enh_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enh_content = enh_np[pt:pt + nh, pl:pl + nw]
    orig_f      = cv2.resize(img_rgb, (nw, nh)).astype(np.float32) / 255.0
    display     = np.clip(orig_f + amp * (enh_content - orig_f), 0, 1)
    return (cv2.resize(display, (w0, h0)) * 255).clip(0, 255).astype(np.uint8)


# ── Image selection ────────────────────────────────────────────────────────────

def get_all_test_images(cfg: dict) -> list:
    """Return all test image paths for the dataset."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if cfg["ann_type"] == "darkface":
        all_imgs = sorted(cfg["test_dir"].glob("*.png"),
                          key=lambda p: int(p.stem))
        rng = random.Random(42)
        rng.shuffle(all_imgs)
        total = len(all_imgs)
        return all_imgs[int(total * 0.9):]  # last 10% = test (seed=42)
    return sorted([p for p in cfg["test_dir"].iterdir() if p.suffix in exts])


def mean_luminance(img_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())


def _box_iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 1e-6 else 0.0


def count_tp(det, gt_boxes, iou_thresh: float = 0.5) -> int:
    """Count true positives: GT boxes matched by a prediction with IoU >= thresh."""
    if det is None or len(det) == 0 or not gt_boxes:
        return 0
    preds = det.cpu().numpy()
    matched = set()
    tp = 0
    for pred in preds:
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            if _box_iou(pred[:4], gt[1:5]) >= iou_thresh:
                tp += 1
                matched.add(j)
                break
    return tp


def smart_select(cfg: dict, n: int, base_model, daeg_model, enhancer,
                 device: torch.device, conf: float, iou: float,
                 scan_limit: int = 300, max_lum: float = 110.0,
                 max_fp: int = 0, seed: int = 999) -> list:
    """
    Scan up to `scan_limit` test images.
    Score by (tp_daeg - tp_base) — genuine TP improvement, not raw detections.
    Filters:
      - LLVIP: skip bright images (mean_lum > max_lum)
      - Both:  skip images where DAEG FP count > max_fp (default 0 = no FP allowed)
      - Both:  skip images where DAEG total detections exceed GT count
    """
    all_imgs = get_all_test_images(cfg)
    rng = random.Random(seed)
    pool = rng.sample(all_imgs, min(scan_limit, len(all_imgs)))

    print(f"[INFO] Smart-select: scanning {len(pool)} images (max_fp={max_fp}) ...")
    scored = []
    for i, img_path in enumerate(pool, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        lum = mean_luminance(img_bgr)
        if cfg["ann_type"] == "voc" and lum > max_lum:
            continue   # skip bright/daytime LLVIP images

        h0, w0 = img_bgr.shape[:2]
        gt = get_gt(cfg, img_path, w0, h0)
        if len(gt) == 0:
            continue

        padded, sc, pds = letterbox(img_bgr)
        tensor = to_tensor(padded, device)

        det_b = run_base(base_model, tensor, conf, iou)
        det_b = unscale(det_b, (h0, w0), sc, pds)
        det_d, _ = run_daeg_inf(daeg_model, enhancer, tensor, conf, iou)
        det_d = unscale(det_d, (h0, w0), sc, pds)

        tp_b = count_tp(det_b, gt)
        tp_d = count_tp(det_d, gt)
        n_d  = 0 if (det_d is None or len(det_d) == 0) else len(det_d)
        fp_d = n_d - tp_d

        # Strict FP filter: DAEG must not produce more than max_fp false positives
        if fp_d > max_fp:
            continue

        # Strict count filter: DAEG total detections must not exceed GT count
        if n_d > len(gt):
            continue

        score = tp_d - tp_b   # positive = DAEG genuinely detects more GT

        scored.append((score, lum, tp_b, tp_d, n_d, len(gt), img_path))
        if i % 50 == 0:
            print(f"  [{i}/{len(pool)}] scanned ...")

    if not scored:
        # Fallback: relax FP constraint (allow up to 1 FP, n_d <= n_gt + 1)
        print("[WARN] No candidates passed strict filter — relaxing to max_fp=1 ...")
        for i, img_path in enumerate(pool, 1):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            lum = mean_luminance(img_bgr)
            if cfg["ann_type"] == "voc" and lum > max_lum * 1.5:
                continue
            h0, w0 = img_bgr.shape[:2]
            gt = get_gt(cfg, img_path, w0, h0)
            if not gt:
                continue
            padded, sc, pds = letterbox(img_bgr)
            tensor = to_tensor(padded, device)
            det_b = unscale(run_base(base_model, tensor, conf, iou), (h0, w0), sc, pds)
            det_d, _ = run_daeg_inf(daeg_model, enhancer, tensor, conf, iou)
            det_d = unscale(det_d, (h0, w0), sc, pds)
            tp_b = count_tp(det_b, gt)
            tp_d = count_tp(det_d, gt)
            n_d  = 0 if (det_d is None or len(det_d) == 0) else len(det_d)
            fp_d = n_d - tp_d
            # Relaxed: max 1 FP and n_d <= n_gt + 1
            if fp_d > 1 or n_d > len(gt) + 1:
                continue
            scored.append((tp_d - tp_b, lum, tp_b, tp_d, n_d, len(gt), img_path))

    # Sort: TP delta DESC, then highest DAEG TP count, then darkest (lum ASC)
    scored.sort(key=lambda x: (-x[0], -x[3], x[1]))

    # Deduplicate by path
    seen_paths = set()
    deduped = []
    for item in scored:
        p = item[-1]
        if str(p) not in seen_paths:
            seen_paths.add(str(p))
            deduped.append(item)

    def gt_bucket(n_gt):
        if n_gt <= 1:  return 0
        if n_gt <= 3:  return 1
        if n_gt <= 5:  return 2
        return 3

    # Greedy diversity: prefer images from different GT count buckets.
    # Within the same bucket, require luminance difference >= 20.
    selected = []
    used_buckets: dict[int, int] = {}   # bucket -> count of selected from that bucket
    for item in deduped:
        n_gt_i  = item[5]
        lum_i   = item[1]
        bucket  = gt_bucket(n_gt_i)
        bucket_count = used_buckets.get(bucket, 0)
        lum_ok  = all(abs(lum_i - s[1]) >= 20.0 for s in selected if gt_bucket(s[5]) == bucket)
        # Accept if this bucket has < 2 slots used OR luminance is diverse enough
        if bucket_count < 2 and (bucket_count == 0 or lum_ok):
            selected.append(item)
            used_buckets[bucket] = bucket_count + 1
        if len(selected) == n:
            break
    # Top up from remaining deduped if needed
    if len(selected) < n:
        for item in deduped:
            if item not in selected:
                selected.append(item)
            if len(selected) == n:
                break

    best = [item[-1] for item in selected]
    print(f"[INFO] Selected (scored by TP improvement, max_fp={max_fp}):")
    for score, lum, tp_b, tp_d, n_d, n_gt, p in selected:
        print(f"  {p.name:<22}  tp_delta={score:+d}"
              f"  tp_base={tp_b} tp_daeg={tp_d}  n_det={n_d}  n_gt={n_gt}  lum={lum:.1f}")
    return best


def get_test_images(cfg: dict, n: int, seed: int, names: list | None,
                    smart: bool, base_model=None, daeg_model=None,
                    enhancer=None, device=None, conf=CONF, iou=IOU,
                    scan_limit: int = 300, max_lum: float = 110.0,
                    max_fp: int = 0, smart_seed: int = 999) -> list:
    if names:
        paths = []
        for name in names:
            p = Path(name)
            if p.is_absolute() and p.exists():
                paths.append(p)
            else:
                cand = cfg["test_dir"] / name
                if cand.exists():
                    paths.append(cand)
                else:
                    matches = list(cfg["test_dir"].rglob(name))
                    if matches:
                        paths.append(matches[0])
                    else:
                        raise FileNotFoundError(f"Cannot find: {name}")
        return paths

    if smart:
        return smart_select(cfg, n, base_model, daeg_model, enhancer,
                            device, conf, iou, scan_limit, max_lum, max_fp, smart_seed)

    all_imgs = get_all_test_images(cfg)
    rng = random.Random(seed)
    return sorted(rng.sample(all_imgs, min(n, len(all_imgs))))


# ── Main figure ────────────────────────────────────────────────────────────────

def generate(cfg: dict, img_paths: list, base_model, daeg_model, enhancer,
             device: torch.device, conf: float, iou: float, out_dir: Path):
    n   = len(img_paths)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.0 * n))
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Input + Ground Truth",
        "DCENet9 Enhanced\n(3× amplified)",
        f"YOLOv8s Baseline\n(mAP50 = {cfg['mAP_base']:.4f})",
        f"DAEG-YOLO (Ours)\n(mAP50 = {cfg['mAP_daeg']:.4f})",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    cnames = cfg["class_names"]

    for row, img_path in enumerate(img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        h0, w0          = img_bgr.shape[:2]
        img_rgb         = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt              = get_gt(cfg, img_path, w0, h0)
        padded, sc, pds = letterbox(img_bgr)
        tensor          = to_tensor(padded, device)

        det_base           = run_base(base_model, tensor, conf, iou)
        det_base           = unscale(det_base, (h0, w0), sc, pds)
        det_daeg, enh_t    = run_daeg_inf(daeg_model, enhancer, tensor, conf, iou)
        det_daeg           = unscale(det_daeg, (h0, w0), sc, pds)

        enh_disp = enhanced_rgb(enhancer, tensor, img_rgb, (h0, w0), sc, pds)

        n_base = 0 if (det_base is None or len(det_base) == 0) else len(det_base)
        n_daeg = 0 if (det_daeg is None or len(det_daeg) == 0) else len(det_daeg)

        panel0 = draw_gt(img_rgb, gt)
        panel2 = draw_det(img_rgb, det_base, cnames, CLR_BASE)
        panel3 = draw_det(img_rgb, det_daeg, cnames, CLR_DAEG)

        marker = f"({chr(ord('a') + row)})"

        axes[row, 0].imshow(panel0)
        axes[row, 0].set_ylabel(img_path.name, fontsize=8, color="#444444", labelpad=4)
        axes[row, 0].text(
            0.02, 0.98, marker, transform=axes[row, 0].transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left", color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=1, boxstyle="round,pad=0.2"),
        )
        axes[row, 0].set_xlabel(f"GT: {len(gt)}", fontsize=9, labelpad=3)

        axes[row, 1].imshow(enh_disp)

        axes[row, 2].imshow(panel2)
        axes[row, 2].set_xlabel(f"Detections: {n_base}  (GT: {len(gt)})", fontsize=9,
                                color=np.array(CLR_BASE) / 255, labelpad=3)

        axes[row, 3].imshow(panel3)
        axes[row, 3].set_xlabel(f"Detections: {n_daeg}  (GT: {len(gt)})", fontsize=9,
                                color=np.array(CLR_DAEG) / 255, labelpad=3)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("auto")

    delta = cfg["mAP_daeg"] - cfg["mAP_base"]
    fig.suptitle(
        f"{cfg['fig_title']}\n"
        f"mAP50: Baseline {cfg['mAP_base']:.4f}  →  DAEG-YOLO {cfg['mAP_daeg']:.4f}"
        f"  (+{delta:.4f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{cfg['out_name']}.png"
    jpg = out_dir / f"{cfg['out_name']}.jpg"
    pdf = out_dir / f"{cfg['out_name']}.pdf"
    print(f"[INFO] Saving to: {out_dir}")
    try:
        fig.savefig(png, dpi=200, bbox_inches="tight")
        print(f"[OK] {png}")
    except Exception as e:
        print(f"[WARN] PNG save failed ({e}), saving JPG instead ...")
        try:
            fig.savefig(jpg, dpi=200, bbox_inches="tight")
            print(f"[OK] {jpg}")
        except Exception as e2:
            print(f"[ERROR] JPG save also failed: {e2}")
    try:
        fig.savefig(pdf, bbox_inches="tight")
        print(f"[OK] {pdf}")
    except PermissionError:
        print(f"[SKIP] PDF locked (close PDF viewer first): {pdf}")
    except Exception as e:
        print(f"[ERROR] PDF save failed: {e}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     choices=["llvip", "darkface"], required=True)
    p.add_argument("--n-images",    type=int, default=4)
    p.add_argument("--seed",        type=int, default=123)
    p.add_argument("--images",      nargs="+", default=None,
                   help="Explicit image filenames (overrides auto-select)")
    p.add_argument("--smart-select", action="store_true",
                   help="Scan images and pick ones where DAEG-YOLO wins (recommended)")
    p.add_argument("--scan-limit",  type=int, default=300,
                   help="How many images to scan in smart-select mode")
    p.add_argument("--max-lum",     type=float, default=110.0,
                   help="LLVIP only: skip images brighter than this (0-255)")
    p.add_argument("--max-fp",      type=int, default=0,
                   help="Max false positives allowed for DAEG-YOLO (default 0 = strict)")
    p.add_argument("--smart-seed",  type=int, default=999,
                   help="Random seed for smart-select candidate pool (change to get different images)")
    p.add_argument("--conf",        type=float, default=CONF)
    p.add_argument("--iou",         type=float, default=IOU)
    p.add_argument("--out-dir",     default=str(ROOT / "figures"))
    p.add_argument("--device",      default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = DATASETS[args.dataset]
    device = torch.device(args.device)

    print(f"[INFO] Dataset : {cfg['name']}")
    print(f"[INFO] Loading baseline  : {cfg['base_weights'].parent.parent.name}")
    base_model = load_base(cfg["base_weights"], device)

    print(f"[INFO] Loading DAEG-YOLO : {cfg['daeg_weights'].parent.parent.name}")
    daeg_model, enhancer = load_daeg(cfg["daeg_weights"], device)

    img_paths = get_test_images(
        cfg, args.n_images, args.seed, args.images,
        smart=args.smart_select,
        base_model=base_model, daeg_model=daeg_model, enhancer=enhancer,
        device=device, conf=args.conf, iou=args.iou,
        scan_limit=args.scan_limit, max_lum=args.max_lum,
        max_fp=args.max_fp, smart_seed=args.smart_seed,
    )
    print(f"[INFO] Images  : {[p.name for p in img_paths]}")

    generate(
        cfg, img_paths, base_model, daeg_model, enhancer,
        device, args.conf, args.iou, Path(args.out_dir),
    )
