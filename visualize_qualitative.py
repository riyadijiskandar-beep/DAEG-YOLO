"""
visualize_qualitative.py
========================
Generate qualitative detection comparison figure for paper.

Layout (N rows × 3 columns):
  Col 1: Original low-light image
  Col 2: YOLOv8s baseline detections
  Col 3: DAEG-YOLO detections

For each lighting condition (default: Low, Object, Ambient, Twilight):
  - Run both models on the same image
  - Draw bounding boxes with class labels and confidence scores
  - Show side by side

Usage:
    python visualize_qualitative.py
    python visualize_qualitative.py --light-conditions 1 3 2 9
    python visualize_qualitative.py --images img1.jpg img2.jpg img3.jpg img4.jpg
    python visualize_qualitative.py --conf 0.25 --device cuda:0

Output:
    figures/qualitative_comparison.png  (300 DPI)
    figures/qualitative_comparison.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import cv2

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import DAFMC2f, DAFMBlock  # noqa: F401 — registers custom blocks
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

# Import all custom classes from training script so torch.load can deserialize checkpoint
from train_daeg_yolo import (  # noqa: F401
    DCENet9, _FPNCapture, _TrainState, _CondReconStats, _PatchedLoss,
    DAFMTrainer, DAFMValidator,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DAEG_WEIGHTS = ROOT / "results" / (
    ""  # fill in your DAEG-YOLO run folder (results/<exp_name>)
) / "weights" / "best.pt"

BASE_WEIGHTS = ROOT / "results" / (
    "Baseline_yolov8s_ExDark_20260606_125223"
) / "weights" / "best.pt"

TEST_DIR  = ROOT / "datasets" / "ExDark_Official" / "images" / "test"
ANNNO_TXT = ROOT / "ExDark_Annno" / "imageclasslist.txt"
OUT_DIR   = ROOT / "figures"

CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
LIGHT_NAMES = {
    1: "Low", 2: "Ambient", 3: "Object", 4: "Single",
    5: "Weak", 6: "Strong", 7: "Window", 8: "Shadow",
    9: "Twilight", 10: "Fluorescent",
}

IMGSZ   = 640
# Distinct colors per class (12 classes)
_PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168),  (0,   194, 255),
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_daeg_model(weights: Path, device: torch.device):
    ckpt     = torch.load(str(weights), map_location=device, weights_only=False)
    model    = (ckpt.get("ema") or ckpt["model"]).float().to(device).eval()
    enhancer = model.enhancer.to(device).eval()
    return model, enhancer


def load_base_model(weights: Path, device: torch.device):
    ckpt  = torch.load(str(weights), map_location=device, weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().to(device).eval()
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────

def letterbox_image(img_bgr: np.ndarray, imgsz: int = IMGSZ):
    """Letterbox to square, return padded image + scale + pads."""
    h0, w0  = img_bgr.shape[:2]
    scale   = imgsz / max(h0, w0)
    nh, nw  = int(round(h0 * scale)), int(round(w0 * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top = (imgsz - nh) // 2
    pad_lft = (imgsz - nw) // 2
    padded[pad_top:pad_top + nh, pad_lft:pad_lft + nw] = resized
    return padded, scale, (pad_lft, pad_top)


def img_to_tensor(img_bgr_padded: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb    = cv2.cvtColor(img_bgr_padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def run_inference(model, tensor: torch.Tensor, conf: float, iou: float) -> torch.Tensor:
    """Run model and return NMS detections (N, 6): [x1,y1,x2,y2,conf,cls]."""
    with torch.no_grad():
        raw = model(tensor)
    preds = raw[0] if isinstance(raw, (list, tuple)) else raw
    det   = non_max_suppression(preds, conf_thres=conf, iou_thres=iou)[0]
    return det  # (N, 6) or empty tensor


def run_daeg_inference(
    model, enhancer, tensor: torch.Tensor, conf: float, iou: float
) -> torch.Tensor:
    """Run DCENet9 → YOLO → NMS."""
    with torch.no_grad():
        enhanced, _ = enhancer(tensor)
        raw         = model(enhanced)
    preds = raw[0] if isinstance(raw, (list, tuple)) else raw
    det   = non_max_suppression(preds, conf_thres=conf, iou_thres=iou)[0]
    return det


def unscale_boxes(
    det: torch.Tensor,
    imgsz: int,
    orig_hw: tuple[int, int],
    scale: float,
    pads: tuple[int, int],
) -> torch.Tensor:
    """Convert padded-image boxes back to original image coordinates."""
    if det is None or len(det) == 0:
        return det
    boxes      = det[:, :4].clone()
    pad_lft, pad_top = pads
    boxes[:, [0, 2]] -= pad_lft
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale
    h0, w0 = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h0)
    det = det.clone()
    det[:, :4] = boxes
    return det


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(
    img_rgb: np.ndarray,
    det: torch.Tensor,
    class_names: list[str],
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on RGB image. Returns new RGB array."""
    canvas = img_rgb.copy()
    if det is None or len(det) == 0:
        return canvas

    for *box, conf, cls_id in det.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        color  = _PALETTE[cls_id % len(_PALETTE)]
        label  = f"{class_names[cls_id]} {conf:.2f}"

        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        # Label text
        cv2.putText(
            canvas, label, (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def count_det(det) -> int:
    return 0 if (det is None or len(det) == 0) else len(det)


# ── Image selection ───────────────────────────────────────────────────────────

def pick_images_by_lighting(
    test_dir: Path,
    annno_txt: Path,
    light_codes: list[int],
) -> dict[int, Path]:
    stem_to_light: dict[str, int] = {}
    with open(annno_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0].startswith("Name"):
                continue
            fname, _cls, light, _io, split = parts[:5]
            if int(split) == 3:
                stem_to_light[Path(fname).stem] = int(light)

    result: dict[int, Path] = {}
    for img_path in sorted(test_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        code = stem_to_light.get(img_path.stem)
        if code in light_codes and code not in result:
            result[code] = img_path
        if len(result) == len(light_codes):
            break
    return result


# ── Main figure ───────────────────────────────────────────────────────────────

def generate_qualitative_figure(
    image_paths: list[Path],
    light_labels: list[str],
    base_model,
    daeg_model,
    enhancer,
    device: torch.device,
    out_dir: Path,
    conf: float = 0.25,
    iou:  float = 0.45,
) -> None:
    n   = len(image_paths)
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.5 * n))
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original Image", "DCENet9 Enhanced\n(3× amplified)", "YOLOv8s (Baseline)", "DAEG-YOLO (Ours)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row, (img_path, lbl) in enumerate(zip(image_paths, light_labels)):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        orig_hw             = img_bgr.shape[:2]
        padded, scale, pads = letterbox_image(img_bgr)
        tensor              = img_to_tensor(padded, device)
        img_rgb             = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # DCENet9 enhanced image (computed once for display + inference)
        with torch.no_grad():
            enhanced, _ = enhancer(tensor)
        # Crop letterbox padding, resize back to original resolution for clean display
        h0, w0 = orig_hw
        nh = int(round(h0 * scale))
        nw = int(round(w0 * scale))
        pad_lft, pad_top = pads
        enh_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enh_content = enh_np[pad_top:pad_top + nh, pad_lft:pad_lft + nw]
        # 3× amplified delta for display (CondRecon keeps enhancement conservative)
        orig_content = cv2.resize(img_rgb, (nw, nh)).astype(np.float32) / 255.0
        enh_display  = np.clip(orig_content + 3.0 * (enh_content - orig_content), 0, 1)
        enhanced_rgb = (cv2.resize(enh_display, (w0, h0)) * 255).clip(0, 255).astype(np.uint8)

        # Baseline inference on original padded tensor
        det_base = run_inference(base_model, tensor, conf, iou)
        det_base = unscale_boxes(det_base, IMGSZ, orig_hw, scale, pads)

        # DAEG inference on already-enhanced image (reuse enhanced tensor)
        with torch.no_grad():
            raw = daeg_model(enhanced)
        preds    = raw[0] if isinstance(raw, (list, tuple)) else raw
        det_daeg = non_max_suppression(preds, conf_thres=conf, iou_thres=iou)[0]
        det_daeg = unscale_boxes(det_daeg, IMGSZ, orig_hw, scale, pads)

        drawn_base = draw_detections(img_rgb, det_base, CLASS_NAMES)
        drawn_daeg = draw_detections(img_rgb, det_daeg, CLASS_NAMES)

        n_base = count_det(det_base)
        n_daeg = count_det(det_daeg)

        # Row marker (a)(b)... as text overlay inside top-left of col 0
        marker = f"({chr(ord('a') + row)})"

        # Col 0: original
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_ylabel(f"Lighting: {lbl}", fontsize=10, fontweight="bold", labelpad=6)
        axes[row, 0].text(
            0.02, 0.98, marker, transform=axes[row, 0].transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left", color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=1, boxstyle="round,pad=0.2"),
        )

        # Col 1: DCENet9 enhanced
        axes[row, 1].imshow(enhanced_rgb)

        # Col 2: baseline detections
        axes[row, 2].imshow(drawn_base)
        axes[row, 2].set_xlabel(f"Detections: {n_base}", fontsize=9, color="#d62728", labelpad=4)

        # Col 3: DAEG-YOLO detections
        axes[row, 3].imshow(drawn_daeg)
        axes[row, 3].set_xlabel(f"Detections: {n_daeg}", fontsize=9, color="#1f77b4", labelpad=4)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")  # fill subplot cell uniformly regardless of image aspect ratio

    fig.suptitle(
        "Figure 8: Qualitative Detection Comparison: Original → DCENet9 Enhanced → YOLOv8s Baseline → DAEG-YOLO (ExDark Test Set)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "qualitative_comparison.png"
    pdf_path = out_dir / "qualitative_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")
    print(f"[INFO] Detection counts per row:")
    print(f"       Labels: {light_labels}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--daeg-weights",      default=str(DAEG_WEIGHTS))
    p.add_argument("--base-weights",      default=str(BASE_WEIGHTS))
    p.add_argument("--images",            nargs="+", default=None,
                   help="Manual image paths (4 images recommended)")
    p.add_argument("--labels",            nargs="+", default=None,
                   help="Row labels matching --images (e.g. Ambient Shadow Strong Low)")
    p.add_argument("--light-conditions",  nargs="+", type=int,
                   default=[1, 3, 2, 9],
                   help="Lighting condition codes: 1=Low,2=Ambient,3=Object,9=Twilight")
    p.add_argument("--conf",              type=float, default=0.25)
    p.add_argument("--iou",               type=float, default=0.45)
    p.add_argument("--out-dir",           default=str(OUT_DIR))
    p.add_argument("--device",            default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device)

    print("[INFO] Loading DAEG-YOLO model ...")
    daeg_model, enhancer = load_daeg_model(Path(args.daeg_weights), device)

    print("[INFO] Loading baseline YOLOv8s model ...")
    base_model = load_base_model(Path(args.base_weights), device)

    if args.images:
        image_paths  = [Path(p) for p in args.images]
        light_labels = args.labels if args.labels else [p.stem for p in image_paths]
    else:
        print(f"[INFO] Auto-selecting images for conditions: {args.light_conditions}")
        code_to_path = pick_images_by_lighting(
            TEST_DIR, ANNNO_TXT, args.light_conditions
        )
        image_paths  = [code_to_path[c] for c in args.light_conditions if c in code_to_path]
        light_labels = [
            LIGHT_NAMES.get(c, str(c))
            for c in args.light_conditions if c in code_to_path
        ]
        print(f"[INFO] Selected: {[str(p.name) for p in image_paths]}")

    generate_qualitative_figure(
        image_paths, light_labels,
        base_model, daeg_model, enhancer,
        device, Path(args.out_dir),
        conf=args.conf, iou=args.iou,
    )
