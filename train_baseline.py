"""
Baseline YOLOv8 Training on ExDark Dataset
===========================================
Model  : YOLOv8 pretrained on COCO (transfer learning)
Dataset: ExDark_Official - 7363 images, 12 classes, 80/10/10 split

Usage examples:
  # Default (batch=12)
  python train_baseline.py

  # Custom model and epoch
  python train_baseline.py --model yolov8s.pt --epochs 150

  # Lower GPU
  python train_baseline.py --batch 8 --workers 2 --no-amp

  # Change dataset path
  python train_baseline.py --data path/to/MyDataset.yaml

  # Full 
  python train_baseline.py --model yolov8s.pt --epochs 100 --batch 32 \
      --imgsz 640 --workers 8 --device 0 --optimizer SGD --lr0 0.01 \
      --patience 20 --qualitative 16

Output structure:
  results/
    Baseline_yolov8s_ExDark_YYYYMMDD_HHMMSS/
      weights/          best.pt, last.pt
      metrics/          config.json, per_epoch.csv,
                        test_results.json, per_class_ap.csv
      plots/            loss_curves.png, PR_curve.png, F1_curve.png,
                        confusion_matrix.png
      qualitative/      sample_detections/  (N gambar test + bbox)
      report/           summary_YYYYMMDD_HHMMSS.txt
"""

import os, sys, json, csv, time, shutil, argparse
from datetime import datetime
from pathlib import Path

sys.path = [p for p in sys.path if "ProjectResNetDCE1" not in p and p != ""]

_user = [p for p in sys.path if "AppData" in p and "Roaming" in p]
_rest = [p for p in sys.path if p not in _user]
sys.path = _rest + _user

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONSTANT
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES  = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
]
# ExDark 10 illumination conditions (imageclasslist.txt column 3)
LIGHT_NAMES = {
    1: "Low",        2: "Ambient",     3: "Object",   4: "Single",
    5: "Weak",       6: "Strong",      7: "Window",   8: "Shadow",
    9: "Twilight",  10: "Fluorescent",
}
ANNNO_TXT    = PROJECT_ROOT / "ExDark_Annno" / "imageclasslist.txt"
DEFAULT_DATA = str(PROJECT_ROOT / "datasets" / "ExDark_Official" / "ExDark.yaml")

# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline YOLOv8 Training on ExDark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="yolov8s.pt",
        help="Model weight: yolov8n/s/m/l/x.pt atau path ke .pt lokal")

    # ── Dataset ───────────────────────────────────────────────
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
        help="Path ke dataset YAML config")

    # ── Training ──────────────────────────────────────────────
    parser.add_argument("--epochs",   type=int,   default=100,
        help="Jumlah epoch training")
    parser.add_argument("--batch",    type=int,   default=12,
        help="Batch size (default=12 untuk konsistensi dengan proposed method EAFM-DAEL)")
    parser.add_argument("--imgsz",    type=int,   default=640,
        help="Ukuran input gambar (piksel)")
    parser.add_argument("--workers",  type=int,   default=8,
        help="Jumlah dataloader worker threads")
    parser.add_argument("--device",   type=str,   default="0",
        help="GPU device: '0', '0,1', atau 'cpu'")
    parser.add_argument("--patience", type=int,   default=20,
        help="Early stopping patience (epoch tanpa peningkatan)")

    # ── Optimizer ─────────────────────────────────────────────
    parser.add_argument("--optimizer", type=str, default="SGD",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer")
    parser.add_argument("--lr0",  type=float, default=0.01,
        help="Learning rate awal")
    parser.add_argument("--lrf",  type=float, default=0.01,
        help="Learning rate akhir (sebagai faktor pengali lr0)")
    parser.add_argument("--momentum",     type=float, default=0.937,
        help="SGD momentum / Adam beta1")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
        help="Weight decay (L2 regularization)")

    # ── Feature ─────────────────────────────────────────────────
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Automatic Mixed Precision (--amp / --no-amp)")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False,
        help="Cache dataset ke RAM (--cache / --no-cache)")

    # ── Output ────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "results"),
        help="Folder root untuk menyimpan semua hasil")
    parser.add_argument("--qualitative", type=int, default=16,
        help="Jumlah gambar test untuk visualisasi deteksi kualitatif")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed untuk reproducibility")

    return parser.parse_args()


# ============================================================
# RUN SETUP 
# ============================================================

def build_run_context(args):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    model_stem = Path(args.model).stem
    run_name   = f"Baseline_{model_stem}_ExDark_{timestamp}"
    run_dir    = Path(args.project) / run_name

    train_cfg = dict(
        data             = args.data,
        epochs           = args.epochs,
        imgsz            = args.imgsz,
        batch            = args.batch,
        workers          = args.workers,
        optimizer        = args.optimizer,
        lr0              = args.lr0,
        lrf              = args.lrf,
        momentum         = args.momentum,
        weight_decay     = args.weight_decay,
        warmup_epochs    = 3,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,
        box              = 7.5,
        cls              = 0.5,
        dfl              = 1.5,
        mosaic           = 1.0,
        mixup            = 0.0,
        copy_paste       = 0.0,
        degrees          = 0.0,
        translate        = 0.1,
        scale            = 0.5,
        shear            = 0.0,
        flipud           = 0.0,
        fliplr           = 0.5,
        hsv_h            = 0.015,
        hsv_s            = 0.7,
        hsv_v            = 0.4,
        patience         = args.patience,
        save             = True,
        save_period      = -1,
        cache            = args.cache,
        device           = args.device,
        exist_ok         = True,
        pretrained       = True,
        amp              = args.amp,
        verbose          = True,
        seed             = args.seed,
        deterministic    = True,
        project          = args.project,
        name             = run_name,
    )

    return timestamp, run_name, run_dir, train_cfg


# ============================================================
# UTILITIES
# ============================================================

def make_dirs(run_dir: Path):
    for sub in ["weights", "metrics", "plots", "qualitative/sample_detections", "report"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def save_config_snapshot(run_dir: Path, train_cfg: dict, args, timestamp: str, run_name: str):
    cfg          = {k: str(v) if isinstance(v, Path) else v for k, v in train_cfg.items()}
    cfg["run_name"]  = run_name
    cfg["timestamp"] = timestamp
    cfg["model"]     = args.model
    cfg["gpu"]       = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cfg["vram_gb"]   = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) \
                       if torch.cuda.is_available() else 0
    out = run_dir / "metrics" / f"config_{timestamp}.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[CONFIG] Saved: {out.name}")


def copy_training_metrics(train_dir: Path, run_dir: Path, timestamp: str):
    src = train_dir / "results.csv"
    if src.exists():
        dst = run_dir / "metrics" / f"per_epoch_{timestamp}.csv"
        shutil.copy2(src, dst)
        print(f"[METRICS] Per-epoch CSV saved: {dst.name}")
        return dst
    return None


def save_test_results(metrics: dict, run_dir: Path, run_name: str, timestamp: str):
    results = {
        "timestamp" : timestamp,
        "split"     : "test",
        "model"     : run_name,
        "mAP50"     : round(float(metrics.get("metrics/mAP50(B)",    0)), 4),
        "mAP50_95"  : round(float(metrics.get("metrics/mAP50-95(B)", 0)), 4),
        "precision" : round(float(metrics.get("metrics/precision(B)", 0)), 4),
        "recall"    : round(float(metrics.get("metrics/recall(B)",    0)), 4),
    }
    out = run_dir / "metrics" / f"test_results_{timestamp}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[RESULTS] Test results saved: {out.name}")
    return results


def save_per_class_ap(val_result, run_dir: Path, timestamp: str):
    out = run_dir / "metrics" / f"per_class_ap_test_{timestamp}.csv"
    try:
        ap_per_class = val_result.box.ap_class_index
        maps         = val_result.box.maps
        ap50         = val_result.box.ap50
        rows = []
        for i, cls_idx in enumerate(ap_per_class):
            rows.append({
                "class_id"  : int(cls_idx),
                "class_name": CLASS_NAMES[int(cls_idx)],
                "AP50"      : round(float(ap50[i]), 4),
                "AP50_95"   : round(float(maps[i]), 4),
            })
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "AP50", "AP50_95"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[METRICS] Per-class AP saved: {out.name}")
        return rows
    except Exception as e:
        print(f"[WARN] Could not save per-class AP: {e}")
        return []


def copy_plots(train_dir: Path, run_dir: Path):
    plot_files = [
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
        "results.png", "val_batch0_pred.jpg", "val_batch1_pred.jpg",
    ]
    copied = []
    for fname in plot_files:
        src = train_dir / fname
        if src.exists():
            shutil.copy2(src, run_dir / "plots" / fname)
            copied.append(fname)
    if copied:
        print(f"[PLOTS] Copied {len(copied)} plot files to plots/")


def plot_loss_curves(metrics_csv: Path, run_dir: Path, run_name: str, timestamp: str):
    if metrics_csv is None or not metrics_csv.exists():
        return
    epochs, box_loss, cls_loss, dfl_loss, map50, map5095 = [], [], [], [], [], []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # strip whitespace dari key/value Ultralytics CSV
                clean = {k.strip(): v.strip() for k, v in row.items()}
                epochs.append(int(float(clean.get("epoch", 0))))
                box_loss.append(float(clean.get("train/box_loss", 0)))
                cls_loss.append(float(clean.get("train/cls_loss", 0)))
                dfl_loss.append(float(clean.get("train/dfl_loss", 0)))
                map50.append(float(clean.get("metrics/mAP50(B)", 0)))
                map5095.append(float(clean.get("metrics/mAP50-95(B)", 0)))
            except Exception:
                continue
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=9)

    axes[0].plot(epochs, box_loss, label="Box Loss",  color="#e74c3c")
    axes[0].plot(epochs, cls_loss, label="Cls Loss",  color="#3498db")
    axes[0].plot(epochs, dfl_loss, label="DFL Loss",  color="#2ecc71")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Losses"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, map50,   label="mAP@0.5",      color="#9b59b6")
    axes[1].plot(epochs, map5095, label="mAP@0.5:0.95", color="#f39c12")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
    axes[1].set_title("Validation mAP"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = run_dir / "plots" / f"loss_and_map_curves_{timestamp}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOTS] Loss curves saved: {out.name}")


def save_qualitative_detections(model_path: Path, run_dir: Path,
                                 timestamp: str, args, dataset_yaml: str):
    test_img_dir = Path(dataset_yaml).parent / "images" / "test"
    out_dir      = run_dir / "qualitative" / "sample_detections"

    img_files = sorted(f for f in test_img_dir.glob("*")
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n = args.qualitative
    step    = max(1, len(img_files) // n)
    samples = img_files[::step][:n]

    if not samples:
        print("[QUALITATIVE] No test images found.")
        return

    model  = YOLO(str(model_path))
    colors = matplotlib.colormaps.get_cmap("tab20").resampled(12)

    print(f"[QUALITATIVE] Generating {len(samples)} detection samples...")
    for idx, img_path in enumerate(samples):
        try:
            result = model.predict(
                source=str(img_path), conf=0.25, iou=0.45,
                device=args.device, verbose=False, save=False,
            )[0]

            img = Image.open(img_path).convert("RGB")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(img); ax.axis("off")

            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0]); conf = float(box.conf[0])
                    color  = colors(cls_id)[:3]
                    ax.add_patch(patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=color, facecolor="none"))
                    ax.text(x1, y1-4, f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                            fontsize=7, color="white",
                            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))

            n_det = len(result.boxes) if result.boxes is not None else 0
            ax.set_title(f"{img_path.name}  |  {n_det} detections", fontsize=8, pad=4)
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{idx+1:02d}_{img_path.stem}_{timestamp}.jpg",
                        dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

    print(f"[QUALITATIVE] Saved {len(samples)} images to qualitative/sample_detections/")


def parse_imageclasslist(annno_txt: Path) -> dict:
    """
    Parse imageclasslist.txt → {stem_lower: light_code}.
    Format: Name Class Light In/Out Train/Val/Test
    """
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


def eval_by_lighting(
    model_path: Path,
    test_img_dir: Path,
    data_yaml: Path,
    annno_txt: Path,
    run_dir: Path,
    timestamp: str,
    args,
) -> list:
    """
    Evaluate best.pt on the test set grouped by ExDark lighting condition.
    Creates temp per-condition txt + YAML, runs model.val() for each group,
    saves results to metrics/per_lighting_<timestamp>.csv.
    """
    import yaml as _yaml

    if not annno_txt.exists():
        print(f"  [SKIP] imageclasslist.txt not found: {annno_txt}")
        return []

    img_light = parse_imageclasslist(annno_txt)

    # Collect test images and group by lighting code
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    test_images = [p for p in sorted(test_img_dir.iterdir())
                   if p.suffix in suffixes or p.suffix.lower() in suffixes]

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

    # Load base YAML for nc/names
    with open(data_yaml, encoding="utf-8") as f:
        base_cfg = _yaml.safe_load(f)

    out_dir  = run_dir / "metrics"
    tmp_dir  = run_dir / "_tmp_lighting"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    rows  = []

    dataset_root = str(data_yaml.resolve().parent).replace("\\", "/")

    for code in sorted(groups.keys()):
        img_paths  = groups[code]
        light_name = LIGHT_NAMES.get(code, f"Cond_{code}")
        n          = len(img_paths)
        print(f"  [{code:2d}] {light_name:12s} ({n:4d} images) ...", end="", flush=True)

        # Write txt with absolute forward-slash paths
        txt_path = tmp_dir / f"light_{code}.txt"
        txt_path.write_text(
            "\n".join(str(p.resolve()).replace("\\", "/") for p in img_paths),
            encoding="utf-8",
        )

        # Temp YAML — 'test' points to the txt file
        # 'train' and 'val' are required by Ultralytics validator even for test-only eval
        txt_fwd = str(txt_path.resolve()).replace("\\", "/")
        tmp_cfg = {
            "path":  dataset_root,
            "train": txt_fwd,
            "val":   txt_fwd,
            "test":  txt_fwd,
            "nc":    base_cfg["nc"],
            "names": base_cfg["names"],
        }
        tmp_yaml = tmp_dir / f"light_{code}.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            _yaml.dump(tmp_cfg, f, allow_unicode=True, default_flow_style=False)

        try:
            val_res = model.val(
                data    = str(tmp_yaml),
                split   = "test",
                batch   = args.batch,
                imgsz   = args.imgsz,
                device  = args.device,
                verbose = False,
                plots   = False,
                project = str(tmp_dir),
                name    = f"val_light_{code}",
                exist_ok= True,
            )
            rd = val_res.results_dict
            map50   = round(float(rd.get("metrics/mAP50(B)",    0)), 4)
            map5095 = round(float(rd.get("metrics/mAP50-95(B)", 0)), 4)
            prec    = round(float(rd.get("metrics/precision(B)", 0)), 4)
            rec     = round(float(rd.get("metrics/recall(B)",    0)), 4)
            print(f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            map50 = map5095 = prec = rec = 0.0

        rows.append({
            "light_code": code,
            "light_name": light_name,
            "n_images":   n,
            "mAP50":      map50,
            "mAP50_95":   map5095,
            "precision":  prec,
            "recall":     rec,
        })

    # Clean temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Save CSV
    csv_path = out_dir / f"per_lighting_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["light_code", "light_name", "n_images",
                      "mAP50", "mAP50_95", "precision", "recall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [METRICS] Per-lighting CSV saved: {csv_path.name}")

    return rows


def write_summary_report(run_dir: Path, run_name: str, timestamp: str,
                          args, train_cfg: dict,
                          test_results: dict, per_class_rows: list,
                          timing: dict, lighting_rows: list = None,
                          compute_cost: dict = None):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram     = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) \
               if torch.cuda.is_available() else 0

    lines = [
        "=" * 65,
        "  BASELINE TRAINING REPORT",
        "=" * 65,
        f"  Run Name   : {run_name}",
        f"  Timestamp  : {timestamp}",
        f"  Model      : {args.model} (COCO pretrained, fine-tuned on ExDark)",
        f"  Dataset    : {args.data}",
        f"  GPU        : {gpu_name}  |  VRAM: {vram} GB",
        f"  Epochs     : {train_cfg['epochs']}  |  Batch: {train_cfg['batch']}  "
        f"|  imgsz: {train_cfg['imgsz']}  |  AMP: {train_cfg['amp']}",
        f"  Optimizer  : {train_cfg['optimizer']}  |  lr0: {train_cfg['lr0']}  "
        f"|  patience: {train_cfg['patience']}",
        "",
        "-" * 65,
        "  COMPUTATIONAL TIME",
        "-" * 65,
        f"  Phase 1  Training        : {timing.get('phase1_train_min',      0):.2f} menit",
        f"  Phase 2  Copy weights    : {timing.get('phase2_copy_min',        0):.2f} menit",
        f"  Phase 3  Metrics & plots : {timing.get('phase3_metrics_min',     0):.2f} menit",
        f"  Phase 4  Test eval       : {timing.get('phase4_test_eval_min',   0):.2f} menit",
        f"  Phase 4b Lighting eval   : {timing.get('phase4b_lighting_min',   0):.2f} menit",
        f"  Phase 5  Qualitative     : {timing.get('phase5_qualitative_min', 0):.2f} menit",
        f"  ─────────────────────────────────────────",
        f"  Total                    : {timing.get('total_min',              0):.2f} menit",
        "",
        "-" * 65,
        "  TEST SET RESULTS",
        "-" * 65,
        f"  mAP@0.5        : {test_results.get('mAP50',    0):.4f}",
        f"  mAP@0.5:0.95   : {test_results.get('mAP50_95', 0):.4f}",
        f"  Precision      : {test_results.get('precision', 0):.4f}",
        f"  Recall         : {test_results.get('recall',    0):.4f}",
        "",
    ]
    if per_class_rows:
        lines += ["-" * 65, "  PER-CLASS AP@0.5  (urut dari tertinggi)", "-" * 65]
        for row in sorted(per_class_rows, key=lambda x: -x["AP50"]):
            lines.append(
                f"  {row['class_name']:12s}  "
                f"AP50: {row['AP50']:.4f}  |  AP50-95: {row['AP50_95']:.4f}"
            )

    if lighting_rows:
        lines += ["", "-" * 65, "  PER-LIGHTING CONDITION AP@0.5", "-" * 65]
        for row in lighting_rows:
            lines.append(
                f"  [{row['light_code']:2d}] {row['light_name']:12s}  "
                f"n={row['n_images']:4d}  "
                f"mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50_95']:.4f}"
            )

    if compute_cost:
        lines += [
            "",
            "-" * 65,
            "  COMPUTATIONAL COST",
            "-" * 65,
            f"  Parameters        : {compute_cost.get('params_M', 0):.3f} M",
            f"  GFLOPs            : {compute_cost.get('gflops', 0):.1f}",
            f"  Model size        : {compute_cost.get('model_size_mb', 0):.1f} MB",
            f"  Inference speed   : {compute_cost.get('inference_ms_per_image', 0):.3f} ms/image",
            f"  Preprocess        : {compute_cost.get('preprocess_ms', 0):.3f} ms/image",
            f"  Postprocess       : {compute_cost.get('postprocess_ms', 0):.3f} ms/image",
            f"  FPS               : {compute_cost.get('fps', 0):.1f}",
            f"  Peak VRAM (train) : {compute_cost.get('peak_vram_train_mb', 0):.1f} MB",
        ]

    lines += [
        "",
        "-" * 65,
        "  OUTPUT FILES",
        "-" * 65,
        f"  weights/best.pt",
        f"  weights/last.pt",
        f"  metrics/config_{timestamp}.json",
        f"  metrics/per_epoch_{timestamp}.csv",
        f"  metrics/test_results_{timestamp}.json",
        f"  metrics/per_class_ap_test_{timestamp}.csv",
        f"  metrics/per_lighting_{timestamp}.csv",
        f"  metrics/compute_cost_{timestamp}.json",
        f"  metrics/timing_{timestamp}.json",
        f"  plots/loss_and_map_curves_{timestamp}.png",
        f"  plots/confusion_matrix.png",
        f"  plots/PR_curve.png  |  F1_curve.png",
        f"  qualitative/sample_detections/  ({args.qualitative} gambar)",
        "=" * 65,
    ]

    out = run_dir / "report" / f"summary_{timestamp}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Summary saved: {out.name}")
    print("\n" + "\n".join(lines))


# ============================================================
# YAML PATH FIXER
# ============================================================

def fix_yaml_path(yaml_path: str) -> str:
    
    import re
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Dataset YAML tidak ditemukan: {yaml_path}")

    correct_path = str(yaml_file.parent).replace("\\", "/")

    with open(yaml_file, "r") as f:
        content = f.read()

    
    new_content = re.sub(r"^path\s*:.*$", f"path: {correct_path}", content, flags=re.MULTILINE)

    if new_content != content:
        with open(yaml_file, "w") as f:
            f.write(new_content)
        print(f"[YAML] Path diperbarui -> {correct_path}")
    else:
        print(f"[YAML] Path sudah benar: {correct_path}")

    return yaml_path


# ============================================================
# MAIN
# ============================================================

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[SEED] Reproducibility seed = {seed}")


def main():
    args = parse_args()
    set_seed(args.seed)
    timestamp, run_name, run_dir, train_cfg = build_run_context(args)

    print("=" * 65)
    print("  BASELINE YOLOv8 — ExDark Training")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}  |  Batch: {args.batch}  |  AMP: {args.amp}")
    print(f"  Run      : {run_name}")
    print("=" * 65)

    
    fix_yaml_path(args.data)

    make_dirs(run_dir)
    save_config_snapshot(run_dir, train_cfg, args, timestamp, run_name)

    # ── 1. TRAINING ──────────────────────────────────────────
    print("\n[PHASE 1] Training...")
    t_total = time.time()
    t_start = t_total
    model = YOLO(args.model)

    
    def _fitness_map50(trainer):
        trainer.fitness = trainer.metrics.get("metrics/mAP50(B)", 0.0)

    model.add_callback("on_fit_epoch_end", _fitness_map50)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train(**train_cfg)
    elapsed_min = (time.time() - t_start) / 60
    peak_vram_train_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1) \
                         if torch.cuda.is_available() else 0.0
    print(f"\n[PHASE 1] Selesai dalam {elapsed_min:.1f} menit.")

    
    train_dir = Path(model.trainer.save_dir)
    print(f"[INFO] Training output: {train_dir}")

    # ── 2. WEIGHTS ─────────────────────────────────────
    print("\n[PHASE 2] Copying weights...")
    t_p2 = time.time()
    for wname in ["best.pt", "last.pt"]:
        src = train_dir / "weights" / wname
        dst = run_dir / "weights" / wname
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                print(f"  Copied: {wname}")
            else:
                print(f"  Already in place: {wname}")
    best_pt = run_dir / "weights" / "best.pt"
    elapsed_p2 = (time.time() - t_p2) / 60

    # ── 3. METRICS & PLOTS ───────────────────────────────────
    print("\n[PHASE 3] Saving metrics and plots...")
    t_p3 = time.time()
    metrics_csv = copy_training_metrics(train_dir, run_dir, timestamp)
    copy_plots(train_dir, run_dir)
    plot_loss_curves(metrics_csv, run_dir, run_name, timestamp)
    elapsed_p3 = (time.time() - t_p3) / 60

    # ── 4. TEST EVALUATION ───────────────────────────────────
    print("\n[PHASE 4] Evaluating on test set...")
    t_p4 = time.time()
    best_model  = YOLO(str(best_pt))
    val_result  = best_model.val(
        data    = args.data,
        split   = "test",
        batch   = args.batch,
        imgsz   = args.imgsz,
        device  = args.device,
        verbose = False,
        project = str(run_dir / "metrics"),
        name    = f"test_eval_{timestamp}",
    )
    test_results   = save_test_results(val_result.results_dict, run_dir, run_name, timestamp)
    per_class_rows = save_per_class_ap(val_result, run_dir, timestamp)
    elapsed_p4 = (time.time() - t_p4) / 60

    
    n_params = sum(p.numel() for p in best_model.model.parameters())
    try:
        _info  = best_model.model.info(verbose=False)
        gflops = float(_info[3]) if isinstance(_info, (list, tuple)) and len(_info) > 3 and _info[3] else 0.0
    except Exception:
        gflops = 0.0
    if gflops == 0.0:
        try:
            import thop
            from copy import deepcopy
            _dev   = next(best_model.model.parameters()).device
            _dummy = torch.zeros(1, 3, 640, 640, device=_dev)
            _macs, _ = thop.profile(deepcopy(best_model.model), inputs=(_dummy,), verbose=False)
            gflops = round(_macs * 2 / 1e9, 1)
        except Exception:
            gflops = 0.0
    _speed        = getattr(val_result, "speed", {})
    inference_ms  = round(_speed.get("inference",   0.0), 3)
    preprocess_ms = round(_speed.get("preprocess",  0.0), 3)
    postprocess_ms= round(_speed.get("postprocess", 0.0), 3)
    compute_cost  = {
        "params_M":               round(n_params / 1e6, 3),
        "gflops":                 round(gflops, 1),
        "model_size_mb":          round(best_pt.stat().st_size / 1024**2, 1),
        "inference_ms_per_image": inference_ms,
        "preprocess_ms":          preprocess_ms,
        "postprocess_ms":         postprocess_ms,
        "fps":                    round(1000.0 / inference_ms, 1) if inference_ms > 0 else 0.0,
        "peak_vram_train_mb":     peak_vram_train_mb,
    }
    compute_cost_path = run_dir / "metrics" / f"compute_cost_{timestamp}.json"
    compute_cost_path.write_text(json.dumps(compute_cost, indent=2), encoding="utf-8")
    print(f"[COMPUTE] Saved: {compute_cost_path.name}")

    # ── 4b. PER-LIGHTING EVALUATION ──────────────────────────
    print("\n[PHASE 4b] Per-lighting condition evaluation...")
    t_p4b = time.time()
    data_yaml    = Path(args.data).resolve()
    test_img_dir = data_yaml.parent / "images" / "test"
    lighting_rows = eval_by_lighting(
        model_path   = best_pt,
        test_img_dir = test_img_dir,
        data_yaml    = data_yaml,
        annno_txt    = ANNNO_TXT,
        run_dir      = run_dir,
        timestamp    = timestamp,
        args         = args,
    )
    elapsed_p4b = (time.time() - t_p4b) / 60

    # ── 5. QUALITATIVE ───────────────────────────────────────
    print("\n[PHASE 5] Qualitative detection samples...")
    t_p5 = time.time()
    save_qualitative_detections(best_pt, run_dir, timestamp, args, args.data)
    elapsed_p5 = (time.time() - t_p5) / 60

    # ── 6. REPORT ────────────────────────────────────────────
    elapsed_total = (time.time() - t_total) / 60
    timing = {
        "phase1_train_min":      round(elapsed_min,   2),
        "phase2_copy_min":       round(elapsed_p2,    2),
        "phase3_metrics_min":    round(elapsed_p3,    2),
        "phase4_test_eval_min":  round(elapsed_p4,    2),
        "phase4b_lighting_min":  round(elapsed_p4b,   2),
        "phase5_qualitative_min":round(elapsed_p5,    2),
        "total_min":             round(elapsed_total,  2),
    }
    timing_path = run_dir / "metrics" / f"timing_{timestamp}.json"
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"\n[PHASE 6] Writing summary report...")
    print(f"[TIMING] Saved: {timing_path.name}")
    write_summary_report(run_dir, run_name, timestamp, args, train_cfg,
                         test_results, per_class_rows, timing,
                         lighting_rows=lighting_rows, compute_cost=compute_cost)

    print(f"\n✓ Output saved in:\n  {run_dir}")


if __name__ == "__main__":
    main()
