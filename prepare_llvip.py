"""
prepare_llvip.py — Prepare LLVIP dataset for YOLOv8 training.

Mendukung dua layout Annotations yang umum:
  Layout A (flat)   : LLVIP/Annotations/*.xml  (struktur asli official)
  Layout B (nested) : LLVIP/Annotations/train/*.xml  +  /test/*.xml

Steps performed:
  1. Convert Pascal VOC XML annotations → YOLO TXT format
  2. Keep official test set intact (3,463 images)
  3. Split official train (12,025) → train (~10,823) + val (~1,202) [seed=42]
  4. Copy visible images to output folder
  5. Generate datasets/LLVIP/LLVIP.yaml

Usage:
  python prepare_llvip.py --src datasets/LLVIP
  python prepare_llvip.py --src datasets/LLVIP --dst datasets/LLVIP_Official

Output structure:
  datasets/LLVIP_Official/
  ├── images/
  │   ├── train/   (~10,823 jpg)
  │   ├── val/     (~1,202 jpg)
  │   └── test/    (3,463 jpg — official)
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── LLVIP.yaml
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

VAL_RATIO = 0.10
SEED      = 42
CLASS_MAP = {"person": 0}
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ─── VOC XML → YOLO TXT ───────────────────────────────────────────────────────

def voc_to_yolo(xml_path: Path) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name   = obj.find("name").text.strip().lower()
        cls_id = CLASS_MAP.get(name)
        if cls_id is None:
            continue

        bb   = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        xc = max(0.0, min(1.0, (xmin + xmax) / 2 / w))
        yc = max(0.0, min(1.0, (ymin + ymax) / 2 / h))
        bw = max(0.0, min(1.0, (xmax - xmin) / w))
        bh = max(0.0, min(1.0, (ymax - ymin) / h))

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


# ─── Detect folder structure ───────────────────────────────────────────────────

def detect_layout(src: Path):
    """
    Layout folder LLVIP detection and return:
      (train_img_dir, test_img_dir, ann_dir, ann_mode)
    ann_mode: "flat"   → Annotations/*.xml  (semua XML di satu folder)
              "nested" → Annotations/train/*.xml + Annotations/test/*.xml
    """
    # Find visible images folder (can be "visible" or "images")
    train_img = test_img = None
    for img_root in ["visible", "images"]:
        if (src / img_root / "train").exists() and (src / img_root / "test").exists():
            train_img = src / img_root / "train"
            test_img  = src / img_root / "test"
            break

    if train_img is None:
        return None, None, None, None

    ann_root = src / "Annotations"
    if not ann_root.exists():
        return None, None, None, None

    # Check is flat or nested
    has_nested = (ann_root / "train").exists() and (ann_root / "test").exists()
    ann_mode   = "nested" if has_nested else "flat"

    return train_img, test_img, ann_root, ann_mode


# ─── Process one split ─────────────────────────────────────────────────────────

def process_split(src_img_dir: Path, ann_dir: Path, ann_mode: str,
                  dst_img_dir: Path, dst_lbl_dir: Path,
                  stems: list, split_name: str) -> int:
    """
    ann_mode "flat"   → search XML in ann_dir/*.xml
    ann_mode "nested" →  search XML in ann_dir/<split_name>/*.xml
                        (split_name: "train" or "test")
    Val uses ann_dir "train" (from pool train)
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if ann_mode == "nested":
        # val from pool train, ann subfolder = "train"
        ann_sub = "test" if split_name == "test" else "train"
        xml_root = ann_dir / ann_sub
    else:
        xml_root = ann_dir   # flat: all XML in one folder

    ok = skip_img = skip_ann = 0
    for stem in stems:
        # search image files
        img_src = next(
            (src_img_dir / (stem + ext) for ext in IMG_EXTS
             if (src_img_dir / (stem + ext)).exists()), None
        )
        if img_src is None:
            skip_img += 1
            continue

        xml_src = xml_root / (stem + ".xml")
        if not xml_src.exists():
            skip_ann += 1
            continue

        shutil.copy2(img_src, dst_img_dir / img_src.name)
        yolo_lines = voc_to_yolo(xml_src)
        (dst_lbl_dir / (stem + ".txt")).write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )
        ok += 1

    if skip_img:
        print(f"  [WARN] {split_name}: {skip_img} image not found")
    if skip_ann:
        print(f"  [WARN] {split_name}: {skip_ann} XML annotation not found")

    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare LLVIP dataset for YOLOv8 training"
    )
    parser.add_argument("--src", required=True,
                        help="Path to root folder LLVIP that have been downloaded")
    parser.add_argument("--dst", default="datasets/LLVIP_Official",
                        help="Path output (default: datasets/LLVIP_Official)")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst)

    # ── Deteksi layout ──
    train_img_dir, test_img_dir, ann_dir, ann_mode = detect_layout(src)

    if train_img_dir is None:
        print(f"\nERROR: Can not detect the folder structure in: {src}")
        print("Make sure the folder contains:")
        print("  visible/train/  dan  visible/test/  (images)")
        print("  Annotations/                        (XML files)")
        return

    print(f"\n  Structure detected:")
    print(f"    Images     : {train_img_dir.parent.name}/train | test")
    print(f"    Annotations: {ann_mode.upper()}  "
          f"({'Annotations/*.xml' if ann_mode == 'flat' else 'Annotations/train/ + test/'})")

    # ── Collect images stems  ──
    all_train_stems = sorted(
        p.stem for p in train_img_dir.iterdir() if p.suffix in IMG_EXTS
    )
    all_test_stems = sorted(
        p.stem for p in test_img_dir.iterdir() if p.suffix in IMG_EXTS
    )

    if not all_train_stems:
        print(f"ERROR: No images in {train_img_dir}")
        return

    # ── Split train → train + val ──
    rng = random.Random(args.seed)
    shuffled = all_train_stems[:]
    rng.shuffle(shuffled)

    n_val       = round(len(shuffled) * args.val_ratio)
    val_stems   = sorted(shuffled[:n_val])
    train_stems = sorted(shuffled[n_val:])
    test_stems  = all_test_stems
    total       = len(train_stems) + len(val_stems) + len(test_stems)

    print(f"\n{'='*60}")
    print(f"  LLVIP Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Source  : {src}")
    print(f"  Output  : {dst.resolve()}")
    print(f"  Seed    : {args.seed}  |  Val ratio : {args.val_ratio:.0%}")
    print(f"\n  Rencana split:")
    print(f"    Train : {len(train_stems):>6,}  ({len(train_stems)/total*100:.1f}%)")
    print(f"    Val   : {len(val_stems):>6,}  ({len(val_stems)/total*100:.1f}%)  ← from official train")
    print(f"    Test  : {len(test_stems):>6,}  ({len(test_stems)/total*100:.1f}%)  ← official")
    print(f"    Total : {total:>6,}")

    # ── Proses setiap split ──
    print(f"\n  Training Process ...")
    n_tr = process_split(train_img_dir, ann_dir, ann_mode,
                         dst/"images"/"train", dst/"labels"/"train",
                         train_stems, "train")

    print(f"  Val Process ...")
    n_vl = process_split(train_img_dir, ann_dir, ann_mode,
                         dst/"images"/"val", dst/"labels"/"val",
                         val_stems, "val")

    print(f"  Test Process ...")
    n_te = process_split(test_img_dir, ann_dir, ann_mode,
                         dst/"images"/"test", dst/"labels"/"test",
                         test_stems, "test")

    # ── Write LLVIP.yaml ──
    yaml_content = (
        "# LLVIP Dataset - Low-Light Visible-Infrared Paired\n"
        "# Paper: Jia et al., ICCV 2021 Workshop\n"
        "# Source: https://github.com/bupt-ai-cz/LLVIP\n"
        "# Visible images only, single class: person\n"
        f"# Split: train={n_tr} | val={n_vl} | test={n_te} (official test preserved)\n"
        "#\n"
        "path: .\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    yaml_path = dst / "LLVIP.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Finish!")
    print(f"    Train : {n_tr:>6,} images + labels")
    print(f"    Val   : {n_vl:>6,} images + labels")
    print(f"    Test  : {n_te:>6,} images + labels  (official)")
    print(f"    YAML  : {yaml_path.resolve()}")
    yaml_abs = yaml_path.resolve().as_posix()
    print(f"\n  Baseline Training Command (YOLOv8s):")
    print(f"    python train_baseline.py --model yolov8s.pt \\")
    print(f"      --data {yaml_abs} \\")
    print(f"      --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \\")
    print(f"      --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \\")
    print(f"      --patience 50 --workers 0 --device 0 --amp --seed 42")
    print(f"\n  DAEG-YOLO training command:")
    print(f"    python train_daeg_yolo.py --model yolov8s.pt \\")
    print(f"      --yaml models/yolov8s_daeg_yolo.yaml \\")
    print(f"      --data {yaml_abs} \\")
    print(f"      --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \\")
    print(f"      --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \\")
    print(f"      --patience 50 --lambda-dael 0.01 --w-tv 0.1 --w-bright 1.0 \\")
    print(f"      --dael-mode fpn --dael-warmup-epochs 5 --fpn-layer 15 \\")
    print(f"      --dafm-mode full --lambda-recon 1.0 --recon-threshold 0.10 \\")
    print(f"      --recon-k 50.0 --workers 0 --device 0 --amp --seed 42")
    print(f"\n  Notes: Per-lighting eval will skipped (LLVIP has no lighting labels).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
