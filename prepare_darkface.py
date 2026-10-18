"""
prepare_darkface.py -- Prepare DarkFace dataset for YOLOv8 training.

Dataset: DARK FACE - Face Detection in Low Light Condition
Paper  : Yang et al., ECCV 2020 Workshop
Source : https://flyywh.github.io/CVPRW2019LowLight/

Raw label format (per image):
  Line 1 : number of faces
  Line 2+ : x1 y1 x2 y2  (absolute pixel coordinates, top-left & bottom-right)

Steps performed:
  1. Convert raw x1 y1 x2 y2 -> YOLO TXT format (normalized cx cy w h)
  2. Split 6000 images -> train / val / test  [80% / 10% / 10%]
  3. Copy images to output folder structure
  4. Generate datasets/DarkFace_Official/DarkFace.yaml

Usage:
  python prepare_darkface.py --src datasets/DarkFace
  python prepare_darkface.py --src datasets/DarkFace --dst datasets/DarkFace_Official

Output structure:
  datasets/DarkFace_Official/
  +-- images/
  |   +-- train/   (4,800 png)
  |   +-- val/     (  600 png)
  |   +-- test/    (  600 png)
  +-- labels/
  |   +-- train/
  |   +-- val/
  |   +-- test/
  +-- DarkFace.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  = 0.10
SEED        = 42
CLASS_ID    = 0   # single class: face
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --- Raw label -> YOLO TXT ---------------------------------------------------

def darkface_to_yolo(lbl_path: Path, img_w: int, img_h: int) -> list:
    """
    Parse DarkFace raw label (x1 y1 x2 y2 absolute) and return YOLO lines
    (class_id cx cy w h, all normalized 0-1).
    Skips degenerate boxes (zero area).
    """
    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []

    try:
        n_faces = int(lines[0].strip())
    except ValueError:
        return []

    yolo_lines = []
    for line in lines[1: n_faces + 1]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        cx = max(0.0, min(1.0, (x1 + x2) / 2.0 / img_w))
        cy = max(0.0, min(1.0, (y1 + y2) / 2.0 / img_h))
        nw = max(0.0, min(1.0, bw / img_w))
        nh = max(0.0, min(1.0, bh / img_h))

        yolo_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


# --- Process one split -------------------------------------------------------

def process_split(src_img_dir: Path, src_lbl_dir: Path,
                  dst_img_dir: Path, dst_lbl_dir: Path,
                  stems: list, split_name: str) -> int:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    ok = skip_img = skip_lbl = skip_box = 0
    for stem in stems:
        # Find source image
        img_src = next(
            (src_img_dir / (stem + ext) for ext in IMG_EXTS
             if (src_img_dir / (stem + ext)).exists()), None
        )
        if img_src is None:
            skip_img += 1
            continue

        lbl_src = src_lbl_dir / (stem + ".txt")
        if not lbl_src.exists():
            skip_lbl += 1
            continue

        with Image.open(img_src) as im:
            img_w, img_h = im.size

        yolo_lines = darkface_to_yolo(lbl_src, img_w, img_h)
        if not yolo_lines:
            skip_box += 1
            # Still copy image and write empty label (consistent with YOLO convention)

        shutil.copy2(img_src, dst_img_dir / img_src.name)
        (dst_lbl_dir / (stem + ".txt")).write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )
        ok += 1

    if skip_img:
        print(f"  [WARN] {split_name}: {skip_img} gambar tidak ditemukan")
    if skip_lbl:
        print(f"  [WARN] {split_name}: {skip_lbl} label tidak ditemukan")
    if skip_box:
        print(f"  [WARN] {split_name}: {skip_box} gambar tanpa box valid (label kosong)")

    return ok


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DarkFace dataset for YOLOv8 training"
    )
    parser.add_argument("--src", required=True,
                        help="Path ke root folder DarkFace (berisi image/ dan label/)")
    parser.add_argument("--dst", default="datasets/DarkFace_Official",
                        help="Path output (default: datasets/DarkFace_Official)")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--seed",        type=int,   default=SEED)
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst)

    src_img_dir = src / "image"
    src_lbl_dir = src / "label"

    if not src_img_dir.exists():
        print(f"ERROR: Folder gambar tidak ditemukan: {src_img_dir}")
        return
    if not src_lbl_dir.exists():
        print(f"ERROR: Folder label tidak ditemukan: {src_lbl_dir}")
        return

    # Collect all stems
    all_stems = sorted(
        p.stem for p in src_img_dir.iterdir() if p.suffix in IMG_EXTS
    )
    if not all_stems:
        print(f"ERROR: Tidak ada gambar di {src_img_dir}")
        return

    total = len(all_stems)

    # Split
    rng = random.Random(args.seed)
    shuffled = all_stems[:]
    rng.shuffle(shuffled)

    n_val   = round(total * args.val_ratio)
    n_test  = round(total * (1.0 - args.train_ratio - args.val_ratio))
    n_train = total - n_val - n_test

    train_stems = sorted(shuffled[:n_train])
    val_stems   = sorted(shuffled[n_train: n_train + n_val])
    test_stems  = sorted(shuffled[n_train + n_val:])

    print(f"\n{'='*60}")
    print(f"  DarkFace Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Source : {src}")
    print(f"  Output : {dst.resolve()}")
    print(f"  Seed   : {args.seed}")
    print(f"\n  Rencana split:")
    print(f"    Train : {len(train_stems):>5,}  ({len(train_stems)/total*100:.1f}%)")
    print(f"    Val   : {len(val_stems):>5,}  ({len(val_stems)/total*100:.1f}%)")
    print(f"    Test  : {len(test_stems):>5,}  ({len(test_stems)/total*100:.1f}%)")
    print(f"    Total : {total:>5,}")

    print(f"\n  Memproses train ...")
    n_tr = process_split(src_img_dir, src_lbl_dir,
                         dst / "images" / "train", dst / "labels" / "train",
                         train_stems, "train")

    print(f"  Memproses val ...")
    n_vl = process_split(src_img_dir, src_lbl_dir,
                         dst / "images" / "val", dst / "labels" / "val",
                         val_stems, "val")

    print(f"  Memproses test ...")
    n_te = process_split(src_img_dir, src_lbl_dir,
                         dst / "images" / "test", dst / "labels" / "test",
                         test_stems, "test")

    # Write YAML
    yaml_content = (
        "# DarkFace Dataset - Face Detection in Low Light Condition\n"
        "# Paper: Yang et al., ECCV 2020 Workshop\n"
        "# Source: https://flyywh.github.io/CVPRW2019LowLight/\n"
        "# Single class: face | Resolution: 1080x720\n"
        f"# Split: train={n_tr} | val={n_vl} | test={n_te} | seed={args.seed}\n"
        "#\n"
        "path: .\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        "nc: 1\n"
        "names: ['face']\n"
    )
    yaml_path = dst / "DarkFace.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    yaml_abs = yaml_path.resolve().as_posix()

    print(f"\n{'='*60}")
    print(f"  Finish!")
    print(f"    Train : {n_tr:>5,} images + label")
    print(f"    Val   : {n_vl:>5,} images + label")
    print(f"    Test  : {n_te:>5,} images + label")
    print(f"    YAML  : {yaml_path.resolve()}")
    print(f"\n  Baseline training command (YOLOv8s):")
    print(f"    python train_baseline_yolo_darkface.py \\")
    print(f"      --model yolov8s.pt \\")
    print(f"      --data {yaml_abs} \\")
    print(f"      --epochs 200 --batch 12 --imgsz 640 \\")
    print(f"      --optimizer SGD --lr0 0.001 --lrf 0.01 \\")
    print(f"      --momentum 0.937 --weight-decay 0.0005 \\")
    print(f"      --patience 50 --workers 0 --device 0 --amp --seed 42")
    print(f"\n  DAEG-YOLO training command:")
    print(f"    python train_daeg_yolo_darkface.py \\")
    print(f"      --model yolov8s.pt \\")
    print(f"      --yaml models/yolov8s_daeg_yolo.yaml \\")
    print(f"      --data {yaml_abs} \\")
    print(f"      --epochs 200 --batch 12 --imgsz 640 \\")
    print(f"      --optimizer SGD --lr0 0.001 --lrf 0.01 \\")
    print(f"      --momentum 0.937 --weight-decay 0.0005 \\")
    print(f"      --patience 50 \\")
    print(f"      --lambda-dael 0.01 --w-tv 0.1 --w-bright 1.0 \\")
    print(f"      --dael-mode fpn --dael-warmup-epochs 5 --fpn-layer 15 \\")
    print(f"      --dafm-mode full \\")
    print(f"      --lambda-recon 1.0 --recon-threshold 0.10 --recon-k 50.0 \\")
    print(f"      --workers 0 --device 0 --amp --seed 42")
    print(f"\n  Notes: Per-lighting eval will be skip")
    print(f"           (DarkFace has no lighting category labels).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
