"""
ExDark to YOLOv8 Format Converter
Converts official ExDark dataset (bbGt v3 format) to YOLOv8 YOLO format.

Source:
  ExDark/          -> images organized by class folder
  ExDark_Annno/    -> annotations in bbGt v3 format (Caltech format)

Output:
  datasets/ExDark_Official/
    images/train/, images/val/, images/test/
    labels/train/, labels/val/, labels/test/
    ExDark.yaml

Annotation format (bbGt v3):
  % bbGt version=3
  ClassName x_topleft y_topleft width height [occluded ignore ...]

YOLO format:
  class_id cx_norm cy_norm w_norm h_norm  (all normalized 0-1)
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

# ==================== CONFIG ====================
SRC_IMAGES = Path(r"D:\BaseCode\ProjectResNetDCE1-main\ExDark")
SRC_ANNNO  = Path(r"D:\BaseCode\ProjectResNetDCE1-main\ExDark_Annno")
DST_ROOT   = Path(r"D:\BaseCode\ProjectResNetDCE1-main\datasets\ExDark_Official")

SEED       = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO = remaining 0.10

CLASS_NAMES = [
    "Bicycle",    # 0
    "Boat",       # 1
    "Bottle",     # 2
    "Bus",        # 3
    "Car",        # 4
    "Cat",        # 5
    "Chair",      # 6
    "Cup",        # 7
    "Dog",        # 8
    "Motorbike",  # 9
    "People",     # 10
    "Table",      # 11
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ==================== HELPERS ====================

def get_image_size(img_path: Path):
    """Return (width, height) of image."""
    with Image.open(img_path) as img:
        return img.width, img.height


def parse_bbgt(anno_path: Path):
    """
    Parse bbGt v3 annotation file.
    Returns list of (class_name, x_topleft, y_topleft, w, h).
    Skips comment lines and unknown classes.
    """
    boxes = []
    if not anno_path.exists():
        return boxes
    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_name = parts[0]
            if class_name not in CLASS_MAP:
                print(f"  [WARN] Unknown class '{class_name}' in {anno_path.name}, skipping.")
                continue
            try:
                x = int(parts[1])
                y = int(parts[2])
                w = int(parts[3])
                h = int(parts[4])
                if w <= 0 or h <= 0:
                    continue
                boxes.append((class_name, x, y, w, h))
            except ValueError:
                continue
    return boxes


def convert_to_yolo(boxes, img_w, img_h):
    """
    Convert list of (class_name, x_tl, y_tl, w, h) to YOLO format lines.
    YOLO: class_id cx_norm cy_norm w_norm h_norm
    """
    lines = []
    for class_name, x, y, w, h in boxes:
        class_id = CLASS_MAP[class_name]
        # Clamp bounding box to image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        w_n = w / img_w
        h_n = h / img_h
        # Clamp to [0, 1]
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w_n = min(max(w_n, 0.0), 1.0)
        h_n = min(max(h_n, 0.0), 1.0)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}")
    return lines


def write_yaml(dst_root: Path, class_names: list):
    """Write ExDark.yaml config for YOLOv8."""
    yaml_content = f"""# ExDark Dataset - Official 7363 images, 12 classes
# Converted from bbGt v3 format to YOLOv8 format
# Source: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
# path menggunakan '.' (relatif ke lokasi file YAML ini) agar portabel lintas komputer

path: .
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = dst_root / "ExDark.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[YAML] Written: {yaml_path}")


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("  ExDark to YOLOv8 Converter")
    print("=" * 60)

    random.seed(SEED)

    # Collect all (image_path, annotation_path, primary_class) tuples
    print("\n[1/5] Collecting image-annotation pairs...")
    all_pairs = []
    missing_anno = 0
    missing_img_size = 0

    for class_folder in sorted(SRC_IMAGES.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in CLASS_MAP:
            print(f"  [SKIP] Unknown class folder: {class_name}")
            continue

        anno_folder = SRC_ANNNO / class_name
        for img_file in sorted(class_folder.iterdir()):
            if img_file.suffix not in IMAGE_EXTS:
                continue
            # Annotation filename = image filename + ".txt"
            anno_file = anno_folder / (img_file.name + ".txt")
            if not anno_file.exists():
                missing_anno += 1
                continue
            all_pairs.append((img_file, anno_file, class_name))

    print(f"  Found: {len(all_pairs)} image-annotation pairs")
    if missing_anno:
        print(f"  [WARN] {missing_anno} images had no annotation file (skipped)")

    # Stratified split per class folder (primary class)
    print("\n[2/5] Splitting into train/val/test (80/10/10, seed=42)...")
    train_pairs, val_pairs, test_pairs = [], [], []

    # Group by primary class for stratified split
    from collections import defaultdict
    by_class = defaultdict(list)
    for pair in all_pairs:
        by_class[pair[2]].append(pair)

    for cls, pairs in sorted(by_class.items()):
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train_pairs.extend(pairs[:n_train])
        val_pairs.extend(pairs[n_train:n_train + n_val])
        test_pairs.extend(pairs[n_train + n_val:])
        print(f"  {cls:12s}: {n} total -> {n_train} train / {n_val} val / {n - n_train - n_val} test")

    print(f"\n  TOTAL: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    # Create output directory structure
    print("\n[3/5] Creating output directories...")
    for split in ["train", "val", "test"]:
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"  Output root: {DST_ROOT}")

    # Convert and copy files
    print("\n[4/5] Converting annotations and copying images...")
    stats = {"train": {"ok": 0, "empty": 0, "err": 0},
             "val":   {"ok": 0, "empty": 0, "err": 0},
             "test":  {"ok": 0, "empty": 0, "err": 0}}

    def process_split(pairs, split_name):
        img_dir = DST_ROOT / "images" / split_name
        lbl_dir = DST_ROOT / "labels" / split_name
        for img_path, anno_path, _ in pairs:
            try:
                # Get image dimensions
                img_w, img_h = get_image_size(img_path)

                # Parse annotation
                boxes = parse_bbgt(anno_path)

                # Convert to YOLO format
                yolo_lines = convert_to_yolo(boxes, img_w, img_h)

                # Filenames are unique across all class folders (verified), use as-is
                dst_img = img_dir / img_path.name
                dst_lbl = lbl_dir / (img_path.stem + ".txt")

                # Copy image
                shutil.copy2(img_path, dst_img)

                # Write label (even if empty, YOLO expects the file)
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(yolo_lines))

                if yolo_lines:
                    stats[split_name]["ok"] += 1
                else:
                    stats[split_name]["empty"] += 1

            except Exception as e:
                stats[split_name]["err"] += 1
                print(f"  [ERR] {img_path.name}: {e}")

    process_split(train_pairs, "train")
    process_split(val_pairs,   "val")
    process_split(test_pairs,  "test")

    # Write YAML config
    print("\n[5/5] Writing ExDark.yaml...")
    write_yaml(DST_ROOT, CLASS_NAMES)

    # Final report
    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        s = stats[split]
        total = s["ok"] + s["empty"] + s["err"]
        print(f"  {split:5s}: {total} images | {s['ok']} with labels | {s['empty']} empty labels | {s['err']} errors")

    print(f"\n  Output: {DST_ROOT}")
    print(f"  YAML:   {DST_ROOT / 'ExDark.yaml'}")
    print("\n  Class mapping used:")
    for name, idx in CLASS_MAP.items():
        print(f"    {idx:2d}: {name}")

    # Verify final counts
    print("\n  Final file counts:")
    for split in ["train", "val", "test"]:
        n_img = len(list((DST_ROOT / "images" / split).iterdir()))
        n_lbl = len(list((DST_ROOT / "labels" / split).iterdir()))
        match = "OK" if n_img == n_lbl else "MISMATCH!"
        print(f"    {split:5s}: {n_img} images, {n_lbl} labels [{match}]")


if __name__ == "__main__":
    main()
