# DAEG-YOLO: Detection-Aware Enhancement Guided YOLO for Low-Light Object Detection

Official implementation of **DAEG-YOLO**, a low-light object detection framework that jointly learns image enhancement and detection through a detection-aware feedback mechanism.

> **Paper:** *DAEG-YOLO: Detection-Aware Enhancement Guided YOLO for Low-Light Object Detection*
> Submitted to the International Journal of Intelligent Engineering and Systems (IJIES), 2026.

---

## Overview

DAEG-YOLO addresses low-light object detection by coupling a lightweight image enhancer (DCENet9) with the YOLOv8s detector through two novel components:

- **DAFM (Detection-Aware Feature Modulator)** — inserted at the P3-FPN layer (Layer 15) of YOLOv8s. DAFM applies channel SE-attention and spatial attention to refine detection features, then exports a *learned* spatial importance map to guide the enhancement loss. This replaces the heuristic activation importance used in prior work.
- **FPN-DAEL (FPN-guided Detection-Aware Enhancement Loss)** — supervises DCENet9 to enhance regions that matter for detection, using the importance map exported by DAFM (warmup: 5 epochs).
- **CondRecon (Conditional Reconstruction Loss)** — constrains the enhancer to behave as identity on already-bright images, preventing over-brightening.

The YOLOv8s backbone is **unchanged** and loads pretrained weights directly. Only DAFM (6,561 new parameters) and DCENet9 (0.081M parameters) are trained from scratch.

```
Total parameters: 11.23M  (YOLOv8s: 11.14M + DAFM: 6,561 + DCENet9: 0.081M)
```

### Architecture

```
Input Image
    │
    ├─► DCENet9 (9-iter curve estimation) ──► Enhanced Image ──► YOLOv8s Backbone
    │        ▲                                                         │
    │        │   FPN-DAEL (importance-guided enhancement loss)         │
    │        │                                                    FPN Head
    │        └────────────────────────────────────────────── DAFM @ Layer 15
    │                                                    (feature mod + importance export)
    └─────────────────────────────────────────────────────────────────► Detection Head
```

---

## Results

### ExDark (12-class low-light detection)

| Model | Params | mAP50 | mAP50:95 |
|---|---|---|---|
| YOLOv8s Baseline | 11.14M | 0.7919 | — |
| **DAEG-YOLO** | **11.23M** | **0.8035** | — |
| Δ | +0.09M | **+0.0116** | — |

### Generalization

| Dataset | Baseline mAP50 | DAEG-YOLO mAP50 | Δ |
|---|---|---|---|
| LLVIP (person, nighttime) | 0.9106 | **0.9138** | +0.0032 |
| DarkFace (face, nighttime) | 0.5642 | **0.5825** | +0.0183 |

---

## Repository Structure

```
daeg-yolo/
├── ultralytics/              # Modified Ultralytics fork (DAFMC2f registered in block.py)
├── losses/
│   ├── dael_loss.py          # StandardDAEL loss
│   └── fpn_dael_loss.py      # FPNAwareDAEL — main DAEL variant used in DAEG-YOLO
├── models/
│   └── yolov8s_daeg_yolo.yaml  # DAEG-YOLO architecture (Layer 15 = DAFMC2f)
├── figures/                  # Paper figures (PNG)
├── train_daeg_yolo.py        # Training on ExDark
├── train_daeg_yolo_llvip.py  # Training on LLVIP
├── train_daeg_yolo_darkface.py  # Training on DarkFace
├── train_baseline.py         # YOLOv8s baseline (ExDark)
├── train_baseline_yolo_llvip.py
├── train_baseline_yolo_darkface.py
├── convert_exdark_to_yolo.py   # ExDark → YOLO format converter
├── prepare_llvip.py            # LLVIP dataset preparation
├── prepare_darkface.py         # DarkFace dataset preparation
├── measure_fps_distribution.py # FPS measurement per lighting condition
├── visualize_generalization.py # Generalization result figures
├── visualize_qualitative.py    # Qualitative comparison figures
├── make_combined_gen_figure.py # Combined generalization figure
└── requirements_training.txt
```

---

## Installation

**Step 1 — Install PyTorch** (CUDA 12.x recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Step 2 — Install dependencies:**

```bash
pip install -r requirements_training.txt
```

> The `ultralytics/` directory in this repo is a **modified local fork**. Do not install `ultralytics` separately — the local copy will be used automatically via `sys.path` insertion in each training script.

---

## Dataset Preparation

### ExDark

Download [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) and place it under `datasets/ExDark_Official/`. Then run:

```bash
python convert_exdark_to_yolo.py
```

This produces `datasets/ExDark_Official/ExDark.yaml` and YOLO-format annotation files.

ExDark contains **12 classes**: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table — all captured under low-light conditions.

### LLVIP

Download [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) and place it under `datasets/LLVIP_Official/`. Then run:

```bash
python prepare_llvip.py
```

### DarkFace

Download [DarkFace](https://flyywh.github.io/CVPRW2019LowLight/) and place it under `datasets/DarkFace_Official/`. Then run:

```bash
python prepare_darkface.py
```

---

## Training

### DAEG-YOLO on ExDark

```bash
python train_daeg_yolo.py \
    --model yolov8s.pt \
    --yaml models/yolov8s_daeg_yolo.yaml \
    --data datasets/ExDark_Official/ExDark.yaml \
    --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \
    --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --patience 50 --lambda-dael 0.01 --w-tv 0.1 --w-bright 1.0 \
    --dael-mode fpn --dael-warmup-epochs 5 --fpn-layer 15 \
    --dafm-mode full \
    --lambda-recon 1.0 --recon-threshold 0.10 --recon-k 50.0 \
    --workers 0 --device 0 --amp --seed 42 --project results
```

### DAEG-YOLO on LLVIP

```bash
python train_daeg_yolo_llvip.py \
    --model yolov8s.pt \
    --yaml models/yolov8s_daeg_yolo.yaml \
    --data datasets/LLVIP_Official/LLVIP.yaml \
    --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \
    --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --patience 50 --lambda-dael 0.01 --w-tv 0.1 --w-bright 1.0 \
    --dael-mode fpn --dael-warmup-epochs 5 --fpn-layer 15 \
    --dafm-mode full \
    --lambda-recon 1.0 --recon-threshold 0.10 --recon-k 50.0 \
    --workers 0 --device 0 --amp --seed 42 --project results
```

### DAEG-YOLO on DarkFace

```bash
python train_daeg_yolo_darkface.py \
    --model yolov8s.pt \
    --yaml models/yolov8s_daeg_yolo.yaml \
    --data datasets/DarkFace_Official/DarkFace.yaml \
    --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \
    --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --patience 50 --lambda-dael 0.01 --w-tv 0.1 --w-bright 1.0 \
    --dael-mode fpn --dael-warmup-epochs 5 --fpn-layer 15 \
    --dafm-mode full \
    --lambda-recon 1.0 --recon-threshold 0.10 --recon-k 50.0 \
    --workers 0 --device 0 --amp --seed 42 --project results
```

### YOLOv8s Baseline (ExDark)

```bash
python train_baseline.py \
    --model yolov8s.pt \
    --data datasets/ExDark_Official/ExDark.yaml \
    --epochs 200 --batch 12 --imgsz 640 --optimizer SGD \
    --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --patience 50 --workers 0 --device 0 --amp --seed 42 --project results
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `--lambda-dael` | 0.01 | Weight for FPN-DAEL loss |
| `--w-tv` | 0.1 | TV smoothness weight in DCENet9 loss |
| `--w-bright` | 1.0 | Brightness loss weight in DCENet9 loss |
| `--dael-mode` | `fpn` | Use FPN-guided importance (vs `standard`) |
| `--dael-warmup-epochs` | 5 | Epochs before activating DAEL |
| `--fpn-layer` | 15 | Layer index of DAFM (P3-FPN) |
| `--dafm-mode` | `full` | DAFM feature-mod + learned importance |
| `--lambda-recon` | 1.0 | Conditional reconstruction loss weight |
| `--recon-threshold` | 0.10 | Brightness threshold for CondRecon |

---

## Ablation: `--dafm-mode`

| Mode | Description | ExDark mAP50 |
|---|---|---|
| `full` | DAFM feature-mod + learned importance → FPN-DAEL | **0.8035** |
| `heuristic` | DAFM feature-mod + heuristic importance → FPN-DAEL | 0.7901 |
| `none` | No DAFM (use standard YOLOv8s C2f at Layer 15) | 0.7956 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{jiskandar2026daegyolo,
  title   = {DAEG-YOLO: Detection-Aware Enhancement Guided YOLO for Low-Light Object Detection},
  author  = {Iskandar, Riyadi Jimmy},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026}
}
```

This implementation is built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Please also cite:

```bibtex
@software{jocher2023ultralytics,
  author  = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  title   = {Ultralytics YOLO},
  version = {8.0.0},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

---

## License

This project is licensed under the **AGPL-3.0 License** — see [LICENSE](LICENSE) for details.
This license is inherited from Ultralytics YOLOv8, which this project extends.
