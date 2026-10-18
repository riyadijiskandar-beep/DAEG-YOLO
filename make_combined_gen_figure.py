"""
make_combined_gen_figure.py 
====================================================================================

Output:  figures/generalization_combined.{png,jpg,pdf}
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT      = Path(__file__).parent.resolve()
FIG_DIR   = ROOT / "figures"
SRC_LLVIP = FIG_DIR / "generalization_llvip_en.png"
SRC_DARK  = FIG_DIR / "generalization_darkface_en.png"
OUT_BASE  = FIG_DIR / "generalization_combined"

TITLE_FRAC = 0.065  
ROW_FRAC   = 0.25    
TARGET_W   = 3200
LABEL_FONT_SZ = 30
LABEL_GAP     = 6    
SECTION_GAP   = 20   
OUT_DPI       = 200

LABEL_LLVIP = (
    "LLVIP (Person Detection, Nighttime)"
    "  —  mAP@0.5: Baseline 0.9106  →  DAEG-YOLO 0.9138  (+0.32 pp)"
)
LABEL_DARK = (
    "DarkFace (Face Detection, Low-Light)"
    "  —  mAP@0.5: Baseline 0.5642  →  DAEG-YOLO 0.5825  (+1.83 pp)"
)


# ── helpers ────────────────────────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ["timesnewroman.ttf", "times.ttf", "arial.ttf", "Arial.ttf"]:
        for folder in [r"C:\Windows\Fonts", r"C:\Windows\fonts"]:
            p = Path(folder) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def find_row_start(arr: np.ndarray) -> int:
    
    col_w = arr.shape[1] // 4
    brightness = arr[:, 0:col_w, :].mean(axis=(1, 2))
    for y in range(80, arr.shape[0] - 1):
        if brightness[y] < 100 and brightness[y - 1] > 130:
            return y
    return int(arr.shape[0] * 0.30)


def find_col0_start(arr: np.ndarray, probe_y: int) -> int:
    
    row_slice = arr[probe_y, :, :].mean(axis=1)
    for x in range(1, 300):
        if row_slice[x] < 80:
            return max(0, x - 2)
    return 60


def crop_row_only(src: Path) -> Image.Image:
    
    img  = Image.open(src).convert("RGB")
    w, h = img.size
    y0   = int(h * TITLE_FRAC)
    content  = img.crop((0, y0, w, h))
    cw, ch   = content.size
    full_row = content.crop((0, 0, cw, int(ch * ROW_FRAC)))

    
    new_h        = int(TARGET_W * full_row.height / full_row.width)
    full_resized = full_row.resize((TARGET_W, new_h), Image.LANCZOS)

    
    arr       = np.array(full_resized)
    row_start = find_row_start(arr)
    row_only  = full_resized.crop((0, row_start, TARGET_W, new_h))
    return row_only


def cover_a_marker(img: Image.Image) -> Image.Image:
    
    img  = img.copy()
    arr  = np.array(img)
    draw = ImageDraw.Draw(img)

    
    probe_y = min(10, img.height - 1)
    col_x   = find_col0_start(arr, probe_y)
    col_w   = img.width // 4
    a_x     = col_x + int(0.02 * col_w)

    x0, y0 = max(0, a_x - 5), 0
    x1, y1 = a_x + 68,        50

    
    sy0 = min(y1 + 5, img.height - 1)
    sy1 = min(sy0 + 50, img.height - 1)
    if sy1 > sy0:
        sample = arr[sy0:sy1, x0:min(x1, img.width)]
        fill   = tuple(int(v) for v in np.median(sample.reshape(-1, 3), axis=0))
    else:
        fill = (8, 8, 8)

    draw.rectangle([x0, y0, x1, y1], fill=fill)
    return img


# ── canvas builder ─────────────────────────────────────────────────────────────

def text_h(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def main():
    llvip = cover_a_marker(crop_row_only(SRC_LLVIP))
    dark  = cover_a_marker(crop_row_only(SRC_DARK))

    font  = get_font(LABEL_FONT_SZ)
    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    lh    = text_h(ddraw, LABEL_LLVIP, font)

    total_h = (lh + LABEL_GAP
               + llvip.height
               + SECTION_GAP
               + lh + LABEL_GAP
               + dark.height
               + 6)

    canvas = Image.new("RGB", (TARGET_W, total_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    y = 0
    # ── LLVIP ──────────────────────────────────────────────────────────────────
    draw.text((TARGET_W // 2, y), LABEL_LLVIP,
              fill=(10, 10, 10), font=font, anchor="mt")
    y += lh + LABEL_GAP
    canvas.paste(llvip, (0, y))
    y += llvip.height + SECTION_GAP

    # ── DarkFace ───────────────────────────────────────────────────────────────
    draw.text((TARGET_W // 2, y), LABEL_DARK,
              fill=(10, 10, 10), font=font, anchor="mt")
    y += lh + LABEL_GAP
    canvas.paste(dark, (0, y))

    # ── Save ───────────────────────────────────────────────────────────────────
    png = OUT_BASE.with_suffix(".png")
    jpg = OUT_BASE.with_suffix(".jpg")
    pdf = OUT_BASE.with_suffix(".pdf")

    canvas.save(str(png))
    print(f"[OK] PNG -> {png}")
    canvas.save(str(jpg), quality=95, subsampling=0)
    print(f"[OK] JPG -> {jpg}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(TARGET_W / OUT_DPI, total_h / OUT_DPI))
        ax.imshow(np.array(canvas)); ax.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(str(pdf), bbox_inches="tight", dpi=OUT_DPI)
        plt.close(fig)
        print(f"[OK] PDF -> {pdf}")
    except Exception as e:
        print(f"[WARN] PDF: {e}")

    print(f"\nDone. Canvas size: {TARGET_W}x{total_h} px")


if __name__ == "__main__":
    main()
