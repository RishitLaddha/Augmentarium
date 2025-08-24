import random
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


# -------------------------
# Loading & preparation
# -------------------------

MAX_SIDE = 768  # target display/process size


def load_and_prepare(filelike) -> Image.Image:
    """Load from file-like, EXIF-correct, force RGB, and limit max side."""
    img = Image.open(filelike)
    img = ImageOps.exif_transpose(img)  # respect camera orientation
    img = img.convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# -------------------------
# Augmentations (lecture-aligned set)
# -------------------------

def _ensure_size(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    return img.resize(target_size, Image.LANCZOS)

def aug_flip_h(img):  # Horizontal Flip
    return ImageOps.mirror(img)

def aug_flip_v(img):  # Vertical Flip
    return ImageOps.flip(img)

def aug_rotate_45(img):
    w, h = img.size
    im = img.rotate(45, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    W, H = im.size
    left = (W - w) // 2
    top = (H - h) // 2
    return im.crop((left, top, left + w, top + h))

def aug_rotate_90(img):
    w, h = img.size
    im = img.rotate(90, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    W, H = im.size
    left = (W - w) // 2
    top = (H - h) // 2
    return im.crop((left, top, left + w, top + h))

def aug_translate(img):
    w, h = img.size
    dx = int(0.08 * w)
    dy = int(0.05 * h)
    return img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy),
                         resample=Image.BICUBIC, fillcolor=(0, 0, 0))

def aug_center_zoom(img):
    w, h = img.size
    crop_ratio = 0.85
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    im = img.crop((left, top, left + cw, top + ch))
    return im.resize((w, h), Image.LANCZOS)

def aug_grayscale(img):
    return img.convert("L").convert("RGB")

def aug_red_channel(img):
    arr = np.array(img, dtype=np.uint8)
    out = np.zeros_like(arr)
    out[..., 0] = arr[..., 0]  # keep R
    return Image.fromarray(out, mode="RGB")

def aug_green_channel(img):
    arr = np.array(img, dtype=np.uint8)
    out = np.zeros_like(arr)
    out[..., 1] = arr[..., 1]  # keep G
    return Image.fromarray(out, mode="RGB")

def aug_blue_channel(img):
    arr = np.array(img, dtype=np.uint8)
    out = np.zeros_like(arr)
    out[..., 2] = arr[..., 2]  # keep B
    return Image.fromarray(out, mode="RGB")

def aug_hue_shift(img):  # +12°
    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    h = hsv[..., 0].astype(np.int16)
    h = (h + 8) % 256
    hsv[..., 0] = h.astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")

def aug_saturation(img):  # +25%
    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    s = hsv[..., 1].astype(np.int16)
    s = np.clip(s * 1.25, 0, 255).astype(np.uint8)
    hsv[..., 1] = s
    return Image.fromarray(hsv, mode="HSV").convert("RGB")

def aug_contrast(img):  # +20%
    return ImageEnhance.Contrast(img).enhance(1.2)

def aug_brightness(img):  # +20%
    return ImageEnhance.Brightness(img).enhance(1.2)

def aug_invert(img):
    return ImageOps.invert(img)

def aug_edges(img):
    # Edge detect on luminance, then stack to RGB for clarity
    g = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return Image.merge("RGB", (g, g, g))

def aug_unsharp(img):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Curated list (reorder as you like)
AUGS = [
    ("Horizontal Flip",       aug_flip_h),
    ("Vertical Flip",         aug_flip_v),
    ("Rotate +45°",           aug_rotate_45),
    ("Rotate +90°",           aug_rotate_90),
    ("Translate (shift)",     aug_translate),
    ("Center Crop + Resize",  aug_center_zoom),

    ("Grayscale",             aug_grayscale),
    ("Red Channel Only",      aug_red_channel),
    ("Green Channel Only",    aug_green_channel),
    ("Blue Channel Only",     aug_blue_channel),

    ("Hue +12°",              aug_hue_shift),
    ("Saturation +25%",       aug_saturation),
    ("Contrast +20%",         aug_contrast),
    ("Brightness +20%",       aug_brightness),

    ("Invert Colors",         aug_invert),
    ("Edge Detect",           aug_edges),

    ("Unsharp Mask",          aug_unsharp),
]


# -------------------------
# Pipeline (no metrics, no verdicts)
# -------------------------

def generate_augmentations_with_verdicts(img: Image.Image, out_dir: str, token: str) -> List[Dict]:
    """
    Keeps the old function name for compatibility with app.py,
    but no longer computes or returns any verdict/metrics.
    """
    items = []
    w, h = img.size

    random.seed(42)
    np.random.seed(42)

    for idx, (label, func) in enumerate(AUGS, start=1):
        aug = func(img)
        aug = _ensure_size(aug, (w, h))

        fname = f"{idx:02d}_{label.lower().replace(' ', '_').replace('°','deg').replace('+','plus')}.png"
        fpath = f"{out_dir}/{fname}"
        aug.save(fpath, format="PNG", optimize=True)

        # Only the fields the template needs
        items.append({
            "name": label,
            "filename": fname,
            "idx": idx,
        })

    return items
def parametric_apply(img: Image.Image, key: str, params: dict) -> Image.Image:
    """Apply a parameterized transform for the Interactive Lab."""
    key = (key or "").lower()

    if key == "rotate":
        angle = float(params.get("angle", 15.0))
        w, h = img.size
        im = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0))
        W, H = im.size
        left = (W - w) // 2
        top = (H - h) // 2
        return im.crop((left, top, left + w, top + h))

    if key == "translate":
        dx_pct = float(params.get("dx", 8.0))
        dy_pct = float(params.get("dy", 5.0))
        w, h = img.size
        dx = int(dx_pct / 100.0 * w)
        dy = int(dy_pct / 100.0 * h)
        return img.transform((w, h), Image.AFFINE, (1,0,dx, 0,1,dy), resample=Image.BICUBIC, fillcolor=(0,0,0))

    if key == "zoom":
        ratio = float(params.get("ratio", 85.0)) / 100.0  # 0.60..1.00
        ratio = max(0.60, min(1.00, ratio))
        w, h = img.size
        cw, ch = int(w * ratio), int(h * ratio)
        left = (w - cw) // 2
        top = (h - ch) // 2
        im = img.crop((left, top, left + cw, top + ch))
        return im.resize((w, h), Image.LANCZOS)

    if key == "brightness":
        scale = float(params.get("scale", 1.2))
        return ImageEnhance.Brightness(img).enhance(scale)

    if key == "contrast":
        scale = float(params.get("scale", 1.2))
        return ImageEnhance.Contrast(img).enhance(scale)

    if key == "saturation":
        scale = float(params.get("scale", 1.25))
        hsv = np.array(img.convert("HSV"), dtype=np.uint8)
        s = hsv[..., 1].astype(np.float32)
        s = np.clip(s * scale, 0, 255).astype(np.uint8)
        hsv[..., 1] = s
        return Image.fromarray(hsv, mode="HSV").convert("RGB")

    if key == "hue":
        deg = float(params.get("deg", 12.0))
        offset = int(round(deg / 360.0 * 256.0))
        hsv = np.array(img.convert("HSV"), dtype=np.uint8)
        hch = hsv[..., 0].astype(np.int16)
        hch = (hch + offset) % 256
        hsv[..., 0] = hch.astype(np.uint8)
        return Image.fromarray(hsv, mode="HSV").convert("RGB")

    if key == "blur":
        radius = float(params.get("radius", 1.2))
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if key == "unsharp":
        percent = int(float(params.get("percent", 150)))
        radius = float(params.get("radius", 2.0))
        threshold = int(float(params.get("threshold", 3)))
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    if key == "posterize":
        bits = int(float(params.get("bits", 4)))
        bits = max(1, min(8, bits))
        return ImageOps.posterize(img, bits=bits)

    # Fallback: no-op
    return img
