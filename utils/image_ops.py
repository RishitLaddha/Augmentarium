import io
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from skimage.metrics import structural_similarity as ssim_metric
from skimage.filters import laplace


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
# Metric helpers
# -------------------------

def _to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)


def _grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def _to_hsv_np(img: Image.Image) -> np.ndarray:
    # PIL HSV maps H,S,V to 0..255
    return np.array(img.convert("HSV"), dtype=np.uint8)


def _histogram(arr: np.ndarray, bins=256, range_=(0, 255)) -> np.ndarray:
    # arr uint8 expected
    hist, _ = np.histogram(arr.flatten(), bins=bins, range=(range_[0], range_[1] + 1))
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return np.ones(bins) / bins
    return hist / hist_sum


def _jsd_bits(p: np.ndarray, q: np.ndarray, eps=1e-12) -> float:
    # Jensen-Shannon Divergence with log base 2 => bits
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * (np.log2(a) - np.log2(b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)  # in bits; 0..1


def _luminance_mean(img: Image.Image) -> float:
    # Rec. 709 luminance (scaled 0..255 since uint8)
    arr = _to_np_rgb(img).astype(np.float32)
    y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return float(y.mean())


def _saturation_mean(img: Image.Image) -> float:
    hsv = _to_hsv_np(img).astype(np.float32)
    return float(hsv[..., 1].mean())  # 0..255


def _clipping_fractions_gray(img: Image.Image) -> Tuple[float, float]:
    g = np.array(_grayscale(img), dtype=np.uint8)
    total = g.size
    if total == 0:
        return 0.0, 0.0
    clip0 = float((g == 0).sum()) / total
    clip255 = float((g == 255).sum()) / total
    return clip0, clip255


def _ssim_gray(a: Image.Image, b: Image.Image, target=512) -> float:
    """
    Downscale to manageable size, grayscale, compute SSIM [0..1].
    - Ensures data_range is set for float images (skimage requirement).
    - Adapts win_size for very small images (must be odd and <= min image dim).
    """
    def _prep(x: Image.Image) -> np.ndarray:
        x = x.convert("L")
        w, h = x.size
        # We only downscale large images; tiny ones stay as-is
        scale = target / max(w, h)
        if scale < 1.0:
            x = x.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(x, dtype=np.float32) / 255.0
        return arr

    A = _prep(a)
    B = _prep(b)
    h_min = min(A.shape[0], A.shape[1], B.shape[0], B.shape[1])

    # Choose an odd win_size <= min dimension; prefer 7, then 5, then 3
    if h_min >= 7:
        win = 7
    elif h_min >= 5:
        win = 5
    elif h_min >= 3:
        win = 3
    else:
        # Extreme edge case: SSIM not defined; treat as “very similar”
        return 0.99

    s, _ = ssim_metric(
        A, B,
        full=True,
        win_size=win,
        gaussian_weights=True,
        use_sample_covariance=False,
        data_range=1.0,  # <-- key fix: our arrays are in [0,1]
    )
    return float(s)


def _sharpness_score(img: Image.Image) -> float:
    # variance of Laplacian as sharpness proxy
    g = np.array(_grayscale(img), dtype=np.float32) / 255.0
    L = laplace(g)
    return float(L.var())


# -------------------------
# Verdict mapping (English only)
# -------------------------

@dataclass
class Verdict:
    label: str
    color: str  # tailwind class color
    hints: List[str]


def _band_jsd(v: float) -> str:
    # bits: 0..1 (but real images small)
    if v <= 0.02:
        return "tiny"
    if v <= 0.07:
        return "small"
    if v <= 0.15:
        return "moderate"
    return "large"


def _band_ssim(v: float) -> str:
    if v >= 0.85:
        return "high"
    if v >= 0.70:
        return "mid"
    return "low"


def _band_clip(frac: float) -> str:
    if frac <= 0.01:
        return "low"
    if frac <= 0.05:
        return "medium"
    return "high"


def _delta_band(val: float, scale: float) -> str:
    # val on 0..scale (brightness or saturation scale 255)
    r = abs(val)
    if r <= 0.03 * scale:
        return "small"
    if r <= 0.12 * scale:
        return "moderate"
    return "large"


def english_verdict(
    jsd_gray_bits: float, jsd_hue_bits: float, ssim_val: float,
    clip0: float, clip255: float,
    d_brightness: float, d_saturation: float,
    sharpness_orig: float, sharpness_aug: float
) -> Verdict:
    jg = _band_jsd(jsd_gray_bits)
    jh = _band_jsd(jsd_hue_bits)
    sband = _band_ssim(ssim_val)
    clip_band = max(_band_clip(clip0), _band_clip(clip255), key=lambda x: ["low","medium","high"].index(x))

    # Build hints (no numbers)
    # intensity/color
    db = _delta_band(d_brightness, 255.0)
    ds = _delta_band(d_saturation, 255.0)
    if db == "small":
        intensity_hint = "Brightness/contrast changed a little."
    elif db == "moderate":
        intensity_hint = "Brightness/contrast changed moderately."
    else:
        intensity_hint = "Brightness/contrast changed a lot."

    if ds == "small":
        color_hint = "Colors barely changed."
    elif ds == "moderate":
        color_hint = "Colors changed noticeably."
    else:
        color_hint = "Colors are very different."

    # structure
    if sband == "high":
        struct_hint = "Edges and shapes are intact."
    elif sband == "mid":
        struct_hint = "Edges and shapes changed somewhat."
    else:
        struct_hint = "Structure is largely altered."

    # exposure
    if clip_band == "low":
        clip_hint = "Few pixels at pure black/white."
    elif clip_band == "medium":
        clip_hint = "Noticeable extremes—consider reducing strength."
    else:
        clip_hint = "Many extremes—likely over/under-exposed."

    # sharpness
    sharp_delta = sharpness_aug - sharpness_orig
    if sharp_delta > 0.0005:
        sharp_hint = "Image looks sharper."
    elif sharp_delta < -0.0005:
        sharp_hint = "Image looks a bit soft."
    else:
        sharp_hint = "Sharpness similar to original."

    # Overall label
    jsd_order = ["tiny", "small", "moderate", "large"]
    jsd_max = max(jg, jh, key=lambda x: jsd_order.index(x))

    if sband == "low" or clip_band == "high":
        label = "Probably destructive — structure lost"
        color = "bg-red-600"
    else:
        if jsd_max == "tiny" and sband in ("mid", "high"):
            label = "Too similar — try stronger tweak"
            color = "bg-slate-600"
        elif jsd_max in ("small",) and sband == "high":
            label = "Subtle variety — safe"
            color = "bg-emerald-600"
        elif jsd_max in ("moderate",):
            label = "Good variety — sweet spot"
            color = "bg-blue-600"
        else:  # large with decent SSIM
            label = "Aggressive — check content"
            color = "bg-amber-600"

    return Verdict(label=label, color=color, hints=[intensity_hint, color_hint, struct_hint, clip_hint, sharp_hint])



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
    im = img.rotate(45, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0))
    W, H = im.size
    left = (W - w) // 2
    top = (H - h) // 2
    return im.crop((left, top, left + w, top + h))

def aug_rotate_90(img):
    w, h = img.size
    im = img.rotate(90, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0))
    W, H = im.size
    left = (W - w) // 2
    top = (H - h) // 2
    return im.crop((left, top, left + w, top + h))

def aug_translate(img):
    w, h = img.size
    dx = int(0.08 * w)
    dy = int(0.05 * h)
    return img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BICUBIC, fillcolor=(0,0,0))

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

# Curated list (you can reorder as you like)
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

    # If you still like it:
    ("Unsharp Mask",          aug_unsharp),
]
# -------------------------
# Pipeline
# -------------------------

def generate_augmentations_with_verdicts(img: Image.Image, out_dir: str, token: str) -> List[Dict]:
    # Original stats
    js_gray_orig = _histogram(np.array(_grayscale(img), dtype=np.uint8))
    hue_orig = _histogram(_to_hsv_np(img)[..., 0])
    bright_orig = _luminance_mean(img)
    sat_orig = _saturation_mean(img)
    sharp_orig = _sharpness_score(img)

    items = []
    w, h = img.size

    # fixed seed for reproducibility on each render
    random.seed(42)
    np.random.seed(42)

    for idx, (label, func) in enumerate(AUGS, start=1):
        aug = func(img)
        aug = _ensure_size(aug, (w, h))

        # Metrics (hidden, used to craft English)
        js_gray_aug = _histogram(np.array(_grayscale(aug), dtype=np.uint8))
        hue_aug = _histogram(_to_hsv_np(aug)[..., 0])

        jsd_gray = _jsd_bits(js_gray_orig, js_gray_aug)
        jsd_hue = _jsd_bits(hue_orig, hue_aug)
        ssim_val = _ssim_gray(img, aug)
        clip0, clip255 = _clipping_fractions_gray(aug)
        bright_aug = _luminance_mean(aug)
        sat_aug = _saturation_mean(aug)
        sharp_aug = _sharpness_score(aug)

        verdict = english_verdict(
            jsd_gray_bits=jsd_gray,
            jsd_hue_bits=jsd_hue,
            ssim_val=ssim_val,
            clip0=clip0,
            clip255=clip255,
            d_brightness=(bright_aug - bright_orig),
            d_saturation=(sat_aug - sat_orig),
            sharpness_orig=sharp_orig,
            sharpness_aug=sharp_aug
        )

        # Save
        fname = f"{idx:02d}_{label.lower().replace(' ', '_').replace('°','deg')}.png"
        fpath = f"{out_dir}/{fname}"
        aug.save(fpath, format="PNG", optimize=True)

        items.append({
            "name": label,
            "filename": fname,
            "verdict_label": verdict.label,
            "verdict_color": verdict.color,
            "hints": verdict.hints,
            "idx": idx,
        })

    return items
