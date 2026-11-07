"""Image-quality analysis used to drive every adaptive decision in the pipeline.

Everything here is no-reference: it only looks at the input (or output) image.
"""
from __future__ import annotations

import cv2
import numpy as np

from .utils import luminance

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise(img: np.ndarray) -> float:
    """Noise std (0-1 scale) via Immerkaer's Laplacian-difference operator,
    restricted to the flattest 50% of pixels so texture is not counted as noise."""
    y = luminance(img) if img.ndim == 3 else img.astype(np.float32)
    resp = np.abs(cv2.filter2D(y, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    grad = np.abs(gx) + np.abs(gy)
    flat = grad <= np.percentile(grad, 50)
    if flat.sum() < 100:
        flat = np.ones_like(grad, dtype=bool)
    return float(np.sqrt(np.pi / 2) * resp[flat].mean() / 6.0)


def colorfulness(img: np.ndarray) -> float:
    """Hasler & Suesstrunk colourfulness metric (0-1 scale input)."""
    r, g, b = img[..., 0] * 255, img[..., 1] * 255, img[..., 2] * 255
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def entropy(img: np.ndarray) -> float:
    y = (np.clip(luminance(img), 0, 1) * 255).astype(np.uint8)
    hist = np.bincount(y.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def analyze(img: np.ndarray) -> dict:
    """Global statistics that select the enhancement parameters."""
    y = luminance(img)
    h, w = y.shape
    # Large-scale illumination: heavy blur of max-RGB, used for non-uniformity.
    k = max(3, (min(h, w) // 8) | 1)
    illum = cv2.GaussianBlur(img.max(2), (k, k), 0)
    log_i = np.log(illum + 0.01)
    means = img.reshape(-1, 3).mean(0)
    cast = float((means.max() - means.min()) / (means.mean() + 1e-6))
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    stats = {
        "mean_luma": float(y.mean()),
        "median_luma": float(np.median(y)),
        "p95_luma": float(np.percentile(y, 95)),
        "dark_frac": float((y < 0.1).mean()),
        "bright_frac": float((y > 0.85).mean()),
        "nonuniformity": float(log_i.std()),
        "noise_sigma": estimate_noise(img),
        "contrast_rms": float(y.std()),
        "color_cast": cast,
        "saturation": float(hsv[..., 1][y > 0.05].mean()) if (y > 0.05).any() else 0.0,
        "entropy": entropy(img),
    }
    stats["condition"] = classify(stats)
    return stats


def classify(s: dict) -> str:
    if s["mean_luma"] > 0.35 and s["dark_frac"] < 0.3:
        return "well-lit"
    if s["nonuniformity"] > 0.75 and s["bright_frac"] > 0.02:
        return "non-uniform / backlit"
    if s["mean_luma"] < 0.05:
        return "extremely dark"
    if s["mean_luma"] < 0.2:
        return "low-light"
    return "dim"
