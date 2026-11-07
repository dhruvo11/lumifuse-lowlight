"""Shared image helpers. All images are float32 RGB in [0, 1], shape (H, W, 3)."""
from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return to_float_rgb(img, bgr=True)


def to_float_rgb(img: np.ndarray, bgr: bool = False) -> np.ndarray:
    """Normalise any uint8/uint16/float image (gray, RGB, RGBA) to float32 RGB [0,1]."""
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if bgr:
        img = img[:, :, ::-1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
    return np.ascontiguousarray(np.clip(img, 0, 1))


def save_image(path: str, img: np.ndarray) -> None:
    out = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if out.ndim == 3:
        out = out[:, :, ::-1]
    cv2.imwrite(str(path), out)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def luminance(img: np.ndarray) -> np.ndarray:
    return (img @ LUMA).astype(np.float32)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    k = max_side / s
    return cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)


def box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (gray guide). Edge-preserving, O(N) via box filters."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    if src.ndim == 3:
        return np.stack([guided_filter(guide, src[:, :, c], r, eps) for c in range(src.shape[2])], -1)
    mean_i = box(guide, r)
    mean_p = box(src, r)
    cov_ip = box(guide * src, r) - mean_i * mean_p
    var_i = box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a, r) * guide + box(b, r)


def colorize(gray: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO, vmin=None, vmax=None) -> np.ndarray:
    """Map a single-channel array to an RGB float heatmap for visualisation."""
    g = gray.astype(np.float32)
    lo = np.percentile(g, 1) if vmin is None else vmin
    hi = np.percentile(g, 99) if vmax is None else vmax
    g = np.clip((g - lo) / (hi - lo + 1e-8), 0, 1)
    c = cv2.applyColorMap((g * 255).astype(np.uint8), cmap)[:, :, ::-1]
    return c.astype(np.float32) / 255.0
