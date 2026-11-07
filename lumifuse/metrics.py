"""Evaluation metrics. Full-reference: PSNR, SSIM, LPIPS, edge preservation.
No-reference: NIQE, residual noise sigma."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .niqe import niqe as _niqe
from .quality import estimate_noise
from .utils import luminance

_LPIPS = None


def psnr(out, gt):
    return float(peak_signal_noise_ratio(gt, out, data_range=1.0))


def ssim(out, gt):
    return float(structural_similarity(gt, out, data_range=1.0, channel_axis=2))


def lpips(out, gt):
    global _LPIPS
    import torch
    if _LPIPS is None:
        import lpips as _lp
        _LPIPS = _lp.LPIPS(net="alex", verbose=False).eval()
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float() * 2 - 1
    with torch.no_grad():
        return float(_LPIPS(t(out), t(gt)).item())


def edge_preservation(out, gt):
    """Pearson correlation of Sobel gradient magnitudes (1 = edges perfectly kept)."""
    def mag(a):
        y = luminance(a)
        return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)).ravel()
    a, b = mag(out), mag(gt)
    return float(np.corrcoef(a, b)[0, 1])


def niqe(out):
    return _niqe(out)


def noise_sigma(out):
    return estimate_noise(out)
