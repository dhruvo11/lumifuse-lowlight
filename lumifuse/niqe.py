"""NIQE (Mittal et al. 2013), numpy port of the BasicSR implementation.

Matches the MATLAB reference: Y channel, rounded, 96x96 blocks, two scales with
MATLAB-style antialiased bicubic downsampling. Lower is better.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma

_PARAMS = np.load(os.path.join(os.path.dirname(__file__), "niqe_pris_params.npz"), allow_pickle=False)
_MU = _PARAMS["mu_pris_param"]
_COV = _PARAMS["cov_pris_param"]
_WIN = _PARAMS["gaussian_window"]

_GAM = np.arange(0.2, 10.001, 0.001)
_R_GAM = np.square(gamma(2 / _GAM)) / (gamma(1 / _GAM) * gamma(3 / _GAM))


def _aggd(block):
    block = block.flatten()
    left = block[block < 0]
    right = block[block > 0]
    if left.size == 0 or right.size == 0:
        return np.nan, np.nan, np.nan
    left_std = np.sqrt(np.mean(left ** 2))
    right_std = np.sqrt(np.mean(right ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
    alpha = _GAM[np.argmin((_R_GAM - rhatnorm) ** 2)]
    k = np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return alpha, left_std * k, right_std * k


def _features(block):
    alpha, bl, br = _aggd(block)
    feat = [alpha, (bl + br) / 2]
    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        alpha, bl, br = _aggd(block * np.roll(block, shift, axis=(0, 1)))
        mean = (br - bl) * (gamma(2 / alpha) / gamma(1 / alpha)) if np.isfinite(alpha) else np.nan
        feat.extend([alpha, mean, bl, br])
    return feat


def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))


def _resize_matrix(n_in: int, n_out: int, scale: float) -> np.ndarray:
    kw = 4.0 / scale if scale < 1 else 4.0
    x = np.arange(1, n_out + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kw / 2)
    p = int(math.ceil(kw)) + 2
    idx = left[:, None] + np.arange(p)[None, :]
    dist = u[:, None] - idx
    w = scale * _cubic(dist * scale) if scale < 1 else _cubic(dist)
    w = w / w.sum(1, keepdims=True)
    idx = idx.astype(int)
    idx = np.where(idx < 1, 1 - idx, idx)
    idx = np.where(idx > n_in, 2 * n_in + 1 - idx, idx)
    m = np.zeros((n_out, n_in))
    rows = np.repeat(np.arange(n_out), p)
    np.add.at(m, (rows, idx.ravel() - 1), w.ravel())
    return m


def _imresize_half(img):
    h, w = img.shape
    oh, ow = int(math.ceil(h * 0.5)), int(math.ceil(w * 0.5))
    return _resize_matrix(h, oh, 0.5) @ img @ _resize_matrix(w, ow, 0.5).T


def _niqe_y(img, block=96):
    h, w = img.shape
    nh, nw = h // block, w // block
    img = img[: nh * block, : nw * block]
    dist = []
    for scale in (1, 2):
        mu = convolve(img, _WIN, mode="nearest")
        sigma = np.sqrt(np.abs(convolve(np.square(img), _WIN, mode="nearest") - np.square(mu)))
        norm = (img - mu) / (sigma + 1)
        bs = block // scale
        feats = [_features(norm[ih * bs:(ih + 1) * bs, iw * bs:(iw + 1) * bs]) for iw in range(nw) for ih in range(nh)]
        dist.append(np.array(feats))
        if scale == 1:
            img = _imresize_half(img / 255.0) * 255.0
    dist = np.concatenate(dist, axis=1)
    mu_d = np.nanmean(dist, axis=0)
    clean = dist[~np.isnan(dist).any(axis=1)]
    cov_d = np.cov(clean, rowvar=False) if len(clean) > 1 else np.zeros_like(_COV)
    inv = np.linalg.pinv((_COV + cov_d) / 2)
    diff = _MU - mu_d
    return float(np.sqrt(np.squeeze(diff @ inv @ diff.T)))


def niqe(img_rgb: np.ndarray) -> float:
    """NIQE of a float RGB [0,1] image (needs at least 96x96 pixels)."""
    y = img_rgb.astype(np.float64) @ np.array([65.481, 128.553, 24.966]) + 16.0
    y = np.round(y)
    if min(y.shape) < 96:
        return float("nan")
    return _niqe_y(y)
