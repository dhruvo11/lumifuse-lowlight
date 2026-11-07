"""Adaptive classical low-light pipeline.

Stages (each one can be switched off for the ablation study):
  1. Retinex decomposition  I = R * L   (L = max-RGB refined with a guided filter)
  2. Illumination adjustment L' = L^g    (g chosen from the image's log-mean illumination)
  3. Gain-aware edge-preserving denoising (guided or bilateral filter, strength from
     the measured noise level and the per-pixel amplification)
  4. CLAHE on the LAB lightness channel (clip limit chosen from measured contrast)
  5. Colour correction: gray-world white balance + saturation compensation, both
     scaled by how strong the measured cast / desaturation is
  6. Contrast stretch + soft-knee highlight protection (prevents over-exposure)

With adaptive=False every parameter falls back to a fixed, hand-picked constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .quality import analyze, estimate_noise
from .utils import guided_filter, luminance, to_uint8

TARGET_LUMA = 0.45  # mean brightness of well-exposed references (LOL ground truth ~0.4-0.5)


@dataclass
class ClassicalConfig:
    adaptive: bool = True
    use_retinex: bool = True
    use_denoise: bool = True
    use_clahe: bool = True
    use_color: bool = True
    use_highlight: bool = True
    denoiser: str = "guided"  # "guided" | "bilateral"
    dn_k: float = 5.0          # luma eps = (dn_k * sigma)^2 (tuned on val split)
    dn_floor: float = 0.85     # minimum denoise weight in un-amplified regions (tuned on val)
    noise_radius: bool = False # scale filter radius with the measured noise level
    stretch_max_sigma: float = 0.02  # skip contrast stretch on noisy images (tuned on val)
    target: float = TARGET_LUMA


def estimate_illumination(img: np.ndarray, r: int, eps: float) -> np.ndarray:
    l0 = img.max(2)
    return np.clip(guided_filter(l0, l0, r, eps), 1e-3, 1.0)


def soft_knee(x: np.ndarray, knee: float = 0.88) -> np.ndarray:
    """Identity below the knee, smooth roll-off above it (never clips hard)."""
    hi = x > knee
    out = x.copy()
    span = 1.0 - knee
    out[hi] = knee + span * np.tanh((x[hi] - knee) / span)
    return out


def enhance_classical(img: np.ndarray, cfg: ClassicalConfig | None = None, stats: dict | None = None):
    cfg = cfg or ClassicalConfig()
    stats = stats or analyze(img)
    h, w = img.shape[:2]
    P, inter = {}, {}
    x = img.astype(np.float32)

    # 1-2. Retinex: estimate illumination and re-light it ------------------------------
    if cfg.adaptive:
        r = int(max(4, round(min(h, w) / 50)))
        eps = float(max((0.1 * np.percentile(x.max(2), 90)) ** 2, 1e-5))  # scale-invariant smoothing
    else:
        r, eps = 8, 1e-2
    L = estimate_illumination(x, r, eps)
    log_mean = float(np.exp(np.log(L).mean()))
    if cfg.adaptive:
        g = float(np.clip(np.log(cfg.target) / np.log(np.clip(log_mean, 1e-3, 0.999)), 0.15, 1.0))
    else:
        g = 0.45
    P.update(illum_radius=r, illum_eps=round(eps, 6), illum_logmean=round(log_mean, 4), gamma=round(g, 3))
    inter["illumination"] = L
    if cfg.use_retinex:
        L_adj = L ** g
        gain = L_adj / L
        x = np.clip(x * gain[..., None], 0, 1)
        inter["illumination_adjusted"] = L_adj
        inter["reflectance"] = np.clip(img / L[..., None], 0, 1)
    else:
        gain = np.ones_like(L)
    inter["gain"] = gain
    inter["after_retinex"] = x

    # 3. Gain-aware edge-preserving denoising -------------------------------------------
    sigma = estimate_noise(x)
    P["noise_sigma_after_gain"] = round(sigma, 4)
    if cfg.use_denoise:
        k = cfg.dn_k if cfg.adaptive else 2.0
        sig = sigma if cfg.adaptive else 0.02
        ycc = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
        Y = ycc[..., 0]
        if cfg.denoiser == "bilateral":
            Yd = cv2.bilateralFilter(Y, 5, float(3 * sig), 3)
        else:
            if cfg.adaptive and cfg.noise_radius:
                ry = int(np.clip(round(2 + 60 * sig), 2, 7))
            else:
                ry = (2 if sig < 0.01 else 3) if cfg.adaptive else 2
            Yd = guided_filter(Y, Y, ry, float((k * sig) ** 2 + 1e-6))
        # Chroma noise dominates in low light: joint-filter Cr/Cb guided by clean luma.
        rc = (int(np.clip(round(4 + 120 * sig), 4, 12)) if cfg.noise_radius else 6) if cfg.adaptive else 4
        Cd = guided_filter(Yd, ycc[..., 1:], rc, float((2 * k * sig) ** 2 + 1e-6))
        if cfg.adaptive:  # filter harder where the image was amplified more
            p90 = np.percentile(gain, 90)
            W = np.clip((gain - 1) / (p90 - 1 + 1e-6), 0.0, 1.0) ** 0.5
            W = np.maximum(W, cfg.dn_floor).astype(np.float32)
        else:
            W = np.ones_like(Y)
        Y2 = Y + W * (Yd - Y)
        x = np.clip(cv2.cvtColor(np.dstack([Y2, Cd]).astype(np.float32), cv2.COLOR_YCrCb2RGB), 0, 1)
        inter["denoise_weight"] = W
        P.update(denoiser=cfg.denoiser, denoise_eps_luma=round(float((k * sig) ** 2), 6))
    inter["after_denoise"] = x

    # 4. CLAHE on lightness -----------------------------------------------------------------
    if cfg.use_clahe:
        contrast = float(luminance(x).std())
        if cfg.adaptive:
            clip = float(np.clip(1.0 + 2.5 * (0.22 - contrast) / 0.22, 1.0, 3.0))
            mix = 0.55 if sigma > 0.03 else 0.75  # noisy images: lighter CLAHE (it amplifies noise)
            tiles = 8 if min(h, w) >= 384 else 4
        else:
            clip, mix, tiles = 2.0, 1.0, 8
        lab = cv2.cvtColor(to_uint8(x), cv2.COLOR_RGB2LAB)
        l_eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
        lab[..., 0] = np.clip(mix * l_eq + (1 - mix) * lab[..., 0], 0, 255).astype(np.uint8)
        x = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        P.update(clahe_clip=round(clip, 3), clahe_tiles=tiles, clahe_mix=mix, contrast_before_clahe=round(contrast, 4))
    inter["after_clahe"] = x

    # 5. Colour correction ------------------------------------------------------------------
    if cfg.use_color:
        means = x.reshape(-1, 3).mean(0) + 1e-6
        cast = float((means.max() - means.min()) / means.mean())
        s = float(np.clip(cast / 0.3, 0, 1) * 0.6) if cfg.adaptive else 0.5
        gains = (means.mean() / means) ** s
        x = np.clip(x * gains[None, None, :], 0, 1)
        hsv = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2HSV)
        mean_s = float(hsv[..., 1].mean())
        sat_gain = float(np.clip(0.30 / (mean_s + 1e-6), 1.0, 1.35)) if cfg.adaptive else 1.2
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 1)
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        P.update(color_cast=round(cast, 4), wb_strength=round(s, 3), sat_gain=round(sat_gain, 3))

    # 6. Contrast stretch + highlight protection -------------------------------------------
    if cfg.use_highlight:
        y = luminance(x)
        lo, hi = np.percentile(y, 0.5), np.percentile(y, 99.5)
        if cfg.adaptive and hi < 0.92 and sigma <= cfg.stretch_max_sigma:
            scale = min(0.95 / max(hi - lo, 1e-3), 1.3)
            x = (x - lo) * scale
            P["stretch_scale"] = round(float(scale), 3)
        x = soft_knee(np.clip(x, 0, None))
    x = np.clip(x, 0, 1).astype(np.float32)
    return x, inter, P
