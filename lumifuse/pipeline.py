"""LumiFuse: the full system. One object, four methods:

  classical  - adaptive classical pipeline only           (lumifuse.classical)
  zerodce    - official Zero-DCE weights, no fine-tuning   (deep learning, zero-reference)
  dlonly     - LumiNet trained on LOL without classical priors (deep learning only)
  hybrid     - the proposed system:
                 1. image analysis -> condition + adaptive parameters
                 2. adaptive input conditioning for out-of-range exposure
                 3. LumiNet conditioned on classical priors (illumination map, gain map)
                 4. classical branch in parallel
                 5. reliability-weighted per-pixel fusion of the two branches
                 6. no-reference quality gates (exposure, residual noise, colour cast)
                 7. soft-knee highlight protection
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .classical import ClassicalConfig, enhance_classical, soft_knee
from .models import LumiNet, load_official_zerodce
from .quality import analyze, estimate_noise
from .utils import colorize, guided_filter, luminance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_LOW_MEAN = 0.05   # mean luma of LOL low-light training inputs (measured)
EXPO_ALPHA = 0.5             # weight of the classical exposure estimate when exposure_prior=True
TRUST_LO, TRUST_HI = 0.12, 0.40  # input brightness range over which trust shifts to the classical branch (tuned on val)
COND_LO, COND_HI = 0.6, 1.2  # input conditioning band, as multiples of TRAIN_LOW_MEAN (tuned on val)
EXPOSURE_RANGE = (0.30, 0.62)
NOISE_GATE = 0.012
CAST_GATE = 0.10


@dataclass
class HybridConfig:
    use_prior: bool = True          # classical priors fed to the network (else: DL-only net)
    use_refine: bool = True         # RefineNet stage (else: curve stage only)
    input_conditioning: bool = True # adaptive exposure normalisation of the input
    use_fusion: bool = True         # reliability-weighted fusion with classical branch
    global_trust: bool = True       # brightness-dependent global share of the classical branch
    use_gates: bool = True          # quality-aware post-processing
    use_highlight: bool = True      # soft-knee highlight protection
    exposure_prior: bool = False    # classical exposure target; tried, lowered val PSNR 18.54 -> 17.39, so off
    official_dce: bool = False      # swap fine-tuned net for official Zero-DCE (no fine-tuning)


def _denoise(x: np.ndarray, sigma: float, k: float = 3.0) -> np.ndarray:
    ycc = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    yd = guided_filter(ycc[..., 0], ycc[..., 0], 2, float((k * sigma) ** 2))
    cd = guided_filter(yd, ycc[..., 1:], 5, float((2 * k * sigma) ** 2))
    return np.clip(cv2.cvtColor(np.dstack([yd, cd]).astype(np.float32), cv2.COLOR_YCrCb2RGB), 0, 1)


class LumiFuse:
    def __init__(self, weights_dir: str | None = None, device: str = "cpu"):
        self.device = device
        wd = weights_dir or os.path.join(ROOT, "weights")
        self.zdce = load_official_zerodce(os.path.join(wd, "zerodce_official.pth")).eval().to(device)
        self.nets = {}
        for v in ("hybrid", "dlonly"):
            p = os.path.join(wd, f"luminet_{v}.pth")
            if os.path.exists(p):
                net = LumiNet(use_prior=(v == "hybrid"))
                net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
                self.nets[v] = net.eval().to(device)

    # ------------------------------------------------------------------ helpers
    def _t(self, img):
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(self.device)

    @staticmethod
    def _n(t):
        return t[0].detach().permute(1, 2, 0).float().cpu().numpy()

    def run_zerodce(self, img):
        scale = 1 if max(img.shape[:2]) <= 1200 else 2
        e, a = self.zdce(self._t(img), scale=scale)
        return np.clip(self._n(e), 0, 1), self._n(a)

    def run_net(self, variant, img, use_refine=True):
        o = self.nets[variant](self._t(img), use_refine=use_refine)
        return {k: self._n(v) for k, v in o.items()}

    # ------------------------------------------------------------------ public API
    @torch.inference_mode()  # grad mode is thread-local; web servers call us from worker threads
    def enhance(self, img: np.ndarray, method: str = "hybrid", hcfg: HybridConfig | None = None,
                ccfg: ClassicalConfig | None = None):
        """Returns (output, intermediates: dict[str, HxWx3], info: dict)."""
        t0 = time.perf_counter()
        if method == "classical":
            out, inter, P = enhance_classical(img, ccfg)
            vis = {"Illumination map L": colorize(inter["illumination"], vmin=0, vmax=1),
                   "Reflectance R = I / L": inter.get("reflectance", img),
                   "Gain map L'/L": colorize(np.log(inter["gain"])),
                   "After Retinex": inter["after_retinex"], "After denoise": inter["after_denoise"],
                   "After CLAHE": inter["after_clahe"]}
            return out, vis, {"params": P, "latency_ms": {"total": 1000 * (time.perf_counter() - t0)}}
        if method == "zerodce":
            out, a = self.run_zerodce(img)
            vis = {"Curve strength |A|": colorize(np.abs(a).mean(2))}
            return out, vis, {"latency_ms": {"total": 1000 * (time.perf_counter() - t0)}}
        if method == "dlonly":
            o = self.run_net("dlonly", img)
            vis = {"Curve stage output": o["dce"], "Curve strength |A|": colorize(np.abs(o["curves"]).mean(2)),
                   "Refiner correction": colorize(np.abs(o["out"] - o["dce"]).mean(2))}
            return o["out"], vis, {"latency_ms": {"total": 1000 * (time.perf_counter() - t0)}}
        if method == "hybrid":
            return self.hybrid(img, hcfg or HybridConfig(), ccfg)
        raise ValueError(method)

    def hybrid(self, img: np.ndarray, cfg: HybridConfig, ccfg: ClassicalConfig | None = None):
        T, info, vis = {}, {}, {}
        h, w = img.shape[:2]
        t = time.perf_counter()
        stats = analyze(img)
        T["analysis"] = time.perf_counter() - t
        info["stats"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}

        # 1. adaptive input conditioning: pull extreme inputs into the training range
        # (the network learned LOL's exposure mapping, so inputs far darker OR brighter than
        #  LOL's low-light inputs are rescaled into that range before inference)
        k = 1.0
        m = stats["mean_luma"]
        if cfg.input_conditioning:
            if m < COND_LO * TRAIN_LOW_MEAN:
                k = float(min(TRAIN_LOW_MEAN / max(m, 1e-4), 4.0))
            elif m > COND_HI * TRAIN_LOW_MEAN:
                k = float(max(COND_HI * TRAIN_LOW_MEAN / m, 0.1))
        x = np.clip(img * k, 0, 1).astype(np.float32)
        info["input_gain"] = round(k, 3)

        # 2. deep branch
        t = time.perf_counter()
        if cfg.official_dce:
            dl, a = self.run_zerodce(x)
            o = {"out": dl, "dce": dl, "curves": a}
        else:
            o = self.run_net("hybrid" if cfg.use_prior else "dlonly", x, use_refine=cfg.use_refine)
            dl = o["out"]
        T["network"] = time.perf_counter() - t

        # 3. classical branch
        t = time.perf_counter()
        cl, cinter, cP = enhance_classical(img, ccfg, stats=stats)
        T["classical_branch"] = time.perf_counter() - t
        info["classical_params"] = cP

        # 4. reliability-weighted fusion
        t = time.perf_counter()
        out = dl
        W = np.zeros((h, w), np.float32)
        if cfg.use_fusion:
            yd, yc = luminance(dl), luminance(cl)
            under = np.clip((0.12 - yd) / 0.12, 0, 1) * (yc > yd + 0.02)   # network left it too dark
            over = np.clip((yd - 0.90) / 0.10, 0, 1) * (yc < yd - 0.02)    # network blew it out
            W = cv2.GaussianBlur(np.maximum(under, over).astype(np.float32), (0, 0), max(2.0, min(h, w) / 80))
            W = np.clip(W, 0, 0.9)
            # Global trust: the network is most reliable in the LOL regime (very dark, noisy);
            # as the input gets brighter its heavy denoising flattens detail, so trust shifts to
            # the classical branch.
            beta = float(np.clip((stats["mean_luma"] - TRUST_LO) / (TRUST_HI - TRUST_LO), 0.0, 0.9)) \
                if cfg.global_trust else 0.0
            W = np.maximum(W, beta).astype(np.float32)
            info["global_classical_trust"] = round(beta, 3)
            out = dl * (1 - W[..., None]) + cl * W[..., None]
        info["fusion_classical_share"] = round(float(W.mean()), 4)
        if stats["condition"] == "well-lit":  # do not over-process already good images
            alpha = float(np.clip((0.6 - stats["mean_luma"]) / 0.25, 0.2, 1.0))
            out = alpha * out + (1 - alpha) * img
            info["well_lit_blend"] = round(alpha, 3)

        # 5. no-reference quality gates
        gates = {}
        if cfg.use_gates:
            y = luminance(out)
            m = float(y.mean())
            # Exposure target: the network alone tends to one learned brightness; the classical
            # Retinex illumination estimate knows how bright *this* scene should be (dark-albedo
            # scenes stay darker). Blend the two in log space, then keep inside the safe range.
            tgt = m
            if cfg.exposure_prior:
                m_cl = float(luminance(cl).mean())
                tgt = float(np.exp((1 - EXPO_ALPHA) * np.log(max(m, 1e-3)) + EXPO_ALPHA * np.log(max(m_cl, 1e-3))))
            tgt = float(np.clip(tgt, *EXPOSURE_RANGE))
            if abs(np.log(tgt / max(m, 1e-3))) > 0.02:
                g = float(np.log(tgt) / np.log(max(m, 1e-3)))
                ratio = np.clip((np.clip(y, 1e-4, 1) ** g) / np.clip(y, 1e-4, 1), 0.5, 4.0)
                out = np.clip(out * ratio[..., None], 0, 1)
                gates["exposure_gamma"] = round(g, 3)
            s = estimate_noise(out)
            gates["residual_noise"] = round(s, 4)
            if s > NOISE_GATE:
                out = _denoise(out, s)
                gates["denoised"] = True
            means = out.reshape(-1, 3).mean(0) + 1e-6
            cast = float((means.max() - means.min()) / means.mean())
            if cast > CAST_GATE:
                sw = min(cast / 0.3, 1.0) * 0.5
                out = np.clip(out * ((means.mean() / means) ** sw)[None, None], 0, 1)
                gates["white_balance"] = round(sw, 3)
        info["quality_gates"] = gates
        if cfg.use_highlight:
            out = soft_knee(out)
        out = np.clip(out, 0, 1).astype(np.float32)
        T["fusion_and_gates"] = time.perf_counter() - t
        info["latency_ms"] = {k: round(1000 * v, 1) for k, v in T.items()}
        info["latency_ms"]["total"] = round(sum(info["latency_ms"].values()), 1)

        vis["Illumination prior L (classical)"] = colorize(cinter["illumination"], vmin=0, vmax=1)
        vis["Reflectance R = I / L"] = cinter.get("reflectance", img)
        vis["Curve strength |A| (DCE stage)"] = colorize(np.abs(o["curves"]).mean(2))
        vis["DCE stage output"] = np.clip(o["dce"], 0, 1)
        if "gain" in o:
            vis["Amplification gain G"] = colorize(np.log(np.clip(o["gain"][..., 0], 1, None)))
        vis["Refiner correction |out - DCE|"] = colorize(np.abs(dl - o["dce"]).mean(2))
        vis["Classical branch output"] = cl
        vis["Fusion weight (classical share)"] = colorize(W, vmin=0, vmax=1)
        return out, vis, info
