"""Synthesise paired test sets for specific lighting conditions from the (unseen)
LOL-v2-real test ground truths, using a physically-motivated camera model:
linearise (gamma 2.2) -> scale exposure (global or spatially varying) ->
Poisson shot noise + Gaussian read noise -> re-encode -> 8-bit quantise.

  moderate     : exposure x0.08-0.15
  extreme      : exposure x0.003-0.006 (near-black)
  nonuniform   : smooth gradient + spotlight, exposure x0.004 .. x0.6 across the frame
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 30


def camera(gt, expo, rng, photons=20000.0, read=0.0015):
    lin = np.power(gt, 2.2) * expo
    noisy = rng.poisson(lin * photons) / photons + rng.normal(0, read, lin.shape)
    return np.round(np.clip(np.power(np.clip(noisy, 0, 1), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def field(h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    spot = np.exp(-(((yy - cy) / (0.25 * h)) ** 2 + ((xx - cx) / (0.25 * w)) ** 2))
    f = np.clip(0.6 * grad ** 2 + spot, 0, 1)
    return (0.004 + 0.596 * f)[..., None]


def main():
    gts = sorted(glob.glob(os.path.join(ROOT, "data/lolv2_real/Test/GT/*.png")))[:N]
    rng = np.random.default_rng(123)
    for cond in ("moderate", "extreme", "nonuniform"):
        for sub in ("low", "high"):
            os.makedirs(os.path.join(ROOT, f"data/synthetic/{cond}/{sub}"), exist_ok=True)
        for p in gts:
            gt = cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255
            if cond == "moderate":
                e = rng.uniform(0.08, 0.15)
            elif cond == "extreme":
                e = rng.uniform(0.003, 0.006)
            else:
                e = field(*gt.shape[:2], rng)
            low = camera(gt, e, rng)
            name = os.path.basename(p)
            cv2.imwrite(os.path.join(ROOT, f"data/synthetic/{cond}/low/{name}"), low[:, :, ::-1])
            cv2.imwrite(os.path.join(ROOT, f"data/synthetic/{cond}/high/{name}"), (gt * 255).round().astype(np.uint8)[:, :, ::-1])
    print("ok", len(gts))


if __name__ == "__main__":
    sys.exit(main())
