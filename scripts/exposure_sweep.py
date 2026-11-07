"""Degradation under darker captures: re-shoot 20 unseen LOL-v2 test references at progressively
lower exposure with the Poisson-Gaussian camera model and evaluate every method
(results/exposure_sweep.json)."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lumifuse import metrics as M  # noqa: E402
from lumifuse.pipeline import LumiFuse  # noqa: E402
from lumifuse.utils import load_image  # noqa: E402
from scripts.make_synthetic import camera  # noqa: E402

LEVELS = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
METHODS = {"Classical": "classical", "Zero-DCE (official)": "zerodce", "DL-only": "dlonly", "LumiFuse (hybrid)": "hybrid"}


def main():
    gts = sorted(glob.glob(os.path.join(ROOT, "data/lolv2_real/Test/GT/*.png")))[30:50]  # not used by the synthetic sets
    lf = LumiFuse(device="mps" if torch.backends.mps.is_available() else "cpu")
    res = {m: {"psnr": [], "ssim": []} for m in METHODS}
    for e in LEVELS:
        rng = np.random.default_rng(int(e * 1e4))
        lows = [(camera(load_image(g), e, rng).astype(np.float32) / 255, load_image(g)) for g in gts]
        for name, m in METHODS.items():
            p, s = [], []
            for low, gt in lows:
                o, _, _ = lf.enhance(low, m)
                p.append(M.psnr(o, gt))
                s.append(M.ssim(o, gt))
            res[name]["psnr"].append(float(np.mean(p)))
            res[name]["ssim"].append(float(np.mean(s)))
            print(f"exposure x{e}: {name:20s} PSNR {np.mean(p):.2f} SSIM {np.mean(s):.3f}", flush=True)
    json.dump({"levels": LEVELS, "n_images": len(gts), "results": res},
              open(os.path.join(ROOT, "results", "exposure_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
