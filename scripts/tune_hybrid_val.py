"""Tune the hybrid post-processing thresholds on the VALIDATION split only (never the test sets)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lumifuse.pipeline as P  # noqa: E402
from lumifuse import metrics as M  # noqa: E402
from lumifuse.utils import load_image  # noqa: E402

man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
val = [(load_image(os.path.join(ROOT, a)), load_image(os.path.join(ROOT, b))) for a, b in man["val"]]
lf = P.LumiFuse(device="mps" if torch.backends.mps.is_available() else "cpu")


def score(cfg=None):
    ps, ss = [], []
    for a, b in val:
        o, _, _ = lf.enhance(a, "hybrid", hcfg=cfg)
        ps.append(M.psnr(o, b))
        ss.append(M.ssim(o, b))
    return round(float(np.mean(ps)), 3), round(float(np.mean(ss)), 4)


print("network only (no fusion/gates/highlight):", score(P.HybridConfig(use_fusion=False, use_gates=False, use_highlight=False)))
for ng in (0.010, 0.012, 0.015, 0.02, 1.0):
    P.NOISE_GATE = ng
    print("noise gate", ng, score())
P.NOISE_GATE = 0.012
for lo, hi in ((0.30, 0.62), (0.25, 0.65), (0.35, 0.60)):
    P.EXPOSURE_RANGE = (lo, hi)
    print("exposure range", lo, hi, score())
