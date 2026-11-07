"""Tune the input-conditioning band on the validation split + synthetic variants of the
validation ground truths (moderate / extreme / non-uniform). Test sets are never used."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import lumifuse.pipeline as P  # noqa: E402
from lumifuse import metrics as M  # noqa: E402
from lumifuse.utils import load_image  # noqa: E402
from scripts.make_synthetic import camera, field  # noqa: E402

man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
rng = np.random.default_rng(11)
real = [(load_image(os.path.join(ROOT, a)), load_image(os.path.join(ROOT, b))) for a, b in man["val"]]
S = {"val": real, "moderate": [], "extreme": [], "nonuniform": []}
for _, gt in real[:16]:
    S["moderate"].append((camera(gt, rng.uniform(0.08, 0.15), rng).astype(np.float32) / 255, gt))
    S["extreme"].append((camera(gt, rng.uniform(0.003, 0.006), rng).astype(np.float32) / 255, gt))
    S["nonuniform"].append((camera(gt, field(*gt.shape[:2], rng), rng).astype(np.float32) / 255, gt))
lf = P.LumiFuse(device="mps")
lf.nets.pop("dlonly", None)


def score():
    out = []
    for n, data in S.items():
        r = [(M.psnr(o, b), M.ssim(o, b)) for a, b in data for o in [lf.enhance(a, "hybrid")[0]]]
        out.append(f"{n} {np.mean([x[0] for x in r]):.2f}/{np.mean([x[1] for x in r]):.3f}")
    return " | ".join(out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cond"
    if what == "cond":
        for hi in (100.0, 2.5, 1.6, 1.2, 1.0, 0.8):
            P.COND_HI = hi
            print("COND_HI", hi, score(), flush=True)
    elif what == "trust":
        for lo, hi in ((10.0, 11.0), (0.12, 0.40), (0.08, 0.30), (0.06, 0.20), (0.05, 0.15)):
            P.TRUST_LO, P.TRUST_HI = lo, hi
            print("TRUST", lo, hi, score(), flush=True)
    else:
        for a in (0.0, 0.5, 0.75, 1.0):
            P.EXPO_ALPHA = a
            print("EXPO_ALPHA", a, score(), flush=True)
