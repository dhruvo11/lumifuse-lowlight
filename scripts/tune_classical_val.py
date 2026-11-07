"""Tune classical adaptive rules on the validation split + synthetic extreme/non-uniform
versions of the validation ground truths (test sets are never touched)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from lumifuse import metrics as M  # noqa: E402
from lumifuse.classical import ClassicalConfig, enhance_classical  # noqa: E402
from lumifuse.utils import load_image  # noqa: E402
from scripts.make_synthetic import camera, field  # noqa: E402

man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
rng = np.random.default_rng(7)
real = [(load_image(os.path.join(ROOT, a)), load_image(os.path.join(ROOT, b))) for a, b in man["val"]]
ext, nonu = [], []
for _, gt in real[:20]:
    ext.append((camera(gt, rng.uniform(0.003, 0.006), rng).astype(np.float32) / 255, gt))
    nonu.append((camera(gt, field(*gt.shape[:2], rng), rng).astype(np.float32) / 255, gt))
SETS = {"val": real, "val-extreme": ext, "val-nonuniform": nonu}


def score(cfg):
    out = []
    for name, data in SETS.items():
        p, s = [], []
        for a, b in data:
            o, _, _ = enhance_classical(a, cfg)
            p.append(M.psnr(o, b))
            s.append(M.ssim(o, b))
        out.append(f"{name} {np.mean(p):.2f}/{np.mean(s):.3f}")
    return " | ".join(out)


if __name__ == "__main__":
    for kw in [dict(stretch_max_sigma=0.02, target=0.40), dict(stretch_max_sigma=0.02, dn_floor=0.85),
               dict(stretch_max_sigma=0.02, target=0.5)]:
        print(kw, score(ClassicalConfig(**kw)))
