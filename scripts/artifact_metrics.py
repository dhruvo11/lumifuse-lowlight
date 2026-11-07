"""Quantify the three artefacts named in advanced requirement 4 (results/artifact_metrics.json):

* over-exposure  - % of pixels the method clipped (max channel >= 0.98) that are NOT clipped in the reference
* colour shift   - mean CIEDE2000 colour difference to the reference (perceptual; < 2 barely visible, > 10 obvious)
                   and mean angle (degrees) between output and reference RGB vectors (hue/cast error,
                   independent of brightness)
* detail loss    - Pearson correlation of Sobel gradient magnitudes with the reference (1 = all edges kept)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from skimage.color import deltaE_ciede2000, rgb2lab  # noqa: E402

from lumifuse import metrics as M  # noqa: E402
from lumifuse.utils import load_image, luminance  # noqa: E402
from scripts.evaluate import MAIN, load_sets, slug  # noqa: E402

OUT = os.path.join(ROOT, "results", "outputs")


def measures(o, g):
    clip_o, clip_g = o.max(2) >= 0.98, g.max(2) >= 0.98
    over = float((clip_o & ~clip_g).mean() * 100)
    de = float(deltaE_ciede2000(rgb2lab(o), rgb2lab(g)).mean())
    m = luminance(g) > 0.05
    on, gn = o[m] + 1e-6, g[m] + 1e-6
    cos = (on * gn).sum(1) / (np.linalg.norm(on, axis=1) * np.linalg.norm(gn, axis=1))
    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
    return over, de, ang, M.edge_preservation(o, g)


def main():
    sets = load_sets()
    res = {}
    for sname in ("LOL-v1 eval15", "LOL-v2-real test (unseen)", "Synthetic moderate", "Synthetic extreme-dark",
                  "Synthetic non-uniform"):
        res[sname] = {}
        for m in MAIN:
            d = os.path.join(OUT, slug(sname), slug(m))
            rows = []
            for low, gt in sets[sname]:
                p = os.path.join(d, os.path.splitext(os.path.basename(low))[0] + ".png")
                if os.path.exists(p):
                    rows.append(measures(load_image(p), load_image(gt)))
            if rows:
                a = np.array(rows)
                res[sname][m] = {"overexposed_pct": float(a[:, 0].mean()), "deltaE2000": float(a[:, 1].mean()),
                                 "colour_angle_deg": float(a[:, 2].mean()), "edge_corr": float(a[:, 3].mean()),
                                 "n": len(rows)}
                r = res[sname][m]
                print(f"{sname:26s} {m:36s} over {r['overexposed_pct']:.3f}%  dE {r['deltaE2000']:.2f}  "
                      f"angle {r['colour_angle_deg']:.2f}°  edge {r['edge_corr']:.3f}", flush=True)
    json.dump(res, open(os.path.join(ROOT, "results", "artifact_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
