"""Failure-case diagnostics (writes results/failure_analysis.json).

For each method on each paired set, PSNR is recomputed after scaling the output so its
mean luminance equals the ground truth's ("GT-mean" correction). This is a DIAGNOSTIC
only (it uses the ground truth, so it is never reported as a score): the gap between
plain and GT-mean PSNR tells how much error is global exposure vs. restoration error.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from lumifuse import metrics as M  # noqa: E402
from lumifuse.utils import load_image, luminance  # noqa: E402
from scripts.evaluate import MAIN, load_sets, slug  # noqa: E402

OUT = os.path.join(ROOT, "results", "outputs")


def gt_mean(o, g):
    return np.clip(o * (luminance(g).mean() / max(luminance(o).mean(), 1e-4)), 0, 1)


def main():
    sets = load_sets()
    res = {}
    for sname in ("LOL-v1 eval15", "LOL-v2-real test (unseen)", "Synthetic moderate", "Synthetic extreme-dark",
                  "Synthetic non-uniform"):
        res[sname] = {}
        for m in MAIN:
            d = os.path.join(OUT, slug(sname), slug(m))
            if not os.path.isdir(d):
                continue
            rows = []
            for low, gt in sets[sname]:
                p = os.path.join(d, os.path.splitext(os.path.basename(low))[0] + ".png")
                if not os.path.exists(p):
                    continue
                o, g = load_image(p), load_image(gt)
                rows.append({"name": os.path.basename(low), "psnr": M.psnr(o, g), "psnr_gtmean": M.psnr(gt_mean(o, g), g),
                             "out_luma": float(luminance(o).mean()), "gt_luma": float(luminance(g).mean())})
            if rows:
                res[sname][m] = {
                    "psnr": float(np.mean([r["psnr"] for r in rows])),
                    "psnr_gtmean_diagnostic": float(np.mean([r["psnr_gtmean"] for r in rows])),
                    "mean_abs_exposure_error": float(np.mean([abs(r["out_luma"] - r["gt_luma"]) for r in rows])),
                    "worst": sorted(rows, key=lambda r: r["psnr"])[:5],
                }
                print(f"{sname:28s} {m:38s} psnr {res[sname][m]['psnr']:.2f}  gt-mean diag "
                      f"{res[sname][m]['psnr_gtmean_diagnostic']:.2f}  |exp err| {res[sname][m]['mean_abs_exposure_error']:.3f}")
    json.dump(res, open(os.path.join(ROOT, "results", "failure_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
