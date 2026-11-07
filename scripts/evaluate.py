"""Benchmark every method + the ablation variants on every test set.

  python scripts/evaluate.py                 # everything (results cached per set/method)
  python scripts/evaluate.py --only main     # just the 4-way comparison
  python scripts/evaluate.py --latency       # CPU latency benchmark only

Writes results/raw/<set>__<variant>.json (per-image) and results/summary.json.
Metric order follows the problem statement: PSNR, SSIM, LPIPS, NIQE, latency,
noise suppression (residual noise sigma), edge preservation.
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lumifuse import metrics as M  # noqa: E402
from lumifuse.classical import ClassicalConfig  # noqa: E402
from lumifuse.pipeline import HybridConfig, LumiFuse  # noqa: E402
from lumifuse.utils import load_image, resize_max_side, save_image  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
OUT_IMG = os.path.join(ROOT, "results", "outputs")

MAIN = {
    "Classical": ("classical", None, None),
    "Zero-DCE (official, no fine-tune)": ("zerodce", None, None),
    "DL-only (LumiNet w/o priors)": ("dlonly", None, None),
    "Hybrid (LumiFuse, proposed)": ("hybrid", None, None),
}
ABLATION = {
    "Full hybrid": ("hybrid", HybridConfig(), None),
    "- classical priors (net w/o L, G)": ("hybrid", HybridConfig(use_prior=False), None),
    "- RefineNet (curve stage only)": ("hybrid", HybridConfig(use_refine=False), None),
    "- LOL fine-tuning (official Zero-DCE)": ("hybrid", HybridConfig(official_dce=True), None),
    "- reliability fusion (all)": ("hybrid", HybridConfig(use_fusion=False), None),
    "- global classical trust only": ("hybrid", HybridConfig(global_trust=False), None),
    "- quality gates": ("hybrid", HybridConfig(use_gates=False), None),
    "- input conditioning": ("hybrid", HybridConfig(input_conditioning=False), None),
    "- highlight protection": ("hybrid", HybridConfig(use_highlight=False), None),
}
CLASSICAL_ABL = {
    "Classical (adaptive, full)": ("classical", None, ClassicalConfig()),
    "Classical fixed params": ("classical", None, ClassicalConfig(adaptive=False)),
    "- Retinex": ("classical", None, ClassicalConfig(use_retinex=False)),
    "- denoising": ("classical", None, ClassicalConfig(use_denoise=False)),
    "bilateral instead of guided": ("classical", None, ClassicalConfig(denoiser="bilateral")),
    "- CLAHE": ("classical", None, ClassicalConfig(use_clahe=False)),
    "- colour correction": ("classical", None, ClassicalConfig(use_color=False)),
}


def load_sets():
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    sets = {
        "LOL-v1 eval15": [(os.path.join(ROOT, a), os.path.join(ROOT, b)) for a, b in man["test_lolv1"]],
        "LOL-v2-real test (unseen)": [(os.path.join(ROOT, a), os.path.join(ROOT, b)) for a, b in man["test_lolv2"]],
    }
    for c, name in (("moderate", "Synthetic moderate"), ("extreme", "Synthetic extreme-dark"),
                    ("nonuniform", "Synthetic non-uniform")):
        lows = sorted(glob.glob(os.path.join(ROOT, f"data/synthetic/{c}/low/*.png")))
        sets[name] = [(p, p.replace("/low/", "/high/")) for p in lows]
    zd = os.path.join(ROOT, "data/zdce/Zero-DCE-master/Zero-DCE_code/data/test_data")
    real = sorted(glob.glob(f"{zd}/DICM/*")) + sorted(glob.glob(f"{zd}/LIME/*"))
    sets["Real unpaired DICM+LIME"] = [(p, None) for p in real]
    return sets


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def run(lf, sets, variants, tag, save_outputs=False):
    os.makedirs(RAW, exist_ok=True)
    for sname, items in sets.items():
        for vname, (method, hcfg, ccfg) in variants.items():
            path = os.path.join(RAW, f"{slug(sname)}__{slug(tag + '_' + vname)}.json")
            if os.path.exists(path):
                continue
            rows = []
            t_set = time.time()
            for low_p, gt_p in items:
                img = load_image(low_p)
                if gt_p is None:
                    img = resize_max_side(img, 1024)
                t = time.perf_counter()
                out, _, info = lf.enhance(img, method, hcfg=hcfg, ccfg=ccfg)
                ms = 1000 * (time.perf_counter() - t)
                r = {"name": os.path.basename(low_p), "ms": ms, "niqe": M.niqe(out), "noise": M.noise_sigma(out),
                     "mean_luma": float(out.mean())}
                if gt_p is not None:
                    gt = load_image(gt_p)
                    r.update(psnr=M.psnr(out, gt), ssim=M.ssim(out, gt), lpips=M.lpips(out, gt),
                             edge=M.edge_preservation(out, gt))
                rows.append(r)
                if save_outputs:
                    d = os.path.join(OUT_IMG, slug(sname), slug(vname))
                    os.makedirs(d, exist_ok=True)
                    save_image(os.path.join(d, os.path.splitext(os.path.basename(low_p))[0] + ".png"), out)
            json.dump(rows, open(path, "w"))
            keys = [k for k in ("psnr", "ssim", "lpips", "niqe") if k in rows[0]]
            print(f"[{sname}] {vname}: " + " ".join(f"{k}={np.nanmean([r[k] for r in rows]):.4f}" for k in keys)
                  + f" ({time.time() - t_set:.0f}s)", flush=True)


def latency(lf, n=15, threads=4):
    """CPU latency on 600x400 LOL images (what the deployed CPU Space sees)."""
    torch.set_num_threads(threads)
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    imgs = [load_image(os.path.join(ROOT, a)) for a, _ in man["test_lolv1"][:n]]
    res = {}
    for vname, (method, hcfg, ccfg) in MAIN.items():
        lf.enhance(imgs[0], method)  # warm-up
        ts = []
        for im in imgs:
            t = time.perf_counter()
            lf.enhance(im, method, hcfg=hcfg, ccfg=ccfg)
            ts.append(1000 * (time.perf_counter() - t))
        res[vname] = {"mean_ms": float(np.mean(ts)), "std_ms": float(np.std(ts))}
        print(f"latency {vname}: {np.mean(ts):.1f} +- {np.std(ts):.1f} ms", flush=True)
    json.dump({"device": f"Apple M1 CPU, {threads} threads", "resolution": "600x400", "results": res},
              open(os.path.join(ROOT, "results", "latency.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["main", "ablation", "all"], default="all")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    if args.latency:
        latency(LumiFuse(device="cpu"))
        return
    lf = LumiFuse(device=args.device)
    sets = load_sets()
    run(lf, sets, MAIN, "main", save_outputs=True)
    if args.only in ("ablation", "all"):
        paired = {k: v for k, v in sets.items() if k in ("LOL-v1 eval15", "LOL-v2-real test (unseen)",
                                                         "Synthetic extreme-dark", "Synthetic non-uniform")}
        run(lf, paired, ABLATION, "abl")
        run(lf, paired, CLASSICAL_ABL, "cabl")


if __name__ == "__main__":
    main()
