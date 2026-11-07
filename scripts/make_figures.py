"""Figures for the docs: method comparison grids, intermediate maps, failure cases, ablation chart."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lumifuse.pipeline import LumiFuse  # noqa: E402
from lumifuse.utils import load_image, resize_max_side  # noqa: E402
from scripts.evaluate import MAIN, slug  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "results", "outputs")
SHORT = {"Classical": "Classical", "Zero-DCE (official, no fine-tune)": "Zero-DCE (official)",
         "DL-only (LumiNet w/o priors)": "DL-only", "Hybrid (LumiFuse, proposed)": "Hybrid (ours)"}


def grid(rows, cols, panels, path, cell=2.6, title=None):
    h0, w0 = panels[0][0][0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols * w0 / h0 * 0.75, cell * rows * 0.75 + (0.4 if title else 0)))
    axes = np.atleast_2d(axes)
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(panels[i]):
                im, lab = panels[i][j]
                ax.imshow(np.clip(im, 0, 1))
                ax.set_title(lab, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout(h_pad=1.6)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def method_row(set_slug, name, low_path, gt_path=None, maxs=None):
    low = load_image(low_path)
    if maxs:
        low = resize_max_side(low, maxs)
    row = [(low, "Input")]
    for m in MAIN:
        p = os.path.join(OUT, set_slug, slug(m), name + ".png")
        if os.path.exists(p):
            row.append((load_image(p), SHORT[m]))
    if gt_path:
        row.append((load_image(gt_path), "Ground truth"))
    return row


def main():
    os.makedirs(FIG, exist_ok=True)
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    # 1. LOL comparison
    rows = []
    for lo, hi in man["test_lolv1"][:15:4]:
        n = os.path.splitext(os.path.basename(lo))[0]
        rows.append(method_row(slug("LOL-v1 eval15"), n, os.path.join(ROOT, lo), os.path.join(ROOT, hi)))
    grid(len(rows), 6, rows, os.path.join(FIG, "comparison_lol.png"), title="LOL-v1 eval15: classical vs deep vs hybrid")

    # 2. lighting conditions: synthetic extreme / non-uniform (paired) + real photos
    rows = []
    for cond, sname in (("extreme", "Synthetic extreme-dark"), ("nonuniform", "Synthetic non-uniform")):
        for f in sorted(glob.glob(os.path.join(ROOT, f"data/synthetic/{cond}/low/*.png")))[2:4]:
            n = os.path.splitext(os.path.basename(f))[0]
            rows.append(method_row(slug(sname), n, f, f.replace("/low/", "/high/")))
    grid(len(rows), 6, rows, os.path.join(FIG, "conditions_synthetic.png"),
         title="Extreme-dark (rows 1-2) and non-uniform / spotlight illumination (rows 3-4), unseen scenes")
    zd = os.path.join(ROOT, "data/zdce/Zero-DCE-master/Zero-DCE_code/data/test_data")
    rows = []
    for f in (f"{zd}/DICM/27.jpg", f"{zd}/DICM/07.jpg", f"{zd}/LIME/10.bmp", f"{zd}/DICM/45.jpg"):
        n = os.path.splitext(os.path.basename(f))[0]
        rows.append(method_row(slug("Real unpaired DICM+LIME"), n, f, maxs=1024))
    grid(len(rows), 5, rows, os.path.join(FIG, "conditions_real.png"),
         title="Real unseen photos (DICM / LIME): very dark, backlit, night street, indoor")

    # 3. intermediate maps
    lf = LumiFuse(device="cpu")
    for tag, f in (("lol", os.path.join(ROOT, man["test_lolv1"][0][0])), ("backlit", f"{zd}/DICM/07.jpg")):
        img = resize_max_side(load_image(f), 800)
        out, vis, info = lf.enhance(img, "hybrid")
        panels = [(img, "Input")] + [(v, k) for k, v in vis.items()] + [(out, "Final output")]
        r1 = panels[:6]
        r2 = panels[6:12]
        grid(2, 6, [r1, r2], os.path.join(FIG, f"intermediates_{tag}.png"), title="Hybrid pipeline intermediate states")
        _, cvis, _ = lf.enhance(img, "classical")
        cp = [(img, "Input")] + [(v, k) for k, v in cvis.items()]
        grid(1, len(cp), [cp], os.path.join(FIG, f"intermediates_classical_{tag}.png"), title="Classical branch stages")

    # 4. failure cases: worst hybrid PSNR on LOL-v2 + a real-photo case
    raw = json.load(open(os.path.join(ROOT, "results", "raw",
                                      f"{slug('LOL-v2-real test (unseen)')}__{slug('main_Hybrid (LumiFuse, proposed)')}.json")))
    worst = sorted(raw, key=lambda r: r["psnr"])[:3]
    rows = []
    for r in worst:
        n = os.path.splitext(r["name"])[0]
        lo = os.path.join(ROOT, "data/lolv2_real/Test/Input", r["name"])
        gt = os.path.join(ROOT, "data/lolv2_real/Test/GT", r["name"])
        o = load_image(os.path.join(OUT, slug("LOL-v2-real test (unseen)"), slug("Hybrid (LumiFuse, proposed)"), n + ".png"))
        g = load_image(gt)
        err = np.abs(o - g).mean(2)
        err = plt.cm.magma(np.clip(err / 0.3, 0, 1))[..., :3]
        rows.append([(load_image(lo), f"Input {r['name']}"), (o, f"Hybrid  PSNR {r['psnr']:.1f}"), (g, "Ground truth"),
                     (err, "|error| (0-0.3)")])
    grid(len(rows), 4, rows, os.path.join(FIG, "failure_cases.png"), title="Worst three hybrid results on LOL-v2-real test")

    # 5. ablation chart
    sm = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ("psnr", "ssim")):
        for k, s in enumerate(("LOL-v1 eval15", "LOL-v2-real test (unseen)")):
            d = sm.get(f"abl/{s}", {})
            names = list(d)
            vals = [d[n][key] for n in names]
            y = np.arange(len(names))
            ax.barh(y + k * 0.4, vals, height=0.4, label=s, color=["#2a6fdb", "#e0892b"][k])
            ax.set_yticks(y + 0.2)
            ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(key.upper())
        lo = min(min(d[n][key] for n in d) for s in ("LOL-v1 eval15", "LOL-v2-real test (unseen)")
                 for d in [sm.get(f"abl/{s}", {})] if d)
        ax.set_xlim(lo * 0.95, None)
        ax.legend(fontsize=8)
    fig.suptitle("Ablation: removing one hybrid component at a time")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"), dpi=110)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
