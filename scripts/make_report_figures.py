"""Charts for the technical report (report/figures/*.pdf), all read from results/*.json and training logs."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)
SM = json.load(open(os.path.join(ROOT, "results", "summary.json")))
LAT = json.load(open(os.path.join(ROOT, "results", "latency.json")))["results"]
M = ["Classical", "Zero-DCE (official, no fine-tune)", "DL-only (LumiNet w/o priors)", "Hybrid (LumiFuse, proposed)"]
LAB = ["Classical", "Zero-DCE (official)", "DL-only", "LumiFuse (hybrid)"]
COL = ["#2f7ed8", "#e8673c", "#8e6bbf", "#1fa774"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "axes.titleweight": "bold",
                     "axes.titlesize": 10})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1. training curves -------------------------------------------------------------------------
def val_curve(path):
    its, ps = [], []
    for line in open(path):
        m = re.match(r"VAL it (\d+) psnr ([\d.]+)", line)
        if m:
            its.append(int(m.group(1)))
            ps.append(float(m.group(2)))
    return its, ps


fig, ax = plt.subplots(figsize=(6.4, 3.0))
for f, lab, c, mk in (("train_hybrid.log", "LumiNet with classical priors (hybrid)", COL[3], "o"),
                      ("train_dlonly.log", "LumiNet without priors (DL-only)", COL[2], "s")):
    its, ps = val_curve(os.path.join(ROOT, "results", f))
    ax.plot(np.array(its) / 1000, ps, marker=mk, color=c, label=lab, lw=1.8, ms=4)
cls_val = 15.47  # adaptive classical pipeline on the same validation split (scripts/tune_classical_val.py)
ax.axhline(cls_val, color=COL[0], lw=1.5)
ax.text(3.95, cls_val + 0.12, f"Classical pipeline ({cls_val:.2f} dB)", color=COL[0], ha="right", fontsize=8)
ax.set_xlabel("Training step (thousands)")
ax.set_ylabel("Validation PSNR (dB)")
ax.set_title("Validation PSNR during training (held-out validation split)")
ax.legend(loc="lower right", fontsize=8, frameon=False)
save(fig, "training_curves.pdf")

# 2. method bars (LOL-v1 and LOL-v2) ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (k, title, d) in zip(axes, (("psnr", "PSNR (dB), higher is better", 2), ("ssim", "SSIM, higher is better", 3),
                                     ("lpips", "LPIPS, lower is better", 3))):
    y = np.arange(4)
    for j, (s, h, alpha) in enumerate((("LOL-v1 eval15", -0.2, 1.0), ("LOL-v2-real test (unseen)", 0.2, 0.55))):
        v = [SM[f"main/{s}"][m][k] for m in M]
        bars = ax.barh(y + h, v, height=0.38, color=COL, alpha=alpha)
        for b, val in zip(bars, v):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {val:.{d}f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(LAB if ax is axes[0] else [], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, max(SM[f"main/{s}"][m][k] for m in M for s in ("LOL-v1 eval15", "LOL-v2-real test (unseen)")) * 1.28)
fig.text(0.5, -0.04, "Solid bars: LOL-v1 eval15 (15 images).  Light bars: LOL-v2-real test (100 unseen images).",
         ha="center", fontsize=8)
save(fig, "method_bars.pdf")

# 3. quality vs latency ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for m, lab, c in zip(M, LAB, COL):
    x = LAT[m]["mean_ms"]
    y = SM["main/LOL-v1 eval15"][m]["psnr"]
    ax.scatter(x, y, s=60, color=c, zorder=3)
    ax.annotate(f"{lab}\n{y:.2f} dB, {x:.0f} ms", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=7.5)
ax.set_xscale("log")
ax.set_xlim(40, 1500)
ax.set_xlabel("CPU latency per 600x400 frame (ms, log scale; Apple M1, 4 threads)")
ax.set_ylabel("LOL-v1 PSNR (dB)")
ax.set_title("Quality vs speed")
save(fig, "quality_vs_latency.pdf")

# 4. robustness by test set --------------------------------------------------------------------
sets = [("LOL-v1 eval15", "LOL-v1\n(n=15)"), ("LOL-v2-real test (unseen)", "LOL-v2\n(n=100)"),
        ("Synthetic moderate", "Moderate\n(n=30)"), ("Synthetic extreme-dark", "Extreme-dark\n(n=30)"),
        ("Synthetic non-uniform", "Non-uniform\n(n=30)")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
for ax, k, t in zip(axes, ("psnr", "ssim"), ("PSNR (dB) by test set", "SSIM by test set")):
    x = np.arange(len(sets))
    for i, (m, lab, c) in enumerate(zip(M, LAB, COL)):
        ax.bar(x + (i - 1.5) * 0.2, [SM[f"main/{s}"][m][k] for s, _ in sets], width=0.2, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in sets], fontsize=7)
    ax.set_title(t)
    if k == "psnr":
        ax.set_ylim(8, 21)
axes[0].legend(fontsize=7, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.28))
save(fig, "by_condition.pdf")

# 5. exposure sweep ---------------------------------------------------------------------------------
p = os.path.join(ROOT, "results", "exposure_sweep.json")
if os.path.exists(p):
    E = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    lv = E["levels"]
    for ax, k in zip(axes, ("psnr", "ssim")):
        for (name, r), c, mk in zip(E["results"].items(), COL, "osD^"):
            ax.plot(range(len(lv)), r[k], marker=mk, color=c, label=name, lw=1.8, ms=4)
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([f"x{v:g}" for v in lv], fontsize=7)
        ax.set_xlabel("Exposure relative to the reference (darker to the right)")
        ax.set_title(("PSNR (dB)" if k == "psnr" else "SSIM") + " as the capture gets darker")
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "exposure_sweep.pdf")

# 6. ablation deltas ----------------------------------------------------------------------------------
AB = ["- classical priors (net w/o L, G)", "- RefineNet (curve stage only)", "- LOL fine-tuning (official Zero-DCE)",
      "- input conditioning", "- reliability fusion (all)", "- global classical trust only", "- quality gates",
      "- highlight protection"]
ABN = ["No classical priors (L, G)", "No RefineNet", "No LOL fine-tuning", "No input conditioning",
       "No reliability fusion", "No global trust", "No quality gates", "No highlight protection"]
asets = [("LOL-v1 eval15", "LOL-v1"), ("LOL-v2-real test (unseen)", "LOL-v2"), ("Synthetic extreme-dark", "Extreme-dark"),
         ("Synthetic non-uniform", "Non-uniform")]
fig, ax = plt.subplots(figsize=(7.0, 3.6))
y = np.arange(len(AB))
for j, ((s, lab), c) in enumerate(zip(asets, ["#2f7ed8", "#7fb3ea", "#e8673c", "#1fa774"])):
    full = SM[f"abl/{s}"]["Full hybrid"]["psnr"]
    d = [SM[f"abl/{s}"][a]["psnr"] - full for a in AB]
    ax.barh(y + (j - 1.5) * 0.2, d, height=0.2, color=c, label=lab)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(ABN, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Change in PSNR (dB) relative to the full system (negative = component helps)")
ax.set_title("Ablation: what each component is worth")
ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.32))
save(fig, "ablation_deltas.pdf")
