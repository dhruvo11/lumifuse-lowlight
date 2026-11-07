"""Write eval_<ProbID>_<TeamID>.md from results/summary.json + team.json.
Line 1 = comma-separated USNs (submission requirement). Metric order follows the
problem statement: PSNR, SSIM, LPIPS, NIQE, latency, noise suppression, edge preservation."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.summarize import COLS  # noqa: E402

team = json.load(open(os.path.join(ROOT, "team.json")))
sm = json.load(open(os.path.join(ROOT, "results", "summary.json")))
lat = json.load(open(os.path.join(ROOT, "results", "latency.json")))
man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))


def tbl(key, title, note=""):
    d = sm.get(key)
    if not d:
        return ""
    cols = [c for c in COLS if any(d[v].get(c[0]) is not None for v in d)]
    best = {}
    for k, _, _ in cols:
        vals = {v: d[v][k] for v in d if d[v][k] is not None}
        if vals and k != "ms":
            best[k] = (max if k in ("psnr", "ssim", "edge") else min)(vals, key=vals.get)
    out = [f"### {title}", note, "| Method | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for v in d:
        cells = []
        for k, _, dec in cols:
            x = d[v][k]
            s = "—" if x is None else f"{x:.{dec}f}"
            cells.append(f"**{s}**" if best.get(k) == v else s)
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(x for x in out if x != "") + "\n"


usns = ",".join(m["usn"] for m in team["members"])
st = man["stats"]
L = [usns, "",
     f"# Evaluation results — {team['problem_id']} Robust Low-Light Image Restoration — Team {team['team_id']} ({team['team_name']})",
     "",
     "System: **LumiFuse** (classical Retinex/CLAHE/guided-filter pipeline + illumination-guided Zero-DCE network + "
     "reliability-weighted fusion). All numbers are produced by `python scripts/evaluate.py && python scripts/summarize.py`.",
     "",
     "## Protocol",
     f"* **Training data (leak-free):** LOL-v1 `our485` ∪ LOL-v2-real `Train`, with every pair whose low-light input also "
     f"appears in a test set removed ({st['dropped_test_leak']} pairs) and duplicates removed ({st['dropped_duplicates']}). "
     f"{st['train']} train / {st['val']} validation pairs. Validation (not test) is used for checkpoint and hyper-parameter selection.",
     "* **Paired test sets:** LOL-v1 `eval15` (15), LOL-v2-real `Test` (100, none seen in training — 89 of these are "
     "LOL-v1 training images, which we removed from training), and three synthetic lighting conditions generated from 30 "
     "unseen LOL-v2 ground truths with a Poisson-Gaussian camera noise model: moderate (×0.08–0.15 exposure), "
     "extreme-dark (×0.003–0.006) and non-uniform (gradient + spotlight, ×0.004–0.6 across the frame).",
     "* **Unpaired real test set:** DICM (64) + LIME (10) photographs, resized to ≤1024 px; no-reference metrics only.",
     "* **Metrics (problem-statement order):** PSNR (dB, RGB, data range 1) and SSIM (RGB, skimage, Gaussian-free 7×7 default); "
     "LPIPS (AlexNet, v0.1); NIQE (MATLAB-equivalent BasicSR port, Y channel); processing latency "
     f"({lat['device']}, {lat['resolution']}); noise suppression = residual noise σ of the output (Immerkær estimator on "
     "flat pixels, lower = cleaner); edge preservation = Pearson correlation between Sobel gradient magnitudes of output "
     "and ground truth (higher = edges kept). No GT-mean brightness correction is applied anywhere.",
     "", "## 1. Comparative evaluation (classical vs deep learning vs hybrid)", ""]
L.append(tbl("main/LOL-v1 eval15", "1.1 LOL-v1 eval15 (standard benchmark)",
             "Latency = CPU benchmark (Apple M1, 4 threads, 600×400)."))
L.append(tbl("main/LOL-v2-real test (unseen)", "1.2 LOL-v2-real test — 100 unseen images"))
L.append("## 2. Generalisation across illumination conditions (unseen scenes)\n")
for k, t in (("Synthetic moderate", "2.1 Moderate low light"), ("Synthetic extreme-dark", "2.2 Extremely dark"),
             ("Synthetic non-uniform", "2.3 Non-uniform / spotlight illumination"),
             ("Real unpaired DICM+LIME", "2.4 Real unpaired photos (DICM + LIME) — no-reference")):
    L.append(tbl(f"main/{k}", t))
L.append("## 3. Processing latency (CPU, deployment hardware class)\n")
L.append("| Method | Mean (ms) | Std (ms) |\n|---|---:|---:|")
for k, v in lat["results"].items():
    L.append(f"| {k} | {v['mean_ms']:.0f} | {v['std_ms']:.0f} |")
L.append("")
L.append("## 4. Ablation study\n")
for k in ("LOL-v1 eval15", "LOL-v2-real test (unseen)", "Synthetic extreme-dark", "Synthetic non-uniform"):
    L.append(tbl(f"abl/{k}", f"4.x Hybrid — remove one component — {k}",
                 "Latency column here is evaluation wall-clock (GPU/MPS), not the CPU benchmark."))
for k in ("LOL-v1 eval15", "LOL-v2-real test (unseen)", "Synthetic extreme-dark", "Synthetic non-uniform"):
    L.append(tbl(f"cabl/{k}", f"4.y Classical pipeline — remove one component — {k}"))
L.append("## 5. Failure-case analysis\nSee `docs/ablation_and_failures.md` (worst cases, root causes, mitigations) and "
         "`docs/figures/failure_cases.png`.\n")
path = os.path.join(ROOT, f"eval_{team['problem_id']}_{team['team_id']}.md")
for old in glob.glob(os.path.join(ROOT, "eval_*.md")):
    if old != path:
        os.remove(old)
open(path, "w").write("\n".join(L))
print("wrote", path)
