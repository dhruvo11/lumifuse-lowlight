"""Insert auto-generated tables into the markdown docs between markers
    <!-- AUTO:key -->  ...  <!-- /AUTO:key -->
so that README, design doc, report and failure analysis always show the numbers in
results/summary.json, results/latency.json and results/failure_analysis.json."""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.summarize import COLS  # noqa: E402

SM = json.load(open(os.path.join(ROOT, "results", "summary.json")))
LAT = json.load(open(os.path.join(ROOT, "results", "latency.json")))
FA = json.load(open(os.path.join(ROOT, "results", "failure_analysis.json")))
TEAM = json.load(open(os.path.join(ROOT, "team.json")))
SHORT = {"Classical": "Classical only", "Zero-DCE (official, no fine-tune)": "Zero-DCE (official weights)",
         "DL-only (LumiNet w/o priors)": "Deep learning only (LumiNet, no priors)",
         "Hybrid (LumiFuse, proposed)": "**Hybrid — LumiFuse (proposed)**"}


def md_table(d, keys=None, latency=False):
    if not d:
        return "_not available_"
    if latency:
        d = {v: dict(r, ms=LAT["results"].get(v, {}).get("mean_ms", r.get("ms"))) for v, r in d.items()}
    cols = [c for c in COLS if (keys is None or c[0] in keys) and any(d[v].get(c[0]) is not None for v in d)]
    best = {}
    for k, _, _ in cols:
        vals = {v: d[v][k] for v in d if d[v].get(k) is not None}
        if vals and k != "ms":
            best[k] = (max if k in ("psnr", "ssim", "edge") else min)(vals, key=vals.get)
    out = ["| Method | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for v in d:
        cells = []
        for k, _, dec in cols:
            x = d[v].get(k)
            s = "—" if x is None else f"{x:.{dec}f}"
            cells.append(f"**{s}**" if best.get(k) == v else s)
        out.append(f"| {SHORT.get(v, v)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def glance():
    rows = ["| Test set (all unseen) | Classical | Zero-DCE | DL-only | **Hybrid** |", "|---|---:|---:|---:|---:|"]
    order = list(SHORT)
    for s, lab in (("LOL-v1 eval15", "LOL-v1 eval15 — PSNR / SSIM"), ("LOL-v2-real test (unseen)", "LOL-v2-real test (100) — PSNR / SSIM"),
                   ("Synthetic extreme-dark", "Extreme-dark — PSNR / SSIM"), ("Synthetic non-uniform", "Non-uniform light — PSNR / SSIM")):
        d = SM.get(f"main/{s}", {})
        cells = [f"{d[m]['psnr']:.2f} / {d[m]['ssim']:.3f}" if m in d else "—" for m in order]
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    d = SM.get("main/Real unpaired DICM+LIME", {})
    rows.append("| Real DICM+LIME photos — NIQE ↓ | " + " | ".join(f"{d[m]['niqe']:.2f}" if m in d else "—" for m in order) + " |")
    rows.append("| CPU latency, 600×400 (ms) | " + " | ".join(
        f"{LAT['results'][m]['mean_ms']:.0f}" if m in LAT["results"] else "—" for m in order) + " |")
    return "\n".join(rows)


def latency_table():
    out = [f"Measured on {LAT['device']}, {LAT['resolution']}, mean of 15 LOL images after warm-up.", "",
           "| Method | Mean (ms) | Std (ms) |", "|---|---:|---:|"]
    out += [f"| {SHORT.get(k, k)} | {v['mean_ms']:.0f} | {v['std_ms']:.0f} |" for k, v in LAT["results"].items()]
    return "\n".join(out)


def failure_table():
    out = ["| Test set | Method | PSNR | PSNR after GT-mean exposure match (diagnostic) | Mean abs. exposure error |",
           "|---|---|---:|---:|---:|"]
    for s, d in FA.items():
        for m, r in d.items():
            out.append(f"| {s} | {SHORT.get(m, m)} | {r['psnr']:.2f} | {r['psnr_gtmean_diagnostic']:.2f} | "
                       f"{r['mean_abs_exposure_error']:.3f} |")
    return "\n".join(out)


def worst_table():
    d = FA.get("LOL-v2-real test (unseen)", {}).get("Hybrid (LumiFuse, proposed)", {})
    out = ["| Image | Hybrid PSNR | Output brightness | GT brightness |", "|---|---:|---:|---:|"]
    for r in d.get("worst", []):
        out.append(f"| {r['name']} | {r['psnr']:.2f} | {r['out_luma']:.3f} | {r['gt_luma']:.3f} |")
    return "\n".join(out)


def report_table(sname):
    d = SM.get(f"main/{sname}", {})
    names = [("Classical Baseline", "Classical"), ("Deep Learning Only (LumiNet, no classical priors)", "DL-only (LumiNet w/o priors)"),
             ("Deep Learning — official Zero-DCE (reference)", "Zero-DCE (official, no fine-tune)"),
             ("**Proposed Hybrid System (LumiFuse)**", "Hybrid (LumiFuse, proposed)")]
    out = ["| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual (LPIPS ↓ / NIQE ↓) | Latency (ms / frame, CPU) |",
           "| :--- | :---: | :---: | :---: | :---: |"]
    for lab, k in names:
        if k not in d:
            continue
        r = d[k]
        ms = LAT["results"].get(k, {}).get("mean_ms")
        out.append(f"| {lab} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['lpips']:.4f} / {r['niqe']:.3f} | "
                   + (f"{ms:.0f} ms" if ms else "—") + " |")
    return "\n".join(out)


BLOCKS = {
    "report_lolv1": lambda: report_table("LOL-v1 eval15"),
    "report_lolv2": lambda: report_table("LOL-v2-real test (unseen)"),
    "report_ext": lambda: report_table("Synthetic extreme-dark"),
    "glance": glance,
    "latency": latency_table,
    "failure_diag": failure_table,
    "worst_v2": worst_table,
}
for s, key in (("LOL-v1 eval15", "lolv1"), ("LOL-v2-real test (unseen)", "lolv2"), ("Synthetic moderate", "mod"),
               ("Synthetic extreme-dark", "ext"), ("Synthetic non-uniform", "nonu"), ("Real unpaired DICM+LIME", "real")):
    BLOCKS[f"main_{key}"] = (lambda s=s: md_table(SM.get(f"main/{s}"), latency=True))
    BLOCKS[f"abl_{key}"] = (lambda s=s: md_table(SM.get(f"abl/{s}"), keys=("psnr", "ssim", "lpips", "niqe", "noise", "edge")))
    BLOCKS[f"cabl_{key}"] = (lambda s=s: md_table(SM.get(f"cabl/{s}"), keys=("psnr", "ssim", "lpips", "niqe", "noise", "edge")))

SUBS = {
    "<<TEAM>>": f"{TEAM['team_id']} · {TEAM['team_name']} · " + ", ".join(f"{m['name']} ({m['usn']})" for m in TEAM["members"]),
    "<<GITHUB>>": TEAM["github_url"],
    "<<LIVE_URL>>": TEAM["live_url"],
    "<<EVAL_FILE>>": f"eval_{TEAM['problem_id']}_{TEAM['team_id']}.md",
    "<<TEAM_NAME>>": TEAM["team_name"],
    "<<TEAM_ID>>": TEAM["team_id"],
    "<<MEMBERS>>": "<br>".join(f"{m['name']} — {m['usn']}" for m in TEAM["members"]),
    "<<COMMIT>>": os.popen(f"git -C '{ROOT}' rev-parse --short HEAD 2>/dev/null").read().strip() or "see repo",
}

files = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "Hackathon_Solution_Report_P1.md")] + \
    glob.glob(os.path.join(ROOT, "docs", "*.md"))
for f in files:
    if not os.path.exists(f):
        continue
    txt = open(f).read()
    for key, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- AUTO:{key} -->)(.*?)(<!-- /AUTO:{key} -->)", re.S)
        if pat.search(txt):
            txt = pat.sub(lambda m: m.group(1) + "\n" + fn() + "\n" + m.group(3), txt)
    if "--final" in sys.argv:  # team details / URLs are substituted once, when they are known
        for k, v in SUBS.items():
            txt = txt.replace(k, v)
    open(f, "w").write(txt)
    print("filled", os.path.relpath(f, ROOT))
