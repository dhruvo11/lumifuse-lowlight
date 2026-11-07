"""Generate the LaTeX tables of the IEEE paper (paper/tables/*.tex) from the results files,
so every number in the paper is identical to the evaluation outputs."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "paper", "tables")
os.makedirs(T, exist_ok=True)
SM = json.load(open(os.path.join(ROOT, "results", "summary.json")))
LAT = json.load(open(os.path.join(ROOT, "results", "latency.json")))["results"]
FA = json.load(open(os.path.join(ROOT, "results", "failure_analysis.json")))
AR = json.load(open(os.path.join(ROOT, "results", "artifact_metrics.json")))

M = ["Classical", "Zero-DCE (official, no fine-tune)", "DL-only (LumiNet w/o priors)", "Hybrid (LumiFuse, proposed)"]
NAME = {M[0]: "Classical (adaptive)", M[1]: "Zero-DCE (official)", M[2]: "DL-only (LumiNet, no priors)",
        M[3]: "\\textbf{LumiFuse (hybrid)}"}
HIGHER = {"psnr", "ssim", "edge", "edge_corr"}


def fmt(v, d):
    return f"{v:.{d}f}"


def cells(rows, keys):
    """rows: {name: {key: val}}; keys: [(key, decimals)] -> {name: [cell strings]} with best bolded."""
    out = {n: [] for n in rows}
    for k, d in keys:
        vals = {n: r[k] for n, r in rows.items() if r.get(k) is not None}
        best = (max if k in HIGHER else min)(vals, key=vals.get) if vals else None
        for n, r in rows.items():
            v = r.get(k)
            s = "--" if v is None else fmt(v, d)
            out[n].append(f"\\textbf{{{s}}}" if n == best else s)
    return out


def write(name, body):
    open(os.path.join(T, name), "w").write(body)
    print("wrote", name)


# Table I: LOL-v1 and LOL-v2 main comparison + CPU latency
k4 = [("psnr", 2), ("ssim", 4), ("lpips", 3), ("niqe", 2)]
v1 = cells({m: SM["main/LOL-v1 eval15"][m] for m in M}, k4)
v2 = cells({m: SM["main/LOL-v2-real test (unseen)"][m] for m in M}, k4)
lat = cells({m: {"ms": LAT[m]["mean_ms"]} for m in M}, [("ms", 0)])
rows = "\n".join(f"{NAME[m]} & " + " & ".join(v1[m]) + " & " + " & ".join(v2[m]) + f" & {lat[m][0].replace(chr(92) + 'textbf{', '').replace('}', '')} \\\\" for m in M)
write("tab_main.tex", r"""\begin{table*}[t]
\centering
\caption{Comparison of classical, deep-learning and hybrid approaches on the paired LOL benchmarks (best in bold). LOL-v2-real test images never appear in training. Latency: Apple M1 CPU, 4 threads, $600\times400$ input.}
\label{tab:main}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l cccc cccc c}
\toprule
 & \multicolumn{4}{c}{LOL-v1 eval15 (15 images)} & \multicolumn{4}{c}{LOL-v2-real test (100 images)} & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & NIQE$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & NIQE$\downarrow$ & ms/frame \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table II: generalisation across illumination conditions
k3 = [("psnr", 2), ("ssim", 3), ("lpips", 3)]
c = {s: cells({m: SM[f"main/{s}"][m] for m in M}, k3) for s in ("Synthetic moderate", "Synthetic extreme-dark", "Synthetic non-uniform")}
real = cells({m: SM["main/Real unpaired DICM+LIME"][m] for m in M}, [("niqe", 2)])
rows = "\n".join(f"{NAME[m]} & " + " & ".join(c["Synthetic moderate"][m] + c["Synthetic extreme-dark"][m]
                                             + c["Synthetic non-uniform"][m]) + f" & {real[m][0]} \\\\" for m in M)
write("tab_conditions.tex", r"""\begin{table*}[t]
\centering
\caption{Generalisation across illumination conditions on unseen scenes: synthetic moderate, extremely dark and non-uniform (gradient + spotlight) lighting built from 30 LOL-v2 test references with a Poisson--Gaussian camera model, and 74 real DICM/LIME photographs (no reference, NIQE).}
\label{tab:conditions}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l ccc ccc ccc c}
\toprule
 & \multicolumn{3}{c}{Moderate ($\times$0.08--0.15)} & \multicolumn{3}{c}{Extreme-dark ($\times$0.003--0.006)} & \multicolumn{3}{c}{Non-uniform} & Real \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-11}
Method & PSNR & SSIM & LPIPS & PSNR & SSIM & LPIPS & PSNR & SSIM & LPIPS & NIQE \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table III: noise, edges, over-exposure and colour shift (LOL-v2 and extreme-dark)
def art_rows(s):
    r = {m: dict(AR[s][m], noise=SM[f"main/{s}"][m]["noise"]) for m in M}
    return cells(r, [("overexposed_pct", 3), ("deltaE2000", 1), ("colour_angle_deg", 1), ("edge_corr", 3), ("noise", 4)])
a2, ae = art_rows("LOL-v2-real test (unseen)"), art_rows("Synthetic extreme-dark")
short = {M[0]: "Classical", M[1]: "Zero-DCE", M[2]: "DL-only", M[3]: "\\textbf{LumiFuse}"}
rows = "\\multicolumn{6}{l}{\\textit{LOL-v2-real test}}\\\\\n" + "\n".join(f"{short[m]} & " + " & ".join(a2[m]) + " \\\\" for m in M) \
    + "\n\\midrule\n\\multicolumn{6}{l}{\\textit{Synthetic extreme-dark}}\\\\\n" + "\n".join(f"{short[m]} & " + " & ".join(ae[m]) + " \\\\" for m in M)
write("tab_artifacts.tex", r"""\begin{table}[t]
\centering
\caption{Artefacts: newly clipped pixels (\%), colour shift (CIEDE2000 $\Delta E$, RGB angle in degrees), edge preservation (Sobel-magnitude correlation) and residual noise $\sigma$.}
\label{tab:artifacts}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{l ccccc}
\toprule
Method & Clip\%$\downarrow$ & $\Delta E\downarrow$ & Angle$\downarrow$ & Edge$\uparrow$ & Noise $\sigma\downarrow$ \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Table IV: hybrid ablation
AB = ["Full hybrid", "- classical priors (net w/o L, G)", "- RefineNet (curve stage only)",
      "- LOL fine-tuning (official Zero-DCE)", "- input conditioning", "- reliability fusion (all)",
      "- global classical trust only", "- quality gates", "- highlight protection"]
ABN = ["Full LumiFuse", "$-$ classical priors ($L$, $G$)", "$-$ RefineNet", "$-$ LOL fine-tuning",
       "$-$ input conditioning", "$-$ reliability fusion", "$-$ global trust $\\beta$", "$-$ quality gates",
       "$-$ highlight protection"]
sets = ["LOL-v1 eval15", "LOL-v2-real test (unseen)", "Synthetic extreme-dark", "Synthetic non-uniform"]
cols = {s: cells({a: SM[f"abl/{s}"][a] for a in AB}, [("psnr", 2), ("ssim", 3)]) for s in sets}
rows = "\n".join(f"{n} & " + " & ".join(sum((cols[s][a] for s in sets), [])) + " \\\\" for a, n in zip(AB, ABN))
write("tab_ablation.tex", r"""\begin{table*}[t]
\centering
\caption{Ablation of the hybrid system: each row removes one component (PSNR / SSIM; best in bold).}
\label{tab:ablation}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l cc cc cc cc}
\toprule
 & \multicolumn{2}{c}{LOL-v1} & \multicolumn{2}{c}{LOL-v2-real} & \multicolumn{2}{c}{Extreme-dark} & \multicolumn{2}{c}{Non-uniform} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Variant & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table V: classical ablation
CA = ["Classical (adaptive, full)", "Classical fixed params", "- Retinex", "- denoising", "bilateral instead of guided",
      "- CLAHE", "- colour correction"]
CAN = ["Adaptive, full", "Fixed parameters", "$-$ Retinex", "$-$ denoising", "Bilateral (vs guided)",
       "$-$ CLAHE", "$-$ colour correction"]
csets = ["LOL-v1 eval15", "LOL-v2-real test (unseen)", "Synthetic extreme-dark"]
cc = {s: cells({a: SM[f"cabl/{s}"][a] for a in CA}, [("psnr", 2), ("ssim", 3)]) for s in csets}
rows = "\n".join(f"{n} & " + " & ".join(sum((cc[s][a] for s in csets), [])) + " \\\\" for a, n in zip(CA, CAN))
write("tab_classical.tex", r"""\begin{table}[t]
\centering
\caption{Ablation of the classical pipeline (PSNR / SSIM).}
\label{tab:classical}
\setlength{\tabcolsep}{2.5pt}
\footnotesize
\begin{tabular}{l cc cc cc}
\toprule
 & \multicolumn{2}{c}{LOL-v1} & \multicolumn{2}{c}{LOL-v2} & \multicolumn{2}{c}{Extreme} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Variant & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Table VI: exposure diagnostic
fs = [("LOL-v1 eval15", "LOL-v1"), ("LOL-v2-real test (unseen)", "LOL-v2"), ("Synthetic moderate", "Moderate"),
      ("Synthetic extreme-dark", "Extreme"), ("Synthetic non-uniform", "Non-unif.")]
rows = []
for s, lab in fs:
    c_, h_ = FA[s][M[0]], FA[s][M[3]]
    gm = max(FA[s][m]["psnr_gtmean_diagnostic"] for m in M)
    b = lambda v: f"\\textbf{{{v:.2f}}}" if abs(v - gm) < 1e-9 else f"{v:.2f}"
    rows.append(f"{lab} & {c_['psnr']:.2f} & {b(c_['psnr_gtmean_diagnostic'])} & {h_['psnr']:.2f} & "
                f"{b(h_['psnr_gtmean_diagnostic'])} & {h_['mean_abs_exposure_error']:.3f} \\\\")
write("tab_exposure.tex", r"""\begin{table}[t]
\centering
\caption{Exposure diagnostic: PSNR before and after rescaling each output's mean brightness to the reference (GT-mean; diagnostic only, never reported as a score). Bold: best GT-mean PSNR over all four methods. Last column: hybrid mean absolute brightness error.}
\label{tab:exposure}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{l cc cc c}
\toprule
 & \multicolumn{2}{c}{Classical} & \multicolumn{2}{c}{LumiFuse} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Set & PSNR & GT-mean & PSNR & GT-mean & $|\Delta Y|$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
