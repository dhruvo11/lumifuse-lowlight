# Hackathon Technical Solution Report

| Field | Details |
| :--- | :--- |
| **Problem ID** | `P1` |
| **Problem Title** | `Robust Low-Light Image Restoration and Enhancement` |
| **Team Name** | `Bugweiser` (Team ID `T81`) |
| **Team Members** | Dhruv S P — 1RUA24CSE0133<br>Harshit Rudresh — 1RUA24CSE0157 |
| **GitHub Repository** | https://github.com/dhruvo11/lumifuse-lowlight — evaluated commit `9761bd9` |
| **IEEE paper** | `paper/main.pdf` (LaTeX/Overleaf source in `paper/`) |
| **Live Demo / Deployment** | https://lumifuse.streamlit.app (Streamlit Community Cloud, CPU) |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:** Low-light photos need 5–20× gain in the shadows, but the same gain multiplies sensor
  noise, exaggerates colour casts, and clips the parts of the frame that were already lit (lamps, windows,
  backlight). Fixed curves cannot serve extremely dark, moderately dark and non-uniformly lit images at once, and
  "how bright should the result be" is itself ambiguous. Constraints: run in a few seconds on a free 2-vCPU host,
  generalise to unseen cameras/scenes, and expose the reasoning behind each decision.
* **Proposed Methodology:** *LumiFuse* measures each image first (brightness, noise σ, non-uniformity, colour
  cast) and uses those measurements to set every parameter. It restores the image twice — with an adaptive
  classical pipeline (Retinex + guided-filter denoising + CLAHE + colour correction) and with a Zero-DCE-based
  network that is *conditioned on the classical illumination and amplification maps* — then fuses the two per pixel
  with a reliability map and applies no-reference quality gates. We chose this over a static baseline because the
  measured statistics that describe each degradation directly drive its correction; the ablations quantify each
  piece.

---

## 2. Technical Architecture & Implementation

Architecture diagram: [`docs/architecture.svg`](docs/architecture.svg) · full design rationale:
[`docs/design_doc.md`](docs/design_doc.md).

### 2.1 Classical Algorithmic Baseline (`lumifuse/classical.py`)
* **Algorithms Utilized:** Retinex decomposition I = R ⊙ L with L = max-RGB refined by a **guided filter**
  (edge-aware, avoids halos); illumination re-lighting L′ = L^g; **gain-aware edge-preserving denoising** (guided
  filter on luma, joint guided filter on chroma, bilateral selectable); **CLAHE** on LAB lightness; gray-world white
  balance + saturation compensation; percentile contrast stretch + soft-knee highlight roll-off.
* **Adaptive Selection Logic:** g = ln 0.45 / ln(geomean L) (clipped to [0.15, 1]); guided-filter radius
  r = min(H,W)/50 and ε = (0.1·P90)²; denoising ε = (5σ)² with σ measured by the Immerkær operator on flat pixels,
  weighted per pixel by the amplification L′/L (floor 0.85); CLAHE clip = 1 + 2.5·(0.22 − RMS contrast)/0.22 ∈ [1,3]
  and lighter blending when noisy; white-balance strength ∝ measured cast; saturation gain = 0.30/mean saturation;
  contrast stretch skipped when σ > 0.02. Constants (5, 0.85, 0.02) were tuned on the validation split only.

### 2.2 Deep Learning Pipeline (`lumifuse/models.py`, `scripts/train.py`)
* **Model Architecture:** **LumiNet (1.0 M params)** = **IG-DCE-Net** (Zero-DCE curve estimator, 79 k params,
  8 iterations of LE(x) = x + αx(1−x), curves estimated at ½ resolution as in Zero-DCE++) with a 4th input channel
  for the classical illumination prior, **initialised from the authors' official weights** (extra channel zero-
  initialised), followed by **RefineNet**, a 3-level U-Net (0.93 M) that takes [curve output, L, log-gain G] and
  predicts a residual for denoising and colour correction. Rationale: tiny and monotonic (cannot invert tones or
  invent structure), fast on CPU, pretrained weights available for fine-tuning within the hackathon.
* **Loss Functions & Training:** RefineNet: L1 + 0.2·(1−SSIM) + 0.1·Sobel-gradient L1 (edges) + 0.1·colour-angle
  (colour shifts). Curve stage: low-pass L1 + 0.5·colour-angle + 10·illumination-smoothness TV (Zero-DCE's spatial
  smoothness loss, prevents halos); the curve output is detached before the refiner so it stays a genuine
  enhancement. Data: LOL-v1 + LOL-v2-real, **de-duplicated and leak-free** — we found 89 of the 100 LOL-v2 test
  images inside LOL-v1's training set and removed every test-overlapping pair (98) and duplicate (399): 644 train /
  33 validation pairs. Illumination augmentation (darker ×0.3–1, brighter ×1–2, random smooth non-uniform fields),
  160² crops, AdamW + one-cycle, 4000 iterations, Apple M1 GPU (~35 min per model). The DL-only baseline is the same
  network and recipe without the classical priors.

### 2.3 Hybrid Integration & Adaptive Mechanism (`lumifuse/pipeline.py`)
* **Fusion Strategy:** (1) the guided-filter Retinex illumination map L is an input channel of the curve network;
  (2) L and the amplification map G = Y(enhanced)/Y(input) — a classical noise-amplification model — condition the
  refiner, so it denoises in proportion to how much each pixel was amplified; (3) a per-pixel **reliability map**
  W = blur(under ∨ over) hands regions the network left too dark (Y < 0.12) or blew out (Y > 0.9) to the classical
  branch, plus a global trust term β that grows with input brightness (the network is strongest on very dark,
  noisy inputs; on moderately dark photos the classical branch keeps more contrast); W = max(W_px, β),
  out = (1−W)·DL + W·CL.
* **Decision Logic:** inputs outside the network's training exposure band are rescaled into it (×up to 4 for very
  dark, ×down for moderately dark; band tuned on validation); bright regions get g ≈ 1 in the classical branch, the
  "over" reliability term and a soft knee, so highlights are not pushed into clipping; no-reference quality gates
  then correct exposure outside [0.30, 0.62], residual noise σ > 0.012 (guided filter) and colour cast > 0.10
  (partial gray-world); well-lit inputs are only lightly adjusted.

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

**LOL-v1 eval15** (standard benchmark; latency = Apple M1 CPU, 4 threads, 600×400):

<!-- AUTO:report_lolv1 -->
| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual (LPIPS ↓ / NIQE ↓) | Latency (ms / frame, CPU) |
| :--- | :---: | :---: | :---: | :---: |
| Classical Baseline | 17.06 | 0.6612 | 0.3864 / 6.419 | 64 ms |
| Deep Learning Only (LumiNet, no classical priors) | 17.23 | 0.7344 | 0.2702 / 3.859 | 308 ms |
| Deep Learning — official Zero-DCE (reference) | 14.86 | 0.5627 | 0.3349 / 7.650 | 276 ms |
| **Proposed Hybrid System (LumiFuse)** | 18.09 | 0.7375 | 0.3181 / 4.869 | 529 ms |
<!-- /AUTO:report_lolv1 -->

**LOL-v2-real test — 100 unseen images:**

<!-- AUTO:report_lolv2 -->
| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual (LPIPS ↓ / NIQE ↓) | Latency (ms / frame, CPU) |
| :--- | :---: | :---: | :---: | :---: |
| Classical Baseline | 19.12 | 0.6890 | 0.3995 / 6.581 | 64 ms |
| Deep Learning Only (LumiNet, no classical priors) | 16.14 | 0.7382 | 0.2959 / 3.963 | 308 ms |
| Deep Learning — official Zero-DCE (reference) | 18.06 | 0.5798 | 0.3123 / 7.897 | 276 ms |
| **Proposed Hybrid System (LumiFuse)** | 17.41 | 0.7378 | 0.3542 / 4.716 | 529 ms |
<!-- /AUTO:report_lolv2 -->

**Synthetic extreme-dark (×0.003–0.006 exposure, Poisson–Gaussian noise, unseen scenes):**

<!-- AUTO:report_ext -->
| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual (LPIPS ↓ / NIQE ↓) | Latency (ms / frame, CPU) |
| :--- | :---: | :---: | :---: | :---: |
| Classical Baseline | 14.99 | 0.3168 | 0.9080 / 13.972 | 64 ms |
| Deep Learning Only (LumiNet, no classical priors) | 15.78 | 0.4899 | 0.7457 / 7.099 | 308 ms |
| Deep Learning — official Zero-DCE (reference) | 10.34 | 0.0826 | 1.2997 / 15.890 | 276 ms |
| **Proposed Hybrid System (LumiFuse)** | 16.01 | 0.5007 | 0.7633 / 5.891 | 529 ms |
<!-- /AUTO:report_ext -->

* **Analysis:** The hybrid gives the best structural fidelity (SSIM best or tied on every paired set; PSNR best on LOL-v1 and extreme-dark, +3.2 dB over official Zero-DCE on LOL-v1). The deep-only network has the best perceptual scores (LPIPS/NIQE) because the classical priors make the hybrid refiner denoise harder — a perception–distortion trade-off. Raw PSNR on LOL-v2 favours the classical pipeline because of global exposure, not restoration quality (exposure-matched diagnostic: hybrid 22.30 dB vs classical 21.27 dB). Speed: 64 ms (classical), ~0.3 s (curve networks), 0.53 s (hybrid) per 600×400 frame on a laptop CPU.

All tables, including moderate / non-uniform lighting and 74 real DICM+LIME photos (NIQE), noise σ and edge
preservation: [`eval_P1_T81.md`](eval_P1_T81.md).

### 3.2 Visual Outputs & Intermediate Representations
1. **Extreme low-light:** [`docs/figures/conditions_synthetic.png`](docs/figures/conditions_synthetic.png) (rows 1–2),
   [`docs/figures/conditions_real.png`](docs/figures/conditions_real.png) (DICM 27, mean brightness ≈0.02).
2. **Non-uniform / backlit scenes:** `conditions_synthetic.png` rows 3–4 (spotlight + gradient) and
   `conditions_real.png` (DICM 07 backlit, LIME night street) — bright regions keep their detail.
3. **Intermediate states:** [`docs/figures/intermediates_lol.png`](docs/figures/intermediates_lol.png) and
   [`intermediates_backlit.png`](docs/figures/intermediates_backlit.png): illumination map L, reflectance R, curve
   strength |A|, curve-stage output, amplification map G, refiner correction, classical output, fusion weights;
   [`intermediates_classical_*.png`](docs/figures/) for the classical stages. All are also shown live in the app
   ("Pipeline stages" tab).

### 3.3 Ablation Study
* Classical priors in the network: **+1.00 dB** (LOL-v1) / **+1.14 dB** (LOL-v2) PSNR; slightly worse on synthetic
  extreme/non-uniform and on LPIPS/NIQE (stronger denoising).
* RefineNet: SSIM +0.023 to +0.049, residual noise ÷3, NIQE −0.8 to −3.0.
* LOL fine-tuning vs official weights: **+3.03 dB** LOL-v1, +2.0 dB extreme-dark, −2.6 dB LOL-v2 (exposure).
* Adaptive input conditioning: **+1.75 dB** on non-uniform light; global classical trust: **+1.09 dB**, SSIM +0.054.
* Quality gates: +0.30 dB extreme-dark, +0.14 dB LOL-v2.
* Classical pipeline: Retinex is essential (−7.8 dB without); denoising −1.2 to −3.3 dB without; guided beats
  bilateral on every set; **adaptive beats fixed parameters** by +1.5 to +2.2 dB on LOL and extreme-dark (fixed wins
  only on non-uniform PSNR, §4).

* Artefacts (advanced req. 4, `scripts/artifact_metrics.py`): the hybrid introduces **0 % newly clipped pixels** on
  every test set (classical up to 0.11 %) and has the lowest hue error on moderate (3.19°) and non-uniform light
  (2.96°); official Zero-DCE's colour casts reach 29° on extreme-dark inputs.

Full tables (4 test sets × 9 hybrid variants, 4 × 7 classical variants), artefact metrics and rejected design alternatives:
[`docs/ablation_and_failures.md`](docs/ablation_and_failures.md).

---

## 4. Failure Modes & Limitations

* **Observed Failure Case:** *Global exposure ambiguity.* On a few LOL-v2 scene series (e.g. `00763`, `00764`,
  `00724–00730`) the hybrid output is markedly brighter than the reference (mean brightness ~0.5 vs ~0.33), costing
  up to ~10 dB PSNR on those images, so on LOL-v2 its raw PSNR falls below the classical pipeline even though its
  SSIM, LPIPS and NIQE are better. Secondary: low-frequency chroma blotches in extremely dark regions.
* **Root Cause & Mitigation:** Target brightness is ill-posed from a dark input: LOL references use photographer-chosen exposures and the network learns their population average (≈0.45 mean brightness); PSNR punishes a global offset heavily (a uniform 0.1 error alone caps PSNR at 20 dB). Exposure-matching the outputs to the references (diagnostic only) lifts the hybrid to the best score on LOL-v2 (22.30 vs classical 21.27 dB), so the error is exposure, not restoration. We tried a classical exposure prior; it lowered validation PSNR (18.54 → 17.39), so it is off. Mitigations: a user/target-exposure control (Zero-DCE's exposure
  loss already takes a target E), exposure-conditioned training, RAW or burst input for extreme darkness, and
  training the refiner with synthetic Poisson–Gaussian noise at extreme gains. Detailed analysis with figures:
  [`docs/ablation_and_failures.md`](docs/ablation_and_failures.md).
* **Limitations & ethics:** trained on mostly indoor LOL scenes; the refiner is a learned prior and may smooth or
  invent fine texture, so outputs must not be used as forensic or medical evidence; images are processed in memory
  and never stored; datasets used under research terms. See design doc §4.

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands
```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/dhruvo11/lumifuse-lowlight
cd lumifuse-lowlight
pip install -r requirements.txt

# 2. Run inference on a sample directory
python run_inference.py --input_dir ./samples/ --output_dir ./results/inference/ --save_maps

# 3. Web app (deployed version) — or: docker build -t lumifuse . && docker run -p 7860:7860 lumifuse
pip install streamlit && streamlit run webapp/streamlit_app.py
```

### Deployment Format
* **Streamlit web app** (`webapp/streamlit_app.py`) hosted on **Streamlit Community Cloud** (free CPU), PyTorch
  CPU inference; Apple-style responsive UI with a before/after slider, live intermediate maps, a "compare all four
  methods" mode, loading spinner, PNG download and friendly error states; inputs capped at 768 px for few-second
  latency. The same UI in **Gradio** (`app.py`) runs locally or via the **Dockerfile** (one command). **CLI**:
  `run_inference.py`. No API keys or secrets. (Hugging Face Spaces now requires a paid plan for Gradio apps;
  `scripts/deploy_hf.py` deploys there if one is available.)

---

### Student Submission Checklist
- [x] Working GitHub repository link included with an updated `README.md`.
- [x] No raw code pasted into this report.
- [x] All table metrics match test runs reproducible via the repository (`scripts/evaluate.py` → `summarize.py` → `fill_docs.py`).
- [x] All team member details accurately recorded.
