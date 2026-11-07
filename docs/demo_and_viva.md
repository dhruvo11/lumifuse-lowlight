# 3-minute demo script + viva preparation

> Open the live URL **10 minutes before judging**: free Streamlit Cloud apps sleep after ~12 h without visitors —
> if you see "This app has gone to sleep", click **"Yes, get this app back up"** and wait ~1 minute. Keep the
> GitHub repo and `docs/figures/` open in other tabs. Backup: `streamlit run webapp/streamlit_app.py` locally.

## Demo (3:00)

| Time | Who | Action | What to say |
|---|---|---|---|
| 0:00–0:20 | Member 1 | Show the landing page | "LumiFuse restores dark, noisy, colour-shifted photos. Classical Retinex + a Zero-DCE network, fused adaptively — every parameter is computed from the image." |
| 0:20–0:55 | Member 1 | Pick example **Monument at night — very dark**, method **Hybrid**, Enhance; drag the slider | Read the info panel: *detected condition, noise σ, per-stage latency, NIQE before→after*. "It measured the noise and brightness first; those numbers chose the parameters." |
| 0:55–1:25 | Member 2 | Open **Pipeline stages** | "Illumination map L from the guided-filter Retinex step → fed to the network as a 4th channel. Curve strength = what Zero-DCE did. Gain map = how much each pixel was amplified, which is how much its noise grew — the refiner uses it to denoise exactly where needed. Fusion weight = where the classical branch overrides the network." |
| 1:25–1:55 | Member 2 | Choose **Compare all** on **Synthetic — extremely dark**, then on **Backlit building** | "Classical is fast but grainy; official Zero-DCE is washed out and noisy; DL-only is clean but too bright and flat on moderately dark photos; the hybrid keeps highlights and lifts shadows." |
| 1:55–2:30 | Member 1 | **Judge's own image**: ask a judge for a dark phone photo and upload it | Point to the condition label and latency. If it is well-lit: "it detects that and barely changes it." |
| 2:30–3:00 | Member 2 | Show `eval_P1_<team>.md` + `docs/figures/failure_cases.png` | "Leak-free split — we found 89 of 100 LOL-v2 test images inside LOL-v1 train and removed them. Hybrid: best SSIM on every paired set, +3.2 dB PSNR over official Zero-DCE on LOL, half a second per frame on a laptop CPU. Failure case: on a few LOL-v2 scenes it picks a brighter exposure than the reference — exposure-matched it is the best restorer (22.3 dB)." |

## Who owns what (each member must be able to explain their part)

| Member | Owns | Must be able to explain |
|---|---|---|
| Dhruv S P | Classical pipeline + hybrid fusion + app (`classical.py`, `quality.py`, `pipeline.py`, `webapp/`) | Retinex I = R·L; max-RGB + guided filter for L; how g, ε, CLAHE clip, WB strength are computed; guided vs bilateral; reliability fusion + global trust; quality gates; input conditioning; UI error handling and latency caps |
| Harshit Rudresh | Deep model, training and evaluation (`models.py`, `train.py`, `prepare_data.py`, `evaluate.py`, metrics) | Zero-DCE curve LE(x)=x+αx(1−x), 8 iterations; 4th-channel zero-init; RefineNet and its inputs; losses; the detach fix; leak-free split (89/100 LOL-v2 test images were in LOL-v1 train); PSNR/SSIM/LPIPS/NIQE; reading the ablation and the failure analysis |

## Likely viva questions — short answers

**Why Zero-DCE and not EnlightenGAN / Retinexformer?** 79 k-param curve estimator → fast on a free CPU host;
curves are monotonic so it cannot invert tones or hallucinate structure; official weights exist, so we fine-tune
instead of training from scratch in 5 hours. Its weakness (no denoising, washed colours) is exactly what our
classical priors and the refiner fix — visible in the Zero-DCE vs hybrid rows.

**What exactly is "hybrid" here — isn't it just averaging?** Three couplings: (1) classical illumination L is a network
input; (2) the classical noise-amplification model G = Y(out)/Y(in) is a refiner input; (3) per-pixel reliability
fusion + no-reference gates. Each one has an ablation row showing its effect.

**How are parameters adaptive?** g = ln 0.45 / ln(geomean L); guided-filter ε = (0.1·P90)²; denoise ε = (k·σ)²
with σ measured by the Immerkær estimator, weight ∝ amplification; CLAHE clip from RMS contrast; WB strength
∝ cast; input pre-gain when mean < 0.03; gates on exposure/noise/cast. The fixed-parameter ablation shows the gain.

**How do you avoid over-exposure / halos?** Re-lighting exponent g ≤ 1 leaves L≈1 regions unchanged; edge-aware
guided-filter illumination (halos come from blurring across edges); TV loss on curve maps; soft knee above 0.88;
reliability "over" term hands blown-out regions to the classical branch.

**PSNR vs SSIM vs LPIPS vs NIQE?** PSNR = pixel error (log MSE), sensitive to global brightness; SSIM = local
luminance/contrast/structure agreement, very sensitive to noise; LPIPS = distance in deep (AlexNet) feature space,
closer to human judgement; NIQE = no-reference naturalness (distance of local statistics from a pristine model),
usable on real photos with no ground truth. Lower LPIPS/NIQE is better.

**How do you know results are not overfit / leaked?** Pixel-hash de-duplication removed 98 test-overlapping and 399
duplicate training pairs; validation split for every choice; LOL-v2 100 test images, 3 synthetic conditions from
unseen scenes, and 74 real DICM/LIME photos never seen in training.

**What is your failure case?** Global exposure ambiguity: on LOL-v2 scenes 00724–30, 00750–52, 00763–64 our output is brighter than the reference (≈0.5 vs ≈0.33 mean), costing up to ~10 dB on those images, so raw LOL-v2 PSNR is below classical. Matching exposure to the reference (diagnostic only) makes the hybrid best (22.30 vs 21.27 dB) — the restoration is right, the brightness choice differs. Fix: a target-exposure control (Zero-DCE's exposure loss already takes a target) or exposure-conditioned training. Second failure: chroma blotches in extremely dark regions — information lost to 8-bit quantisation; fix with RAW/burst input.

**Latency?** 0.53 s per 600×400 frame on an M1 laptop CPU (4 threads); classical alone 64 ms; the curve networks ~0.3 s. The app caps inputs at 768 px so the free 2-vCPU Hugging Face host answers in a few seconds; curves are estimated at half resolution (Zero-DCE++ trick) to save 4× compute.

**What would you do with more time?** Train longer / larger refiner; noise-level-conditioned training with synthetic
Poisson-Gaussian noise; learned (instead of rule-based) fusion weights; ONNX/int8 export for faster CPU inference.
