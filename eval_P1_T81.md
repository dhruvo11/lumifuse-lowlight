1RUA24CSE0133,1RUA24CSE0157

# Evaluation results — P1 Robust Low-Light Image Restoration — Team T81 (Bugweiser)

System: **LumiFuse** (classical Retinex/CLAHE/guided-filter pipeline + illumination-guided Zero-DCE network + reliability-weighted fusion). All numbers are produced by `python scripts/evaluate.py && python scripts/summarize.py`.

## Protocol
* **Training data (leak-free):** LOL-v1 `our485` ∪ LOL-v2-real `Train`, with every pair whose low-light input also appears in a test set removed (98 pairs) and duplicates removed (399). 644 train / 33 validation pairs. Validation (not test) is used for checkpoint and hyper-parameter selection.
* **Paired test sets:** LOL-v1 `eval15` (15), LOL-v2-real `Test` (100, none seen in training — 89 of these are LOL-v1 training images, which we removed from training), and three synthetic lighting conditions generated from 30 unseen LOL-v2 ground truths with a Poisson-Gaussian camera noise model: moderate (×0.08–0.15 exposure), extreme-dark (×0.003–0.006) and non-uniform (gradient + spotlight, ×0.004–0.6 across the frame).
* **Unpaired real test set:** DICM (64) + LIME (10) photographs, resized to ≤1024 px; no-reference metrics only.
* **Metrics (problem-statement order):** PSNR (dB, RGB, data range 1) and SSIM (RGB, skimage, Gaussian-free 7×7 default); LPIPS (AlexNet, v0.1); NIQE (MATLAB-equivalent BasicSR port, Y channel); processing latency (Apple M1 CPU, 4 threads, 600x400); noise suppression = residual noise σ of the output (Immerkær estimator on flat pixels, lower = cleaner); edge preservation = Pearson correlation between Sobel gradient magnitudes of output and ground truth (higher = edges kept). No GT-mean brightness correction is applied anywhere.

## 1. Comparative evaluation (classical vs deep learning vs hybrid)

### 1.1 LOL-v1 eval15 (standard benchmark)
Latency = CPU benchmark (Apple M1, 4 threads, 600×400).
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 17.06 | 0.6612 | 0.3864 | 6.419 | 64 | 0.0129 | 0.8106 |
| Zero-DCE (official, no fine-tune) | 14.86 | 0.5627 | 0.3349 | 7.650 | 276 | 0.0170 | **0.8894** |
| DL-only (LumiNet w/o priors) | 17.23 | 0.7344 | **0.2702** | **3.859** | 308 | 0.0023 | 0.8637 |
| Hybrid (LumiFuse, proposed) | **18.09** | **0.7375** | 0.3181 | 4.869 | 529 | **0.0018** | 0.8252 |

### 1.2 LOL-v2-real test — 100 unseen images
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | **19.12** | 0.6890 | 0.3995 | 6.581 | 64 | 0.0151 | 0.7770 |
| Zero-DCE (official, no fine-tune) | 18.06 | 0.5798 | 0.3123 | 7.897 | 276 | 0.0215 | **0.8819** |
| DL-only (LumiNet w/o priors) | 16.14 | **0.7382** | **0.2959** | **3.963** | 308 | 0.0032 | 0.8393 |
| Hybrid (LumiFuse, proposed) | 17.41 | 0.7378 | 0.3542 | 4.716 | 529 | **0.0024** | 0.7922 |

## 2. Generalisation across illumination conditions (unseen scenes)

### 2.1 Moderate low light
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 17.87 | 0.7616 | 0.2542 | 5.776 | 64 | 0.0104 | **0.8733** |
| Zero-DCE (official, no fine-tune) | **18.50** | 0.7382 | **0.2453** | 6.445 | 276 | 0.0113 | 0.8263 |
| DL-only (LumiNet w/o priors) | 13.53 | 0.7371 | 0.3141 | **4.454** | 308 | **0.0016** | 0.8317 |
| Hybrid (LumiFuse, proposed) | 17.10 | **0.7992** | 0.2619 | 4.685 | 529 | 0.0021 | 0.8581 |

### 2.2 Extremely dark
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 14.99 | 0.3168 | 0.9080 | 13.972 | 64 | 0.0402 | 0.1923 |
| Zero-DCE (official, no fine-tune) | 10.34 | 0.0826 | 1.2997 | 15.890 | 276 | 0.0910 | 0.1406 |
| DL-only (LumiNet w/o priors) | 15.78 | 0.4899 | **0.7457** | 7.099 | 308 | 0.0088 | 0.3718 |
| Hybrid (LumiFuse, proposed) | **16.01** | **0.5007** | 0.7633 | **5.891** | 529 | **0.0074** | **0.3848** |

### 2.3 Non-uniform / spotlight illumination
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | **16.88** | 0.7475 | 0.2660 | 5.139 | 64 | 0.0092 | 0.8565 |
| Zero-DCE (official, no fine-tune) | 16.82 | 0.7204 | 0.2737 | 5.630 | 276 | 0.0113 | 0.7980 |
| DL-only (LumiNet w/o priors) | 13.94 | 0.7379 | 0.3218 | **4.412** | 308 | **0.0018** | 0.8304 |
| Hybrid (LumiFuse, proposed) | 16.73 | **0.7963** | **0.2524** | 4.659 | 529 | 0.0033 | **0.8680** |

### 2.4 Real unpaired photos (DICM + LIME) — no-reference
| Method | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ |
|---|---:|---:|---:|
| Classical | 3.856 | 64 | 0.0083 |
| Zero-DCE (official, no fine-tune) | **3.804** | 276 | 0.0073 |
| DL-only (LumiNet w/o priors) | 3.944 | 308 | **0.0026** |
| Hybrid (LumiFuse, proposed) | 4.128 | 529 | 0.0028 |

## 3. Processing latency (CPU, deployment hardware class)

| Method | Mean (ms) | Std (ms) |
|---|---:|---:|
| Classical | 64 | 2 |
| Zero-DCE (official, no fine-tune) | 276 | 8 |
| DL-only (LumiNet w/o priors) | 308 | 10 |
| Hybrid (LumiFuse, proposed) | 529 | 9 |

## 4. Ablation study

### 4.x Hybrid — remove one component — LOL-v1 eval15
Latency column here is evaluation wall-clock (GPU/MPS), not the CPU benchmark.
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full hybrid | 18.09 | 0.7375 | 0.3181 | 4.869 | 294 | 0.0018 | 0.8252 |
| - classical priors (net w/o L, G) | 17.09 | 0.7353 | **0.2774** | **3.911** | 267 | 0.0024 | 0.8635 |
| - RefineNet (curve stage only) | 18.08 | 0.7144 | 0.3159 | 5.653 | 163 | 0.0056 | 0.8687 |
| - LOL fine-tuning (official Zero-DCE) | 15.06 | 0.7002 | 0.2914 | 5.492 | 230 | 0.0041 | **0.9003** |
| - reliability fusion (all) | 18.09 | 0.7373 | 0.3181 | 4.875 | 187 | **0.0018** | 0.8249 |
| - global classical trust only | 18.09 | 0.7375 | 0.3181 | 4.869 | 188 | 0.0018 | 0.8252 |
| - quality gates | **18.13** | **0.7381** | 0.3133 | 4.881 | 288 | 0.0018 | 0.8252 |
| - input conditioning | 18.10 | 0.7357 | 0.3164 | 4.857 | 288 | 0.0018 | 0.8250 |
| - highlight protection | 18.09 | 0.7375 | 0.3181 | 4.869 | 270 | 0.0018 | 0.8252 |

### 4.x Hybrid — remove one component — LOL-v2-real test (unseen)
Latency column here is evaluation wall-clock (GPU/MPS), not the CPU benchmark.
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full hybrid | 17.41 | 0.7378 | 0.3542 | 4.716 | 273 | 0.0024 | 0.7922 |
| - classical priors (net w/o L, G) | 16.27 | 0.7370 | 0.2966 | **3.975** | 258 | 0.0034 | 0.8390 |
| - RefineNet (curve stage only) | 16.85 | 0.7251 | 0.3315 | 5.571 | 170 | 0.0078 | 0.8379 |
| - LOL fine-tuning (official Zero-DCE) | **20.02** | **0.7765** | **0.2626** | 5.449 | 228 | 0.0055 | **0.9005** |
| - reliability fusion (all) | 17.40 | 0.7369 | 0.3561 | 4.746 | 179 | 0.0024 | 0.7915 |
| - global classical trust only | 17.40 | 0.7370 | 0.3560 | 4.742 | 192 | 0.0024 | 0.7915 |
| - quality gates | 17.27 | 0.7380 | 0.3556 | 4.718 | 271 | 0.0024 | 0.7921 |
| - input conditioning | 17.24 | 0.7346 | 0.3550 | 4.727 | 282 | **0.0024** | 0.7914 |
| - highlight protection | 17.41 | 0.7378 | 0.3543 | 4.716 | 278 | 0.0024 | 0.7923 |

### 4.x Hybrid — remove one component — Synthetic extreme-dark
Latency column here is evaluation wall-clock (GPU/MPS), not the CPU benchmark.
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full hybrid | 16.01 | 0.5007 | 0.7633 | 5.891 | 287 | 0.0074 | 0.3848 |
| - classical priors (net w/o L, G) | 16.80 | 0.5296 | **0.7489** | 7.032 | 285 | **0.0071** | 0.3743 |
| - RefineNet (curve stage only) | **17.33** | 0.5130 | 0.7572 | 8.895 | 174 | 0.0147 | 0.3709 |
| - LOL fine-tuning (official Zero-DCE) | 14.01 | **0.5304** | 0.7515 | 8.710 | 207 | 0.0096 | 0.3778 |
| - reliability fusion (all) | 16.01 | 0.5007 | 0.7633 | 5.891 | 198 | 0.0074 | 0.3847 |
| - global classical trust only | 16.01 | 0.5007 | 0.7633 | 5.891 | 202 | 0.0074 | 0.3848 |
| - quality gates | 15.71 | 0.5009 | 0.7627 | 5.883 | 283 | 0.0073 | **0.3853** |
| - input conditioning | 15.95 | 0.5011 | 0.7644 | **5.857** | 282 | 0.0072 | 0.3834 |
| - highlight protection | 16.01 | 0.5007 | 0.7633 | 5.891 | 293 | 0.0074 | 0.3848 |

### 4.x Hybrid — remove one component — Synthetic non-uniform
Latency column here is evaluation wall-clock (GPU/MPS), not the CPU benchmark.
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full hybrid | 16.73 | 0.7963 | 0.2524 | 4.659 | 270 | 0.0033 | 0.8680 |
| - classical priors (net w/o L, G) | **16.96** | **0.8028** | 0.2277 | **4.297** | 276 | 0.0035 | **0.8794** |
| - RefineNet (curve stage only) | 16.25 | 0.7477 | 0.2775 | 5.502 | 173 | 0.0068 | 0.7815 |
| - LOL fine-tuning (official Zero-DCE) | 16.91 | 0.7914 | **0.2240** | 5.335 | 189 | 0.0067 | 0.8779 |
| - reliability fusion (all) | 15.64 | 0.7423 | 0.3794 | 5.373 | 185 | 0.0011 | 0.7882 |
| - global classical trust only | 15.64 | 0.7423 | 0.3794 | 5.373 | 190 | **0.0011** | 0.7882 |
| - quality gates | 16.73 | 0.7963 | 0.2521 | 4.659 | 271 | 0.0033 | 0.8680 |
| - input conditioning | 14.98 | 0.7622 | 0.2907 | 4.693 | 229 | 0.0034 | 0.8455 |
| - highlight protection | 16.73 | 0.7963 | 0.2524 | 4.659 | 178 | 0.0033 | 0.8680 |

### 4.y Classical pipeline — remove one component — LOL-v1 eval15
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical (adaptive, full) | 17.06 | 0.6612 | 0.3864 | 6.419 | 132 | 0.0129 | 0.8106 |
| Classical fixed params | 15.27 | 0.6394 | **0.3281** | **5.376** | 128 | 0.0125 | **0.8709** |
| - Retinex | 9.24 | 0.3452 | 0.4735 | 6.387 | 127 | **0.0027** | 0.8249 |
| - denoising | 15.90 | 0.3741 | 0.5360 | 9.279 | 93 | 0.0515 | 0.7362 |
| bilateral instead of guided | 16.74 | 0.5901 | 0.3862 | 6.378 | 115 | 0.0183 | 0.7951 |
| - CLAHE | **17.17** | **0.7069** | 0.3300 | 5.813 | 123 | 0.0058 | 0.8380 |
| - colour correction | 17.11 | 0.6649 | 0.3928 | 6.405 | 108 | 0.0130 | 0.8108 |

### 4.y Classical pipeline — remove one component — LOL-v2-real test (unseen)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical (adaptive, full) | 19.12 | 0.6890 | 0.3995 | 6.581 | 134 | 0.0151 | 0.7770 |
| Classical fixed params | 17.64 | 0.6389 | **0.3085** | **5.997** | 99 | 0.0211 | **0.8683** |
| - Retinex | 11.40 | 0.3547 | 0.4439 | 6.488 | 109 | **0.0033** | 0.8134 |
| - denoising | 16.67 | 0.4015 | 0.5438 | 9.448 | 98 | 0.0596 | 0.6955 |
| bilateral instead of guided | 18.38 | 0.6066 | 0.3791 | 6.902 | 117 | 0.0219 | 0.7672 |
| - CLAHE | **20.24** | **0.7360** | 0.3694 | 6.126 | 104 | 0.0070 | 0.8026 |
| - colour correction | 18.47 | 0.6887 | 0.4240 | 6.575 | 105 | 0.0151 | 0.7783 |

### 4.y Classical pipeline — remove one component — Synthetic extreme-dark
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical (adaptive, full) | 14.99 | 0.3168 | 0.9080 | 13.972 | 117 | 0.0402 | 0.1923 |
| Classical fixed params | 12.79 | 0.1147 | 1.2824 | 16.448 | 103 | 0.1632 | 0.1521 |
| - Retinex | 7.55 | 0.1501 | 1.0055 | **11.329** | 106 | **0.0094** | 0.2234 |
| - denoising | 11.69 | 0.0738 | 1.3910 | 25.064 | 87 | 0.1772 | 0.0958 |
| bilateral instead of guided | 14.82 | 0.2158 | **0.8115** | 14.205 | 114 | 0.0685 | 0.1939 |
| - CLAHE | 13.25 | **0.4662** | 0.8573 | 13.254 | 94 | 0.0175 | **0.2400** |
| - colour correction | **15.16** | 0.3187 | 0.9149 | 13.929 | 93 | 0.0403 | 0.1927 |

### 4.y Classical pipeline — remove one component — Synthetic non-uniform
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms) ↓ | Noise σ ↓ | Edge corr. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical (adaptive, full) | 16.88 | 0.7475 | 0.2660 | 5.139 | 94 | 0.0092 | 0.8565 |
| Classical fixed params | **17.31** | 0.7340 | 0.2621 | **4.835** | 82 | 0.0113 | 0.8301 |
| - Retinex | 13.92 | 0.7087 | **0.2542** | 5.034 | 98 | 0.0058 | **0.8907** |
| - denoising | 16.17 | 0.5773 | 0.3659 | 7.315 | 98 | 0.0317 | 0.8064 |
| bilateral instead of guided | 16.66 | 0.7078 | 0.2820 | 5.344 | 136 | 0.0119 | 0.8358 |
| - CLAHE | 16.66 | **0.7781** | 0.2556 | 5.291 | 118 | **0.0043** | 0.8787 |
| - colour correction | 16.87 | 0.7570 | 0.2664 | 5.140 | 104 | 0.0091 | 0.8567 |

## 5. Failure-case analysis
See `docs/ablation_and_failures.md` (worst cases, root causes, mitigations) and `docs/figures/failure_cases.png`.
