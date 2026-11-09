# Main experiment v2 — results summary (generated from `main_v2.json`)

runs: 340 {'rand': 300, 'strat': 40}. Every number below is a key in `main_v2.json`.

## Variance decomposition per architecture and metric (random design; REML unless noted)

| arch | metric | est | s2_scene | s2_train | s2_inter | s2_seed (scene-level) | test-side share of predicted | ICC(scene, 1 seed) | share vs JOSA A observed (PSNR) | V_pred/obs (PSNR) |
|---|---|---|---|---|---|---|---|---|---|---|
| direct | si-MSE (gray, Grosse 2009) | reml | 0.000118 | 6.51e-07 | 4.53e-06 | 1.97e-06 | 0.92 | 0.95 | — | — |
| direct | LMSE (Chen & Koltun 2013) | reml | 1.59e-05 | 4.9e-08 | 3.73e-07 | 3.49e-07 | 0.95 | 0.96 | — | — |
| direct | DSSIM (Chen & Koltun 2013) | reml | 0.00163 | 1.96e-05 | 5.41e-05 | 5.99e-05 | 0.89 | 0.93 | — | — |
| direct | R_PSNR (PSNR of scene-mean MSE) | reml | 2.83 | 0.41 | 1.12 | 0.308 | 0.41 | 0.67 | 0.48 | 1.18 |
| direct | PSNR after per-scene optimal RGB gain (H4) | reml | 5.01 | 0.0387 | 0.107 | 0.0691 | 0.93 | 0.97 | — | — |
| proposed | si-MSE (gray, Grosse 2009) | reml | 0.000112 | 4.78e-08 | 6.11e-06 | 9.02e-06 | 0.91 | 0.88 | — | — |
| proposed | LMSE (Chen & Koltun 2013) | reml | 1.45e-05 | 2.06e-07 | 8.97e-07 | 9.54e-07 | 0.84 | 0.89 | — | — |
| proposed | DSSIM (Chen & Koltun 2013) | reml | 0.00144 | 4.55e-05 | 9.18e-05 | 5.99e-05 | 0.77 | 0.90 | — | — |
| proposed | R_PSNR (PSNR of scene-mean MSE) | reml | 1.55 | 0.304 | 1.57 | 0.443 | 0.28 | 0.43 | 0.32 | 1.15 |
| proposed | PSNR after per-scene optimal RGB gain (H4) | reml | 4.53 | 0.0206 | 0.164 | 0.229 | 0.92 | 0.92 | — | — |
| transformer | si-MSE (gray, Grosse 2009) | reml | 8.5e-05 | 7.62e-07 | 4.28e-06 | 1.31e-05 | 0.86 | 0.83 | — | — |
| transformer | LMSE (Chen & Koltun 2013) | reml | 1.07e-05 | 1.36e-07 | 1.4e-06 | 2.18e-06 | 0.77 | 0.75 | — | — |
| transformer | DSSIM (Chen & Koltun 2013) | reml | 0.00151 | 2.51e-05 | 0.000137 | 0.00014 | 0.80 | 0.84 | — | — |
| transformer | R_PSNR (PSNR of scene-mean MSE) | reml | 1.39 | 0.218 | 1.45 | 0.819 | 0.28 | 0.38 | 0.40 | 1.42 |
| transformer | PSNR after per-scene optimal RGB gain (H4) | reml | 3.82 | 0.025 | 0.145 | 0.474 | 0.89 | 0.86 | — | — |
| unet | si-MSE (gray, Grosse 2009) | reml | 0.000102 | 7.55e-16 | 5.2e-06 | 5.3e-06 | 0.93 | 0.91 | — | — |
| unet | LMSE (Chen & Koltun 2013) | reml | 1.64e-05 | 1.06e-07 | 5.2e-07 | 7.72e-07 | 0.91 | 0.93 | — | — |
| unet | DSSIM (Chen & Koltun 2013) | reml | 0.0015 | 3.65e-05 | 0.000113 | 5.27e-05 | 0.79 | 0.90 | — | — |
| unet | R_PSNR (PSNR of scene-mean MSE) | reml | 1.37 | 0.239 | 1.8 | 0.283 | 0.26 | 0.40 | 0.41 | 1.58 |
| unet | PSNR after per-scene optimal RGB gain (H4) | reml | 4.4 | 1.64e-11 | 0.217 | 0.16 | 0.93 | 0.92 | — | — |

## H4 — gauge: scene x training-set interaction under PSNR vs PSNR after per-scene optimal gain

| arch | s2_inter PSNR | s2_inter scene-gain PSNR | reduction | pre-registered >=50% | ICC PSNR -> scene-gain |
|---|---|---|---|---|---|
| direct | 1.12 | 0.107 | 0.90 | True | 0.67 -> 0.97 |
| proposed | 1.57 | 0.164 | 0.90 | True | 0.43 -> 0.92 |
| transformer | 1.45 | 0.145 | 0.90 | True | 0.38 -> 0.86 |
| unet | 1.8 | 0.217 | 0.88 | True | 0.40 -> 0.92 |

## H5 — seed variance by architecture (scene-level and run-level)

| arch | metric | s2_seed scene-level | s2_seed run-level | s2_scene | JOSA A rho (PSNR, partition/seed) |
|---|---|---|---|---|---|
| direct | R_PSNR_of_mean_mse | 0.308 | 0.095 | 2.83 | 15.0 |
| direct | simse_R_gray_mean | 1.97e-06 | 4.81e-07 | 0.000118 | 15.0 |
| proposed | R_PSNR_of_mean_mse | 0.443 | 0.118 | 1.55 | 10.4 |
| proposed | simse_R_gray_mean | 9.02e-06 | 2.31e-06 | 0.000112 | 10.4 |
| transformer | R_PSNR_of_mean_mse | 0.819 | 0.247 | 1.39 | 1.3 |
| transformer | simse_R_gray_mean | 1.31e-05 | 3.51e-06 | 8.5e-05 | 1.3 |
| unet | R_PSNR_of_mean_mse | 0.283 | 0.0798 | 1.37 | 7.6 |
| unet | simse_R_gray_mean | 5.3e-06 | 1.2e-06 | 0.000102 | 7.6 |

## H6 — stratified vs random partitions

| arch | metric | simulated V_test random | simulated V_test stratified | sim reduction | observed var(fold means) random | observed stratified | observed reduction | reps (rand/strat) |
|---|---|---|---|---|---|---|---|---|
| direct | simse_R_gray_mean | 1.87e-05 | 6.98e-06 | 0.63 | 1.64e-05 | 1e-05 | 0.39 | 5/2 |
| direct | lmse_R_chen_mean | 2.5e-06 | 1.55e-06 | 0.38 | 1.93e-06 | 2.14e-06 | -0.11 | 5/2 |
| direct | dssim_R_mean | 0.000259 | 0.000322 | -0.24 | 0.000407 | 0.000388 | 0.05 | 5/2 |
| direct | R_PSNR_of_mean_mse | 0.478 | 0.244 | 0.49 | 1.02 | 0.674 | 0.34 | 5/2 |
| direct | R_siPSNR_scenegain_rgb | 0.782 | 0.283 | 0.64 | 0.776 | 0.346 | 0.55 | 5/2 |
| proposed | simse_R_gray_mean | 1.8e-05 | 9.16e-06 | 0.49 | — | — | — | — |
| proposed | lmse_R_chen_mean | 2.33e-06 | 1.46e-06 | 0.37 | — | — | — | — |
| proposed | dssim_R_mean | 0.00023 | 0.000285 | -0.24 | — | — | — | — |
| proposed | R_PSNR_of_mean_mse | 0.289 | 0.215 | 0.26 | — | — | — | — |
| proposed | R_siPSNR_scenegain_rgb | 0.715 | 0.404 | 0.43 | — | — | — | — |
| transformer | simse_R_gray_mean | 1.37e-05 | 8.09e-06 | 0.41 | — | — | — | — |
| transformer | lmse_R_chen_mean | 1.74e-06 | 1.18e-06 | 0.32 | — | — | — | — |
| transformer | dssim_R_mean | 0.000241 | 0.000296 | -0.23 | — | — | — | — |
| transformer | R_PSNR_of_mean_mse | 0.277 | 0.19 | 0.32 | — | — | — | — |
| transformer | R_siPSNR_scenegain_rgb | 0.606 | 0.39 | 0.36 | — | — | — | — |
| unet | simse_R_gray_mean | 1.64e-05 | 9.05e-06 | 0.45 | 7.1e-06 | 1.72e-05 | -1.43 | 5/2 |
| unet | lmse_R_chen_mean | 2.59e-06 | 1.78e-06 | 0.31 | 1.41e-06 | 3.02e-06 | -1.15 | 5/2 |
| unet | dssim_R_mean | 0.000237 | 0.000292 | -0.23 | 0.000336 | 0.000388 | -0.15 | 5/2 |
| unet | R_PSNR_of_mean_mse | 0.266 | 0.219 | 0.18 | 0.465 | 0.87 | -0.87 | 5/2 |
| unet | R_siPSNR_scenegain_rgb | 0.695 | 0.438 | 0.37 | 0.423 | 1.01 | -1.38 | 5/2 |

## Absolute-score MDE ladder (alpha .05 two-sided, power .80)

| arch | metric | K1_S3 | K3_S3 | K5_S3 | K10_S3 | K23_S3 | fullcov_5fold_S3 |
|---|---|---|---|---|---|---|---|
| direct | simse_R_gray_mean | 0.0178 | 0.0103 | 0.00798 | 0.00564 | 0.00372 | 0.00232 |
| direct | lmse_R_chen_mean | 0.00644 | 0.00372 | 0.00288 | 0.00204 | 0.00134 | 0.000685 |
| direct | dssim_R_mean | 0.0676 | 0.039 | 0.0302 | 0.0214 | 0.0141 | 0.0105 |
| direct | R_PSNR_of_mean_mse | 4.17 | 2.41 | 1.87 | 1.32 | 0.87 | 1.44 |
| direct | R_siPSNR_scenegain_rgb | 3.65 | 2.11 | 1.63 | 1.15 | 0.762 | 0.453 |
| proposed | simse_R_gray_mean | 0.0175 | 0.0101 | 0.00784 | 0.00554 | 0.00365 | 0.00253 |
| proposed | lmse_R_chen_mean | 0.0066 | 0.00381 | 0.00295 | 0.00209 | 0.00138 | 0.00125 |
| proposed | dssim_R_mean | 0.0682 | 0.0394 | 0.0305 | 0.0216 | 0.0142 | 0.0149 |
| proposed | R_PSNR_of_mean_mse | 3.76 | 2.17 | 1.68 | 1.19 | 0.784 | 1.44 |
| proposed | R_siPSNR_scenegain_rgb | 3.5 | 2.02 | 1.57 | 1.11 | 0.73 | 0.478 |
| transformer | simse_R_gray_mean | 0.0159 | 0.00918 | 0.00711 | 0.00503 | 0.00331 | 0.00296 |
| transformer | lmse_R_chen_mean | 0.00593 | 0.00343 | 0.00265 | 0.00188 | 0.00124 | 0.00134 |
| transformer | dssim_R_mean | 0.0686 | 0.0396 | 0.0307 | 0.0217 | 0.0143 | 0.0142 |
| transformer | R_PSNR_of_mean_mse | 3.56 | 2.06 | 1.59 | 1.13 | 0.742 | 1.36 |
| transformer | R_siPSNR_scenegain_rgb | 3.29 | 1.9 | 1.47 | 1.04 | 0.687 | 0.542 |
| unet | simse_R_gray_mean | 0.0166 | 0.00956 | 0.00741 | 0.00524 | 0.00345 | 0.00213 |
| unet | lmse_R_chen_mean | 0.00666 | 0.00385 | 0.00298 | 0.00211 | 0.00139 | 0.000915 |
| unet | dssim_R_mean | 0.0685 | 0.0396 | 0.0307 | 0.0217 | 0.0143 | 0.0143 |
| unet | R_PSNR_of_mean_mse | 3.63 | 2.1 | 1.62 | 1.15 | 0.758 | 1.4 |
| unet | R_siPSNR_scenegain_rgb | 3.42 | 1.98 | 1.53 | 1.08 | 0.714 | 0.423 |

## H2 — all architecture pairs: scene x method interaction and paired MDE

| metric | pair | mean gap | s2(scene x method) | interaction share (K=1,S=3) | MDE K1,S3 | MDE K5,S3 | MDE full-coverage 5-fold,S3 | gap/MDE K1 | gap/MDE fullcov |
|---|---|---|---|---|---|---|---|---|---|
| R_PSNR_of_mean_mse | proposed - direct | +0.08556 | 2.15 | 0.81 | 1.81 | 0.808 | 0.352 | 0.05 | 0.24 |
| R_PSNR_of_mean_mse | transformer - direct | +0.2878 | 1.69 | 0.70 | 1.72 | 0.771 | 0.422 | 0.17 | 0.68 |
| R_PSNR_of_mean_mse | transformer - proposed | +0.2022 | 0.217 | 0.16 | 1.29 | 0.579 | 0.531 | 0.16 | 0.38 |
| R_PSNR_of_mean_mse | unet - direct | +0.1894 | 2.63 | 0.84 | 1.96 | 0.876 | 0.349 | 0.10 | 0.54 |
| R_PSNR_of_mean_mse | unet - proposed | +0.1038 | 0.05 | 0.10 | 0.774 | 0.346 | 0.328 | 0.13 | 0.32 |
| R_PSNR_of_mean_mse | unet - transformer | -0.09837 | 0.203 | 0.15 | 1.31 | 0.585 | 0.541 | 0.08 | 0.18 |
| R_siPSNR_scenegain_rgb | proposed - direct | -0.1724 | 0.73 | 0.71 | 1.12 | 0.501 | 0.268 | 0.15 | 0.64 |
| R_siPSNR_scenegain_rgb | transformer - direct | -0.1957 | 0.746 | 0.59 | 1.25 | 0.557 | 0.357 | 0.16 | 0.55 |
| R_siPSNR_scenegain_rgb | transformer - proposed | -0.02332 | 0.086 | 0.12 | 0.956 | 0.428 | 0.402 | 0.02 | 0.06 |
| R_siPSNR_scenegain_rgb | unet - direct | -0.2294 | 0.836 | 0.75 | 1.17 | 0.524 | 0.264 | 0.20 | 0.87 |
| R_siPSNR_scenegain_rgb | unet - proposed | -0.05701 | 0.023 | 0.09 | 0.559 | 0.25 | 0.239 | 0.10 | 0.24 |
| R_siPSNR_scenegain_rgb | unet - transformer | -0.0337 | 0.123 | 0.17 | 0.937 | 0.419 | 0.381 | 0.04 | 0.09 |
| dssim_R_mean | proposed - direct | -0.001221 | 0.000389 | 0.75 | 0.0253 | 0.0113 | 0.00568 | 0.05 | 0.21 |
| dssim_R_mean | transformer - direct | -0.000283 | 0.000342 | 0.65 | 0.0254 | 0.0114 | 0.00675 | 0.01 | 0.04 |
| dssim_R_mean | transformer - proposed | +0.000938 | 2.6e-05 | 0.10 | 0.0175 | 0.00783 | 0.00741 | 0.05 | 0.13 |
| dssim_R_mean | unet - direct | +0.001257 | 0.000447 | 0.76 | 0.0268 | 0.012 | 0.00585 | 0.05 | 0.22 |
| dssim_R_mean | unet - proposed | +0.002478 | 1.15e-05 | 0.20 | 0.00848 | 0.00379 | 0.0034 | 0.29 | 0.73 |
| dssim_R_mean | unet - transformer | +0.00154 | 2.6e-05 | 0.14 | 0.0152 | 0.0068 | 0.00631 | 0.10 | 0.24 |
| lmse_R_chen_mean | proposed - direct | +0.000688 | 5.05e-06 | 0.70 | 0.00298 | 0.00133 | 0.000729 | 0.23 | 0.94 |
| lmse_R_chen_mean | transformer - direct | +0.0003168 | 5.46e-06 | 0.67 | 0.00316 | 0.00141 | 0.000805 | 0.10 | 0.39 |
| lmse_R_chen_mean | transformer - proposed | -0.0003713 | 0 | 0.00 | 0.00281 | 0.00126 | 0.00126 | 0.13 | 0.30 |
| lmse_R_chen_mean | unet - direct | +0.001067 | 5.4e-06 | 0.85 | 0.0028 | 0.00125 | 0.000487 | 0.38 | 2.19 |
| lmse_R_chen_mean | unet - proposed | +0.0003788 | 7.84e-08 | 0.06 | 0.00128 | 0.000571 | 0.000554 | 0.30 | 0.68 |
| lmse_R_chen_mean | unet - transformer | +0.0007501 | 3.1e-07 | 0.08 | 0.00222 | 0.000994 | 0.000955 | 0.34 | 0.79 |
| simse_R_gray_mean | proposed - direct | +0.000725 | 3.09e-05 | 0.75 | 0.00714 | 0.00319 | 0.00161 | 0.10 | 0.45 |
| simse_R_gray_mean | transformer - direct | -0.0001563 | 3.18e-05 | 0.62 | 0.00792 | 0.00354 | 0.00217 | 0.02 | 0.07 |
| simse_R_gray_mean | transformer - proposed | -0.0008813 | 1.17e-06 | 0.04 | 0.00606 | 0.00271 | 0.00265 | 0.15 | 0.33 |
| simse_R_gray_mean | unet - direct | +0.0008483 | 2.76e-05 | 0.77 | 0.00665 | 0.00297 | 0.00143 | 0.13 | 0.59 |
| simse_R_gray_mean | unet - proposed | +0.0001233 | 1.04e-06 | 0.10 | 0.00353 | 0.00158 | 0.0015 | 0.03 | 0.08 |
| simse_R_gray_mean | unet - transformer | +0.001005 | 1.86e-06 | 0.08 | 0.00523 | 0.00234 | 0.00224 | 0.19 | 0.45 |

## H3 — out-of-sample prediction of JOSA A per-partition gaps (PSNR), all pairs

n = 30 (pairs x partitions), Pearson r = 0.868, sign agreement 19/30, RMSE 0.33 dB.

| pair | partition | predicted gap (dB) | JOSA A observed (dB) |
|---|---|---|---|
| proposed - direct | 101 | -0.43 | -0.29 |
| proposed - direct | 102 | +0.82 | +0.72 |
| proposed - direct | 103 | +0.48 | +0.07 |
| proposed - direct | 104 | -0.03 | +0.18 |
| proposed - direct | 105 | +1.33 | +1.27 |
| transformer - direct | 101 | -0.56 | -0.90 |
| transformer - direct | 102 | +1.38 | +0.85 |
| transformer - direct | 103 | +0.66 | +0.48 |
| transformer - direct | 104 | +0.17 | +0.02 |
| transformer - direct | 105 | +1.41 | +0.90 |
| unet - direct | 101 | -0.51 | -0.47 |
| unet - direct | 102 | +1.17 | +0.72 |
| unet - direct | 103 | +0.54 | -0.10 |
| unet - direct | 104 | -0.01 | +0.14 |
| unet - direct | 105 | +1.37 | +0.99 |
| transformer - proposed | 101 | -0.13 | -0.62 |
| transformer - proposed | 102 | +0.56 | +0.13 |
| transformer - proposed | 103 | +0.18 | +0.41 |
| transformer - proposed | 104 | +0.21 | -0.16 |
| transformer - proposed | 105 | +0.08 | -0.37 |
| unet - proposed | 101 | -0.08 | -0.18 |
| unet - proposed | 102 | +0.35 | -0.00 |
| unet - proposed | 103 | +0.07 | -0.17 |
| unet - proposed | 104 | +0.02 | -0.04 |
| unet - proposed | 105 | +0.04 | -0.28 |
| unet - transformer | 101 | +0.06 | +0.43 |
| unet - transformer | 102 | -0.21 | -0.13 |
| unet - transformer | 103 | -0.12 | -0.58 |
| unet - transformer | 104 | -0.18 | +0.12 |
| unet - transformer | 105 | -0.04 | +0.10 |

## Control — GPU non-determinism (same configuration run in v1 and v2)

common scene rows: 460

| metric | sd of difference (scene-level) | mean |diff| | implied single-run non-determinism variance |
|---|---|---|---|
| R_PSNR_of_mean_mse | 0.583 | 0.344 | 0.17 |
| simse_R_gray_mean | 0.00239 | 0.00134 | 2.85e-06 |
| lmse_R_chen_mean | 0.000974 | 0.000542 | 4.75e-07 |
| dssim_R_mean | 0.00871 | 0.00546 | 3.79e-05 |

## Per-scene difficulty table (si-MSE gray, DirectCNN; mean over 5 training compositions x 3 seeds)

| scene | si-MSE | n_frames |
|---|---|---|
| cave_4 | 0.05072 | 50 |
| temple_2 | 0.03800 | 50 |
| ambush_7 | 0.03551 | 50 |
| mountain_1 | 0.03060 | 50 |
| ambush_5 | 0.03017 | 50 |
| ambush_4 | 0.02358 | 33 |
| ambush_6 | 0.02339 | 20 |
| market_6 | 0.02281 | 40 |
| temple_3 | 0.02212 | 50 |
| bandage_2 | 0.01970 | 50 |
| market_2 | 0.01735 | 50 |
| alley_1 | 0.01677 | 50 |
| shaman_3 | 0.01569 | 50 |
| cave_2 | 0.01555 | 50 |
| shaman_2 | 0.01482 | 50 |
| sleeping_2 | 0.01470 | 50 |
| bandage_1 | 0.01377 | 50 |
| market_5 | 0.01241 | 50 |
| sleeping_1 | 0.01182 | 50 |
| ambush_2 | 0.01089 | 21 |
| alley_2 | 0.00999 | 50 |
| bamboo_2 | 0.00585 | 50 |
| bamboo_1 | 0.00532 | 50 |
