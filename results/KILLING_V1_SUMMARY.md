# Killing experiment v1 — results summary (generated from `killing_v1.json`)

cv runs: 100 (+2 parity). Source of every number: `killing_v1.json`.

## H1 — decomposition of scene-level reflectance score, per architecture and metric

| arch | metric | s2_scene | s2_train | s2_inter | s2_seed | test-side share of predicted | ICC(scene) | H1 decision (PSNR only) | death flag |
|---|---|---|---|---|---|---|---|---|---|
| direct | R_PSNR (PSNR of scene-mean MSE, JOSA A convention) | 2.72 | 0.361 | 1.22 | 0.241 | 0.41 | 0.65 | mixed (0.3<s<0.7) | False |
| direct | si-MSE (gray, Grosse 2009) | 0.000119 | 4.05e-07 | 4.66e-06 | 2.2e-06 | 0.93 | 0.95 | — | — |
| direct | LMSE (Chen & Koltun 2013) | 1.64e-05 | 5.15e-08 | 3.73e-07 | 3.44e-07 | 0.95 | 0.96 | — | — |
| direct | DSSIM (Chen & Koltun 2013) | 0.00169 | 9.95e-06 | 8.11e-05 | 5.28e-05 | 0.90 | 0.93 | — | — |
| unet | R_PSNR (PSNR of scene-mean MSE, JOSA A convention) | 1.49 | 0.336 | 1.53 | 0.379 | 0.26 | 0.44 | mixed (0.3<s<0.7) | True |
| unet | si-MSE (gray, Grosse 2009) | 0.000112 | 2.77e-07 | 3.64e-06 | 6e-06 | 0.94 | 0.92 | — | — |
| unet | LMSE (Chen & Koltun 2013) | 1.56e-05 | 1.25e-07 | 4.13e-07 | 7.72e-07 | 0.92 | 0.93 | — | — |
| unet | DSSIM (Chen & Koltun 2013) | 0.00145 | 3.24e-05 | 7.9e-05 | 4.14e-05 | 0.82 | 0.92 | — | — |

## PSNR: comparison with the JOSA A grid (observed sigma2_part from numbers_josaa.tex)

| arch | JOSA A sigma2_part | V_test closed | V_test sim (frame-weighted) | share closed | share sim | share 95% CI (2-way bootstrap) | V_pred / observed | implied S2_gamma (pure test-side) | estimated S2_gamma | between-scene sd (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| direct | 0.927 | 0.426 | 0.483 | 0.46 | 0.52 | [0.22, 0.85] | 1.12 | 5.92 | 2.72 | 1.65 |
| unet | 0.524 | 0.233 | 0.280 | 0.44 | 0.54 | [0.10, 0.97] | 1.71 | 3.35 | 1.49 | 1.22 |

## Absolute-score MDE (dB, alpha=.05 two-sided, power .80), PSNR

| arch | K1_S3 | K3_S3 | K5_S3 | K10_S3 | K23_S3 | fullcov_5fold_S3 |
|---|---|---|---|---|---|---|
| direct | 4.07 | 2.35 | 1.82 | 1.29 | 0.85 | 1.41 |
| unet | 3.79 | 2.19 | 1.70 | 1.20 | 0.79 | 1.47 |

## H2 — paired comparison UNet − DirectCNN

| metric | mean gap | s2(scene x method) | interaction share of paired var (K=1,S=3) | MDE K1,S3 | MDE K3,S3 | MDE K5,S3 | MDE full-coverage 5-fold,S3 | gap / MDE(K1) |
|---|---|---|---|---|---|---|---|---|
| R_PSNR_of_mean_mse | +0.2229 | 2.39 | 0.74 | 1.98 | 1.15 | 0.888 | 0.449 | 0.11 |
| simse_R_gray_mean | +0.0006523 | 2.75e-05 | 0.82 | 0.00642 | 0.0037 | 0.00287 | 0.00121 | 0.10 |
| lmse_R_chen_mean | +0.000843 | 3.96e-06 | 0.84 | 0.00241 | 0.00139 | 0.00108 | 0.00043 | 0.35 |
| dssim_R_mean | -0.001014 | 0.00048 | 0.82 | 0.0267 | 0.0154 | 0.012 | 0.00501 | 0.04 |

## Per-scene table (DirectCNN mean R_PSNR over 5 training compositions x 2 seeds; UNet − DirectCNN gap, dB)

| scene | DirectCNN PSNR | UNet − DirectCNN |
|---|---|---|
| ambush_6 | 12.03 | +2.56 |
| ambush_5 | 12.27 | +3.94 |
| cave_4 | 12.55 | +1.13 |
| ambush_4 | 13.26 | +2.88 |
| temple_2 | 13.76 | -0.11 |
| ambush_7 | 14.26 | -1.37 |
| ambush_2 | 14.35 | +1.21 |
| mountain_1 | 14.59 | -0.69 |
| bamboo_1 | 14.67 | +0.71 |
| cave_2 | 15.16 | +3.26 |
| bamboo_2 | 15.45 | +0.07 |
| temple_3 | 15.55 | -0.22 |
| market_6 | 15.70 | -0.72 |
| alley_1 | 15.70 | +0.02 |
| alley_2 | 15.95 | +0.22 |
| shaman_3 | 16.23 | -0.98 |
| bandage_2 | 16.73 | -0.83 |
| bandage_1 | 16.81 | -1.47 |
| sleeping_2 | 16.93 | -0.69 |
| shaman_2 | 17.11 | -1.18 |
| market_2 | 17.13 | -0.67 |
| market_5 | 17.40 | -1.01 |
| sleeping_1 | 17.87 | -0.94 |

Spearman(scene difficulty, UNet gain) = -0.683 (p = 0.0003); scenes favoring UNet 10, DirectCNN 13; gap sd 1.57 dB.

## Parity with the JOSA A reference configuration (scene partition 101, seed 42)

| arch | new R_PSNR (batch convention) | JOSA A results_t4 | diff (dB) | minutes |
|---|---|---|---|---|
| direct | 16.311 | 16.432 | -0.121 | 14.3 |
| unet | 15.793 | 15.765 | +0.028 | 22.8 |
