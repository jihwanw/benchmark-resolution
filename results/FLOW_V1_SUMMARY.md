# Optical-flow arm v1 — results summary (generated from `flow_v1.json`)

CV runs: 150; zero-shot C+T baselines: raft_large: EPE 1.400, Fl-all 0.0408, raft_small: EPE 2.047, Fl-all 0.0688. Every number is a key in `flow_v1.json`.

## F-H1 — variance decomposition of scene-level flow error (random design, REML)

| arch | metric | s2_scene | s2_train | s2_inter | s2_seed | test-side share of predicted | ICC(scene) | decision |
|---|---|---|---|---|---|---|---|---|
| raft_large | EPE (pair-weighted scene mean) | 3.02 | 0.00164 | 0.0102 | 0.0313 | 0.99 | 0.99 | test-side dominated (s>=0.7) |
| raft_large | EPE (pixel-weighted) | 3.01 | 0.00163 | 0.0101 | 0.0313 | 0.99 | 0.99 | — |
| raft_large | Fl-all | 0.00246 | 1.74e-07 | 6.86e-07 | 7.62e-07 | 1.00 | 1.00 | — |
| raft_large | >3px outlier rate | 0.00283 | 1.3e-07 | 5.54e-07 | 8.13e-07 | 1.00 | 1.00 | — |
| raft_large | EPE occluded | 22.9 | 0.00343 | 0.0383 | 0.216 | 0.99 | 0.99 | — |
| raft_large | EPE non-occluded | 0.678 | 0.000417 | 0.00504 | 0.00639 | 0.98 | 0.98 | — |
| raft_small | EPE (pair-weighted scene mean) | 10.4 | 0.00659 | 0.0746 | 0.109 | 0.98 | 0.98 | test-side dominated (s>=0.7) |
| raft_small | EPE (pixel-weighted) | 10.4 | 0.00658 | 0.0748 | 0.11 | 0.98 | 0.98 | — |
| raft_small | Fl-all | 0.0062 | 1.09e-06 | 1.75e-06 | 1.82e-06 | 1.00 | 1.00 | — |
| raft_small | >3px outlier rate | 0.00684 | 1.17e-06 | 1.78e-06 | 1.88e-06 | 1.00 | 1.00 | — |
| raft_small | EPE occluded | 48.1 | 0.0469 | 0.188 | 0.24 | 0.99 | 0.99 | — |
| raft_small | EPE non-occluded | 5.45 | 0.000737 | 0.0348 | 0.104 | 0.99 | 0.98 | — |

## Absolute-score MDE ladder (EPE, alpha .05 two-sided, power .80)

| arch | K1_S3 | K3_S3 | K5_S3 | K10_S3 | K23_S3 | fullcov_5fold_S3 | K1 / full-coverage ratio |
|---|---|---|---|---|---|---|---|
| raft_large | 2.743 | 1.584 | 1.227 | 0.867 | 0.572 | 0.138 | 19.9x |
| raft_small | 5.094 | 2.941 | 2.278 | 1.611 | 1.062 | 0.306 | 16.6x |

## F-H2 — raft_small − raft_large

| metric | mean gap | s2(scene x method) | interaction share (K=1,S=3) | MDE K1,S3 | MDE full-coverage | gap/MDE K1 | gap/MDE fullcov |
|---|---|---|---|---|---|---|---|
| epe_mean | +1.072 | 2.46 | 0.94 | 1.79 | 0.188 | 0.60 | 5.70 |
| epe_pixel_weighted | +1.072 | 2.47 | 0.94 | 1.79 | 0.188 | 0.60 | 5.69 |
| fl_all_mean | +0.03271 | 0.000931 | 1.00 | 0.0339 | 0.00101 | 0.96 | 32.52 |
| out3_mean | +0.03304 | 0.000953 | 0.99 | 0.0343 | 0.00119 | 0.96 | 27.74 |
| epe_occ_mean | +2.376 | 5 | 0.92 | 2.59 | 0.332 | 0.92 | 7.15 |
| epe_noc_mean | +0.8734 | 2.33 | 0.96 | 1.73 | 0.154 | 0.51 | 5.68 |

## F-H3 — cross-task correlation with the intrinsic arm (same folds / same scenes)

| comparison | n | Pearson | p | Spearman | p |
|---|---|---|---|---|---|
| raft_large|flow_epe~intrinsic_simse_R_gray_mean | 25 | 0.345 | 0.091 | 0.315 | 0.126 |
| raft_small|flow_epe~intrinsic_simse_R_gray_mean | 25 | 0.411 | 0.041 | 0.304 | 0.140 |
| raft_large|flow_epe~intrinsic_R_PSNR_of_mean_mse | 25 | -0.087 | 0.680 | -0.029 | 0.890 |
| raft_small|flow_epe~intrinsic_R_PSNR_of_mean_mse | 25 | -0.108 | 0.608 | -0.065 | 0.756 |
| raft_large|flow_epe~intrinsic_lmse_R_chen_mean | 25 | 0.211 | 0.312 | 0.178 | 0.393 |
| raft_small|flow_epe~intrinsic_lmse_R_chen_mean | 25 | 0.237 | 0.253 | 0.140 | 0.504 |
| raft_large|scene_level flow_epe~intrinsic_simse_R_gray_mean | 23 | nan | 0.609 | 0.113 | 0.609 |
| raft_large|scene_level flow_epe~intrinsic_R_PSNR_of_mean_mse | 23 | nan | 0.433 | -0.172 | 0.433 |
| raft_small|scene_level flow_epe~intrinsic_simse_R_gray_mean | 23 | nan | 0.520 | 0.141 | 0.520 |
| raft_small|scene_level flow_epe~intrinsic_R_PSNR_of_mean_mse | 23 | nan | 0.354 | -0.203 | 0.354 |

## F-H4 — zero-shot (C+T) scene difficulty vs fine-tuned scene difficulty

| arch | Spearman | p | n scenes | mean scene EPE zero-shot | mean scene EPE fine-tuned |
|---|---|---|---|---|---|
| raft_large | 0.999 | 0.0000 | 23 | 1.673 | 1.478 |
| raft_small | 0.997 | 0.0000 | 23 | 2.335 | 2.550 |

## Per-scene difficulty (fine-tuned EPE, pair-weighted scene mean; mean over 5 training compositions x 3 seeds)

| scene | raft_large | raft_small | zero-shot raft_large | zero-shot raft_small |
|---|---|---|---|---|
| ambush_4 | 5.712 | 12.402 | 7.580 | 9.005 |
| ambush_2 | 4.831 | 6.590 | 6.071 | 6.398 |
| market_5 | 4.643 | 8.532 | 4.722 | 7.639 |
| cave_2 | 3.258 | 4.879 | 3.387 | 4.767 |
| ambush_6 | 3.069 | 4.701 | 3.505 | 4.422 |
| temple_3 | 2.651 | 5.412 | 2.570 | 4.164 |
| ambush_5 | 1.929 | 3.170 | 2.003 | 3.578 |
| cave_4 | 1.794 | 3.107 | 1.878 | 3.144 |
| market_6 | 1.697 | 2.760 | 1.753 | 2.953 |
| temple_2 | 1.263 | 1.922 | 1.569 | 2.185 |
| bamboo_2 | 0.538 | 0.816 | 0.606 | 0.877 |
| market_2 | 0.375 | 0.595 | 0.403 | 0.634 |
| bamboo_1 | 0.373 | 0.575 | 0.391 | 0.585 |
| bandage_1 | 0.322 | 0.581 | 0.356 | 0.590 |
| ambush_7 | 0.251 | 0.425 | 0.286 | 0.460 |
| bandage_2 | 0.222 | 0.355 | 0.239 | 0.372 |
| shaman_2 | 0.194 | 0.273 | 0.210 | 0.298 |
| mountain_1 | 0.183 | 0.326 | 0.198 | 0.335 |
| alley_1 | 0.167 | 0.289 | 0.183 | 0.308 |
| alley_2 | 0.157 | 0.312 | 0.169 | 0.322 |
| shaman_3 | 0.154 | 0.269 | 0.164 | 0.288 |
| sleeping_1 | 0.105 | 0.192 | 0.118 | 0.205 |
| sleeping_2 | 0.101 | 0.166 | 0.109 | 0.180 |

## Compute
mean minutes per run: raft_large 46.9, raft_small 13.8; total ≈ 75.8 GPU-h (T4).

