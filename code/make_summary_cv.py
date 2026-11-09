#!/usr/bin/env python3
"""make_summary_cv.py — render KILLING_V1_SUMMARY.md from killing_v1.json (numbers are never typed by hand)."""
import json, sys
from pathlib import Path
import numpy as np

src = Path(sys.argv[1]); dst = Path(sys.argv[2])
d = json.load(open(src))
L = []
L.append(f"# Killing experiment v1 — results summary (generated from `{src.name}`)\n")
L.append(f"cv runs: {d['n_cv_runs']} (+{len(d['parity'])} parity). Source of every number: `{src.name}`.\n")
L.append("## H1 — decomposition of scene-level reflectance score, per architecture and metric\n")
L.append("| arch | metric | s2_scene | s2_train | s2_inter | s2_seed | test-side share of predicted | ICC(scene) | H1 decision (PSNR only) | death flag |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    for k, m in A['metrics'].items():
        c = m['reml'] if m['used'] == 'reml' else m['moment']
        dec = m.get('decision_H1', '—'); death = m.get('death_criterion_icc_lt_0.5', '—')
        L.append(f"| {a} | {m['label']} | {c['s2_scene']:.3g} | {c['s2_train']:.3g} | {c['s2_inter']:.3g} | {c['s2_seed']:.3g} | "
                 f"{m['share_test_side_of_predicted']:.2f} | {m['icc_scene_single_seed']:.2f} | {dec} | {death} |")
L.append("\n## PSNR: comparison with the JOSA A grid (observed sigma2_part from numbers_josaa.tex)\n")
L.append("| arch | JOSA A sigma2_part | V_test closed | V_test sim (frame-weighted) | share closed | share sim | share 95% CI (2-way bootstrap) | V_pred / observed | implied S2_gamma (pure test-side) | estimated S2_gamma | between-scene sd (dB) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    m = A['metrics']['R_PSNR_of_mean_mse']; b = m['bootstrap']['share_ci95']
    L.append(f"| {a} | {A['josaa_sigma2_part_obs']:.3f} | {m['V_test_closed_form']:.3f} | {m['V_test_sim_frame_weighted']:.3f} | "
             f"{m['share_test_side_closed']:.2f} | {m['share_test_side_sim_fw']:.2f} | [{b[0]:.2f}, {b[1]:.2f}] | {m['V_pred_over_observed']:.2f} | "
             f"{m['H3_implied_s2_gamma_pure_test_side']:.2f} | {m['H3_estimated_s2_gamma']:.2f} | {m['H3_estimated_between_scene_sd_dB']:.2f} |")
L.append("\n## Absolute-score MDE (dB, alpha=.05 two-sided, power .80), PSNR\n")
L.append("| arch | " + " | ".join(d['archs']['direct']['metrics']['R_PSNR_of_mean_mse']['MDE_absolute_dB'].keys()) + " |")
L.append("|---|" + "---|" * len(d['archs']['direct']['metrics']['R_PSNR_of_mean_mse']['MDE_absolute_dB']))
for a, A in d['archs'].items():
    lad = A['metrics']['R_PSNR_of_mean_mse']['MDE_absolute_dB']
    L.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in lad.values()) + " |")
L.append("\n## H2 — paired comparison UNet − DirectCNN\n")
L.append("| metric | mean gap | s2(scene x method) | interaction share of paired var (K=1,S=3) | MDE K1,S3 | MDE K3,S3 | MDE K5,S3 | MDE full-coverage 5-fold,S3 | gap / MDE(K1) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for p in d['pairs']:
    if not p: continue
    m = p['MDE_paired']
    L.append(f"| {p['metric']} | {p['mean_gap']:+.4g} | {p['s2_scene_x_method_interaction']:.3g} | {p['share_interaction_in_paired_var_K1']:.2f} | "
             f"{m['paired_K1_S3']:.3g} | {m['paired_K3_S3']:.3g} | {m['paired_K5_S3']:.3g} | {m['paired_fullcov_5fold_S3']:.3g} | {abs(p['mean_gap'])/m['paired_K1_S3']:.2f} |")
p0 = d['pairs'][0]; g = p0['per_scene_gap']
tab = {r['scene']: r['psnr'] for r in d['archs']['direct']['metrics']['R_PSNR_of_mean_mse']['scene_table']}
sc = sorted(tab, key=lambda s: tab[s])
L.append("\n## Per-scene table (DirectCNN mean R_PSNR over 5 training compositions x 2 seeds; UNet − DirectCNN gap, dB)\n")
L.append("| scene | DirectCNN PSNR | UNet − DirectCNN |"); L.append("|---|---|---|")
for s in sc:
    L.append(f"| {s} | {tab[s]:.2f} | {g[s]:+.2f} |")
x = np.array([tab[s] for s in sc]); gg = np.array([g[s] for s in sc])
from scipy.stats import spearmanr
rho, pv = spearmanr(x, gg)
L.append(f"\nSpearman(scene difficulty, UNet gain) = {rho:.3f} (p = {pv:.4f}); scenes favoring UNet {int((gg>0).sum())}, DirectCNN {int((gg<0).sum())}; gap sd {gg.std(ddof=1):.2f} dB.\n")
L.append("## Parity with the JOSA A reference configuration (scene partition 101, seed 42)\n")
L.append("| arch | new R_PSNR (batch convention) | JOSA A results_t4 | diff (dB) | minutes |"); L.append("|---|---|---|---|---|")
for r in d['parity']:
    L.append(f"| {r['arch']} | {r['new_R_PSNR_batchconv']:.3f} | {r['josaa_results_t4_R_PSNR']:.3f} | {r['diff_dB']:+.3f} | {r['minutes']:.1f} |")
dst.write_text("\n".join(L) + "\n")
print(dst)
