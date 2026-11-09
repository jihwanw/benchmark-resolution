#!/usr/bin/env python3
"""make_summary_cv2.py — render MAIN_V2_SUMMARY.md from main_v2.json (no hand-typed numbers)."""
import json, sys
from pathlib import Path
import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
d = json.load(open(src))
L = [f"# Main experiment v2 — results summary (generated from `{src.name}`)\n",
     f"runs: {d['n_runs']} {d['n_runs_by_design']}. Every number below is a key in `{src.name}`.\n"]
L.append("## Variance decomposition per architecture and metric (random design; REML unless noted)\n")
L.append("| arch | metric | est | s2_scene | s2_train | s2_inter | s2_seed (scene-level) | test-side share of predicted | ICC(scene, 1 seed) | share vs JOSA A observed (PSNR) | V_pred/obs (PSNR) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    for k, m in A['metrics'].items():
        c = m['reml'] if m['used'] == 'reml' else m['moment']
        sj = f"{m['share_test_side_vs_josaa_observed']:.2f}" if 'share_test_side_vs_josaa_observed' in m else '—'
        vp = f"{m['V_pred_over_josaa_observed']:.2f}" if 'V_pred_over_josaa_observed' in m else '—'
        L.append(f"| {a} | {m['label']} | {m['used']} | {c['s2_scene']:.3g} | {c['s2_train']:.3g} | {c['s2_inter']:.3g} | {c['s2_seed']:.3g} | "
                 f"{m['share_test_side_of_predicted']:.2f} | {m['icc_scene_single_seed']:.2f} | {sj} | {vp} |")
L.append("\n## H4 — gauge: scene x training-set interaction under PSNR vs PSNR after per-scene optimal gain\n")
L.append("| arch | s2_inter PSNR | s2_inter scene-gain PSNR | reduction | pre-registered >=50% | ICC PSNR -> scene-gain |"); L.append("|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    if 'H4_gauge' in A:
        h = A['H4_gauge']
        L.append(f"| {a} | {h['s2_inter_psnr']:.3g} | {h['s2_inter_scenegain_psnr']:.3g} | {h['interaction_reduction']:.2f} | {h['passes_pre_registered_50pct']} | {h['icc_psnr']:.2f} -> {h['icc_scenegain_psnr']:.2f} |")
L.append("\n## H5 — seed variance by architecture (scene-level and run-level)\n")
L.append("| arch | metric | s2_seed scene-level | s2_seed run-level | s2_scene | JOSA A rho (PSNR, partition/seed) |"); L.append("|---|---|---|---|---|---|")
for a, H in d['H5_seed_variance'].items():
    for y, v in H.items():
        L.append(f"| {a} | {y} | {v['s2_seed_scene_level']:.3g} | {v['s2_seed_run_level']:.3g} | {v['s2_scene']:.3g} | {d['archs'][a]['josaa_rho_psnr']} |")
L.append("\n## H6 — stratified vs random partitions\n")
L.append("| arch | metric | simulated V_test random | simulated V_test stratified | sim reduction | observed var(fold means) random | observed stratified | observed reduction | reps (rand/strat) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    for k, m in A['metrics'].items():
        o = A.get('H6_observed', {}).get(k)
        ob = f"{o['var_fold_means_random']:.3g} | {o['var_fold_means_stratified']:.3g} | {o['reduction']:.2f} | {o['n_rand_reps']}/{o['n_strat_reps']}" if o else "— | — | — | —"
        L.append(f"| {a} | {k} | {m['H6_sim_V_test_random']:.3g} | {m['H6_sim_V_test_stratified']:.3g} | {m['H6_sim_variance_reduction']:.2f} | {ob} |")
L.append("\n## Absolute-score MDE ladder (alpha .05 two-sided, power .80)\n")
keys = list(next(iter(d['archs'].values()))['metrics'].values())[0]['MDE_absolute'].keys()
L.append("| arch | metric | " + " | ".join(keys) + " |"); L.append("|---|---|" + "---|" * len(keys))
for a, A in d['archs'].items():
    for k, m in A['metrics'].items():
        L.append(f"| {a} | {k} | " + " | ".join(f"{v:.3g}" for v in m['MDE_absolute'].values()) + " |")
L.append("\n## H2 — all architecture pairs: scene x method interaction and paired MDE\n")
L.append("| metric | pair | mean gap | s2(scene x method) | interaction share (K=1,S=3) | MDE K1,S3 | MDE K5,S3 | MDE full-coverage 5-fold,S3 | gap/MDE K1 | gap/MDE fullcov |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(d['pairs'], key=lambda p: (p['metric'], p['pair'])):
    m = p['MDE_paired']
    L.append(f"| {p['metric']} | {p['pair']} | {p['mean_gap']:+.4g} | {p['s2_scene_x_method']:.3g} | {p['share_interaction_K1S3']:.2f} | "
             f"{m['paired_K1_S3']:.3g} | {m['paired_K5_S3']:.3g} | {m['paired_fullcov_5fold_S3']:.3g} | {p['gap_over_MDE_K1']:.2f} | {p['gap_over_MDE_fullcov']:.2f} |")
v = d.get('H3_josaa_validation')
if v and 'pearson_r' in v:
    L.append(f"\n## H3 — out-of-sample prediction of JOSA A per-partition gaps (PSNR), all pairs\n")
    L.append(f"n = {v['n']} (pairs x partitions), Pearson r = {v['pearson_r']:.3f}, sign agreement {v['sign_agreement']}/{v['n']}, RMSE {v['rmse_dB']:.2f} dB.\n")
    L.append("| pair | partition | predicted gap (dB) | JOSA A observed (dB) |"); L.append("|---|---|---|---|")
    for r in v['rows']:
        L.append(f"| {r['pair']} | {r['partition']} | {r['predicted_gap_dB']:+.2f} | {r['observed_gap_dB']:+.2f} |")
nd = d.get('nondeterminism_control')
if nd:
    L.append("\n## Control — GPU non-determinism (same configuration run in v1 and v2)\n")
    L.append(f"common scene rows: {nd['n_common_scene_rows']}\n")
    L.append("| metric | sd of difference (scene-level) | mean |diff| | implied single-run non-determinism variance |"); L.append("|---|---|---|---|")
    for k, vv in nd.items():
        if isinstance(vv, dict):
            L.append(f"| {k} | {vv['sd_diff_scene_level']:.3g} | {vv['mean_abs_diff']:.3g} | {vv['nondet_var_single_run']:.3g} |")
L.append("\n## Per-scene difficulty table (si-MSE gray, DirectCNN; mean over 5 training compositions x 3 seeds)\n")
tab = sorted(d['archs']['direct']['metrics']['simse_R_gray_mean']['scene_table'], key=lambda r: -r['score']) if 'direct' in d['archs'] else []
L.append("| scene | si-MSE | n_frames |"); L.append("|---|---|---|")
for r in tab: L.append(f"| {r['scene']} | {r['score']:.5f} | {r['n']} |")
dst.write_text("\n".join(L) + "\n"); print(dst)
