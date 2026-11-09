#!/usr/bin/env python3
"""make_summary_flow.py — render FLOW_V1_SUMMARY.md from flow_v1.json (no hand-typed numbers)."""
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); d = json.load(open(src))
L = [f"# Optical-flow arm v1 — results summary (generated from `{src.name}`)\n",
     f"CV runs: {d['n_runs']}; zero-shot C+T baselines: " + ", ".join(f"{a}: EPE {z['epe_pair_mean']:.3f}, Fl-all {z['fl_all']:.4f}" for a, z in d['zeroshot'].items()) + f". Every number is a key in `{src.name}`.\n"]
L.append("## F-H1 — variance decomposition of scene-level flow error (random design, REML)\n")
L.append("| arch | metric | s2_scene | s2_train | s2_inter | s2_seed | test-side share of predicted | ICC(scene) | decision |"); L.append("|---|---|---|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    for k, m in A['metrics'].items():
        c = m['reml'] if m['used'] == 'reml' else m['moment']
        L.append(f"| {a} | {m['label']} | {c['s2_scene']:.3g} | {c['s2_train']:.3g} | {c['s2_inter']:.3g} | {c['s2_seed']:.3g} | {m['share_test_side_of_predicted']:.2f} | {m['icc_scene_single_seed']:.2f} | {m.get('decision_FH1', '—')}{' / death' if m.get('death_icc_lt_0.5') else ''} |")
L.append("\n## Absolute-score MDE ladder (EPE, alpha .05 two-sided, power .80)\n")
keys = list(d['archs'][next(iter(d['archs']))]['metrics']['epe_mean']['MDE_absolute'].keys())
L.append("| arch | " + " | ".join(keys) + " | K1 / full-coverage ratio |"); L.append("|---|" + "---|" * (len(keys) + 1))
for a, A in d['archs'].items():
    lad = A['metrics']['epe_mean']['MDE_absolute']
    L.append(f"| {a} | " + " | ".join(f"{v:.3f}" for v in lad.values()) + f" | {lad['K1_S3'] / lad['fullcov_5fold_S3']:.1f}x |")
L.append("\n## F-H2 — raft_small − raft_large\n")
L.append("| metric | mean gap | s2(scene x method) | interaction share (K=1,S=3) | MDE K1,S3 | MDE full-coverage | gap/MDE K1 | gap/MDE fullcov |"); L.append("|---|---|---|---|---|---|---|---|")
for p in d['pairs']:
    m = p['MDE_paired']; L.append(f"| {p['metric']} | {p['mean_gap']:+.4g} | {p['s2_scene_x_method']:.3g} | {p['share_interaction_K1S3']:.2f} | {m['paired_K1_S3']:.3g} | {m['paired_fullcov_5fold_S3']:.3g} | {p['gap_over_MDE_K1']:.2f} | {p['gap_over_MDE_fullcov']:.2f} |")
L.append("\n## F-H3 — cross-task correlation with the intrinsic arm (same folds / same scenes)\n")
L.append("| comparison | n | Pearson | p | Spearman | p |"); L.append("|---|---|---|---|---|---|")
for k, v in d.get('FH3_cross_task', {}).items():
    n = v.get('n_folds', v.get('n_scenes')); L.append(f"| {k} | {n} | {v.get('pearson', float('nan')):.3f} | {v.get('p', float('nan')):.3f} | {v['spearman']:.3f} | {v.get('p_s', v.get('p', float('nan'))):.3f} |")
L.append("\n## F-H4 — zero-shot (C+T) scene difficulty vs fine-tuned scene difficulty\n")
L.append("| arch | Spearman | p | n scenes | mean scene EPE zero-shot | mean scene EPE fine-tuned |"); L.append("|---|---|---|---|---|---|")
for a, A in d['archs'].items():
    h = A.get('FH4_zeroshot_vs_finetuned_scene_difficulty')
    if h: L.append(f"| {a} | {h['spearman']:.3f} | {h['p']:.4f} | {h['n_scenes']} | {h['epe_zeroshot_mean']:.3f} | {h['epe_finetuned_scene_mean']:.3f} |")
L.append("\n## Per-scene difficulty (fine-tuned EPE, pair-weighted scene mean; mean over 5 training compositions x 3 seeds)\n")
tabs = {a: {r['scene']: r['score'] for r in A['metrics']['epe_mean']['scene_table']} for a, A in d['archs'].items()}
scenes = sorted(next(iter(tabs.values())), key=lambda s: -next(iter(tabs.values()))[s])
L.append("| scene | " + " | ".join(tabs.keys()) + " | zero-shot " + " | zero-shot ".join(d['zeroshot'].keys()) + " |"); L.append("|---|" + "---|" * (len(tabs) + len(d['zeroshot'])))
for s in scenes:
    L.append(f"| {s} | " + " | ".join(f"{tabs[a][s]:.3f}" for a in tabs) + " | " + " | ".join(f"{d['zeroshot'][a]['per_scene_epe'].get(s, float('nan')):.3f}" for a in d['zeroshot']) + " |")
L.append(f"\n## Compute\nmean minutes per run: " + ", ".join(f"{a} {A['mean_minutes']:.1f}" for a, A in d['archs'].items()) + f"; total ≈ {sum(A['mean_minutes'] * A['n_runs'] for A in d['archs'].values()) / 60:.1f} GPU-h (T4).\n")
dst.write_text("\n".join(L) + "\n"); print(dst)
