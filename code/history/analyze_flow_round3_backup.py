#!/usr/bin/env python3
"""analyze_flow.py — pre-registered analysis of the Sintel optical-flow arm (v1).

Hypotheses (ideas/2026-11-01_benchmark_resolution.md, "두 번째 벤치마크 arm"):
  F-H1  variance decomposition of scene-level EPE (pair-weighted): test-side share >=0.7 general / <=0.3 intrinsic-specific / mixed;
        death if scene ICC < 0.5.  Also for Fl-all, out3, pixel-weighted EPE, occ/noc EPE (metric-convention robustness).
  F-H2  raft_small vs raft_large: scene x method interaction share of paired variance (same family -> expect small, 0.00-0.20 as in intrinsic).
  F-H3  cross-task: fold-level (rep, fold) scores of the intrinsic arm (v2, 4 archs, si-MSE and PSNR) vs flow EPE fold means -> Pearson/Spearman.
  F-H4  zero-shot (C+T) per-scene EPE vs fine-tuned scene difficulty (scene main effect) -> Spearman.
Uses the estimators of analyze_cv2.py (REML crossed VCs + moment cross-check).
"""
import argparse, glob, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import analyze_cv2 as A

warnings.filterwarnings('ignore')
U, N_TEST, Z = 23, 5, 1.959964 + 0.841621
METRICS = {'epe_mean': 'EPE (pair-weighted scene mean)', 'epe_pixel_weighted': 'EPE (pixel-weighted)',
           'fl_all_mean': 'Fl-all', 'out3_mean': '>3px outlier rate', 'epe_occ_mean': 'EPE occluded', 'epe_noc_mean': 'EPE non-occluded'}


def load(results_dir):
    rows, zs, seen = [], {}, set()
    for fp in sorted(glob.glob(str(Path(results_dir) / '**' / 'flow_*_v1.json'), recursive=True)):
        n = Path(fp).name
        if n in seen or 'folds' in n or n.startswith('calib'): continue
        seen.add(n); d = json.load(open(fp))
        if d.get('kind') == 'zeroshot':
            zs[d['arch']] = d; continue
        if d.get('kind') != 'cv': continue
        for sc, ps in d['per_scene'].items():
            rows.append(dict(arch=d['arch'], design='rand', rep=d['rep'], fold=d['fold'], seed=d['seed'], scene=sc,
                             trainset=f"r{d['rep']}f{d['fold']}", scene_rep=f"{sc}|r{d['rep']}", n_frames=ps['n_pairs'],
                             R_MSE_mean=np.nan, minutes=d['minutes'], **{k: ps.get(k, np.nan) for k in METRICS}))
    return pd.DataFrame(rows), zs


def decompose(df, arch, ycol):
    d = df[df['arch'] == arch]
    reml = A.reml_components(d, ycol); mom = A.moment_components(d, ycol); comp, used = A.pick(reml, mom)
    s2s, s2t, s2i, s2e = comp['s2_scene'], comp['s2_train'], comp['s2_inter'], comp['s2_seed']
    v_test = (1 - N_TEST / U) * s2s / N_TEST; s2_run = A.run_level_seed_var(d, ycol)
    v_pred = v_test + s2t + s2i / N_TEST + s2_run / 6
    den = s2s + s2i + s2e
    lad = {f'K{K}_S3': Z * math.sqrt(2 * ((v_test + s2t + s2i / N_TEST) / K + s2_run / (3 * K))) for K in [1, 3, 5, 10, 23]}
    lad['fullcov_5fold_S3'] = Z * math.sqrt(2 * (A.C_T * s2t + s2i / U + A.C_T * s2_run / 3))   # exact for folds 5,5,5,4,4
    rep_means = d.groupby(['rep', 'scene'])[ycol].mean().groupby(level=0).mean(); pred = float(A.C_T * s2t + s2i / U + A.C_T * s2_run / 3)
    fc = {'observed_var_repeat_means': float(rep_means.var(ddof=1)), 'n_repeats': int(len(rep_means)), 'predicted_var_fullcov_S3': pred, 'ratio_obs_pred': float(rep_means.var(ddof=1) / pred) if pred > 0 else float('nan')}
    return {'label': METRICS[ycol], 'used': used, 'reml': reml, 'moment': mom, 'reml_consistency_ratio': A.reml_consistency(reml, mom), 'V_test_closed_form': v_test, 's2_seed_run_level': s2_run, 'fullcov_check': fc,
            'V_pred_partition_var': v_pred, 'share_test_side_of_predicted': v_test / v_pred if v_pred > 0 else np.nan,
            'icc_scene_single_seed': s2s / den if den > 0 else np.nan, 'MDE_absolute': lad,
            'scene_table': d.groupby('scene')[ycol].mean().reset_index().rename(columns={ycol: 'score'}).to_dict('records')}


def fold_means(df, ycol):
    """Fold-level score (rep, fold): mean over scenes (equal weight) and seeds."""
    return df.groupby(['rep', 'fold', 'seed'])[ycol].mean().groupby(level=[0, 1]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir'); ap.add_argument('out_json')
    ap.add_argument('--intrinsic-v2', default='/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution/results_cv_v2')
    args = ap.parse_args()
    df, zs = load(args.results_dir)
    archs = sorted(df['arch'].unique()) if len(df) else []
    out = {'version': 'v1', 'n_runs': int(df.groupby(['arch', 'rep', 'fold', 'seed']).ngroups) if len(df) else 0, 'archs': {}, 'pairs': [],
           'zeroshot': {a: {'epe_pair_mean': z['agg']['epe_pair_mean'], 'fl_all': z['agg']['fl_all_pair_mean'],
                            'per_scene_epe': {sc: ps['epe_mean'] for sc, ps in z['per_scene'].items()}} for a, z in zs.items()}}
    for a in archs:
        d = df[df['arch'] == a]
        out['archs'][a] = {'n_runs': int(d.groupby(['rep', 'fold', 'seed']).ngroups), 'mean_minutes': float(d.groupby(['rep', 'fold', 'seed'])['minutes'].first().mean()),
                           'metrics': {y: decompose(df, a, y) for y in METRICS if not d[y].isna().all()}}
        m = out['archs'][a]['metrics'].get('epe_mean')
        if m:
            s = m['share_test_side_of_predicted']
            m['decision_FH1'] = 'test-side dominated (s>=0.7)' if s >= 0.7 else ('train-side/interaction dominated (s<=0.3)' if s <= 0.3 else 'mixed')
            m['death_icc_lt_0.5'] = bool(m['icc_scene_single_seed'] < 0.5)
        # F-H4: zero-shot per-scene EPE vs fine-tuned scene difficulty (scene means)
        if a in zs and m:
            zt = out['zeroshot'][a]['per_scene_epe']; ft = {r['scene']: r['score'] for r in m['scene_table']}
            sc = sorted(set(zt) & set(ft)); x = [zt[s] for s in sc]; y = [ft[s] for s in sc]
            rho, p = spearmanr(x, y); out['archs'][a]['FH4_zeroshot_vs_finetuned_scene_difficulty'] = {'spearman': float(rho), 'p': float(p), 'n_scenes': len(sc),
                'epe_zeroshot_mean': float(np.mean(x)), 'epe_finetuned_scene_mean': float(np.mean(y))}
    # F-H2 pairs
    if len(archs) >= 2:
        for y in METRICS:
            r = A.analyze_pair(df, archs[0], archs[1], y)
            if r: out['pairs'].append(r)
    # F-H3 cross-task fold correlation with the intrinsic arm (v2)
    if Path(args.intrinsic_v2).exists() and len(df):
        di = A.load_runs(args.intrinsic_v2); di = di[di['design'] == 'rand']
        xt = {}
        for y in ['simse_R_gray_mean', 'R_PSNR_of_mean_mse', 'lmse_R_chen_mean']:
            fi = di.groupby(['rep', 'fold', 'seed', 'arch'])[y].mean().groupby(level=[0, 1]).mean()     # mean over 4 archs and seeds
            for a in archs:
                ff = fold_means(df[df['arch'] == a], 'epe_mean'); idx = fi.index.intersection(ff.index)
                if len(idx) >= 5:
                    pr, pp = pearsonr(fi.loc[idx].values, ff.loc[idx].values); sr, sp = spearmanr(fi.loc[idx].values, ff.loc[idx].values)
                    xt[f'{a}|flow_epe~intrinsic_{y}'] = {'n_folds': int(len(idx)), 'pearson': float(pr), 'p': float(pp), 'spearman': float(sr), 'p_s': float(sp)}
        # scene-level cross-task correlation (scene main effects)
        for a in archs:
            fs = df[df['arch'] == a].groupby('scene')['epe_mean'].mean()
            for y in ['simse_R_gray_mean', 'R_PSNR_of_mean_mse']:
                si = di.groupby('scene')[y].mean(); idx = fs.index.intersection(si.index)
                sr, sp = spearmanr(si.loc[idx].values, fs.loc[idx].values)
                xt[f'{a}|scene_level flow_epe~intrinsic_{y}'] = {'n_scenes': int(len(idx)), 'spearman': float(sr), 'p': float(sp)}
        out['FH3_cross_task'] = xt
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True); Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))

    print(f"flow runs: {out['n_runs']} archs: {archs}; zero-shot: " + ', '.join(f"{a}: EPE {z['epe_pair_mean']:.3f}" for a, z in out['zeroshot'].items()))
    for a in archs:
        Aa = out['archs'][a]; print(f"\n[{a}] runs={Aa['n_runs']} mean {Aa['mean_minutes']:.1f} min")
        for y, m in Aa['metrics'].items():
            c = m['reml'] if m['used'] == 'reml' else m['moment']
            print(f"  {m['label'][:30]:30s} [{m['used']}] scene {c['s2_scene']:.3g} train {c['s2_train']:.3g} inter {c['s2_inter']:.3g} seed {c['s2_seed']:.3g} | test-share {m['share_test_side_of_predicted']:.2f} icc {m['icc_scene_single_seed']:.2f}"
                  + (f" | F-H1 {m['decision_FH1']} death={m['death_icc_lt_0.5']}" if 'decision_FH1' in m else ''))
        if 'FH4_zeroshot_vs_finetuned_scene_difficulty' in Aa:
            h = Aa['FH4_zeroshot_vs_finetuned_scene_difficulty']; print(f"  F-H4 zero-shot vs fine-tuned scene difficulty: Spearman {h['spearman']:.3f} (p={h['p']:.4f}); EPE zero-shot {h['epe_zeroshot_mean']:.3f} -> fine-tuned {h['epe_finetuned_scene_mean']:.3f}")
        print("  MDE abs (EPE): " + ', '.join(f"{k}={v:.3f}" for k, v in Aa['metrics']['epe_mean']['MDE_absolute'].items()))
    for p in out['pairs']:
        print(f"\n[F-H2 {p['pair']} | {p['metric']}] gap {p['mean_gap']:+.4g} s2(scene x method) {p['s2_scene_x_method']:.3g} inter share K1 {p['share_interaction_K1S3']:.2f} MDE K1 {p['MDE_paired']['paired_K1_S3']:.3g} fullcov {p['MDE_paired']['paired_fullcov_5fold_S3']:.3g} gap/MDE_K1 {p['gap_over_MDE_K1']:.2f}")
    for k, v in out.get('FH3_cross_task', {}).items():
        print(f"[F-H3 {k}] " + ', '.join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))


if __name__ == '__main__':
    main()
