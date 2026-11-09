#!/usr/bin/env python3
"""bootstrap_components.py — parametric-bootstrap intervals for the test-side share, the scene ICC, the MDE ladder and the paired
gap/MDE ratios, for every unit-level fit (killing experiment; intrinsic main experiment, random design; flow arm) and for the
headline pairs. Added in review round 1 (B1); revised in round 2 (B1, B2, C1):
  * replicates are simulated from the *reported* components (REML where the paper uses REML, else moments), not from the moment
    estimates, so the interval accompanies the point estimate it is printed next to;
  * the scene difficulties are the observed scene means, centred and rescaled so that their variance equals the fitted s2_scene
    (the raw means carry estimation noise of order (s2_train + s2_inter + s2_seed/S)/R);
  * seed noise is split into a run-level shift zeta_{T,s} shared by the units of a run and an independent part, with
    Var(zeta) + Var(eps) = s2_seed (within-cell) and Var(zeta) + Var(eps)/n = tau2 (run level), so both fitted variances are matched;
  * each replicate is refitted with the design-exact moment estimator (Henderson III) of analyze_cv2.py and the design formulas, so
    the intervals describe the sampling variability of the estimation procedure under the fitted model, conditional on the 23 scenes
    and on the design (R assignments, S seeds).
Why the scenes are not resampled: the estimand is defined on the finite population of the 23 scenes; a non-parametric bootstrap over
scenes duplicates units inside folds and inflates the apparent training-set variance (checked in round 1).
Output: results/benchmark_resolution/bootstrap_ci.json.  Usage: python3 bootstrap_components.py [--n-boot 1000] [--seed 0]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
import analyze_cv2 as A
import analyze_flow as F

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N_TEST, S_PRED, Z = 23, 5, 6, A.Z_MDE
C_T = A.C_T


def design_var(s2_scene, s2_train, s2_inter, s2_run, K, S, fullcov):
    if fullcov: return (C_T * s2_train + s2_inter / U + C_T * s2_run / S) / (K / 5)
    return ((1 - N_TEST / U) * s2_scene / N_TEST + s2_train + s2_inter / N_TEST) / K + s2_run / (K * S)


def stats(d, ycol, comps=None):
    """Quantities of analyze_cv2 from given components (comps) or from the moment fit of d."""
    mom = A.moment_components(d, ycol); c = comps or mom
    s2s, s2t, s2i, s2e = c['s2_scene'], c['s2_train'], c['s2_inter'], c['s2_seed']
    v_test = (1 - N_TEST / U) * s2s / N_TEST; s2_run = A.run_level_seed_var(d, ycol)
    v_pred = v_test + s2t + s2i / N_TEST + s2_run / S_PRED; den = s2s + s2i + s2e
    return {'share': v_test / v_pred if v_pred > 0 else np.nan, 'icc': s2s / den if den > 0 else np.nan,
            's2_scene': s2s, 's2_train': s2t, 's2_inter': s2i, 's2_seed': s2e, 's2_seed_run': s2_run, 'nf': mom['nf'], 'mean': float(d[ycol].mean()),
            'mde_K1_S3': Z * math.sqrt(2 * design_var(s2s, s2t, s2i, s2_run, 1, 3, False)), 'mde_full_S3': Z * math.sqrt(2 * design_var(s2s, s2t, s2i, s2_run, 5, 3, True)),
            'paired_mde_K1_S3': Z * math.sqrt(design_var(s2s, s2t, s2i, s2_run, 1, 3, False)), 'paired_mde_full_S3': Z * math.sqrt(design_var(s2s, s2t, s2i, s2_run, 5, 3, True))}


def simulate(d, ycol, st, rng):
    """One parametric replicate on the actual design (same scene x trainset x seed cells)."""
    b = d[['scene', 'rep', 'fold', 'seed', 'trainset', 'scene_rep', 'n_frames', 'R_MSE_mean']].copy()
    g = d.groupby('scene')[ycol].mean(); g = g - g.mean()
    if g.var(ddof=1) > 0 and st['s2_scene'] > 0: g = g * math.sqrt(st['s2_scene'] / g.var(ddof=1))
    else: g = g * 0.0
    gamma = g + st['mean']
    ts = b['trainset'].unique(); delta = pd.Series(rng.normal(0, math.sqrt(max(st['s2_train'], 0)), len(ts)), index=ts)
    cells = b[['scene', 'trainset']].drop_duplicates(); cells['gd'] = rng.normal(0, math.sqrt(max(st['s2_inter'], 0)), len(cells))
    n = st['nf']; s2e, tau2 = st['s2_seed'], st['s2_seed_run']
    var_zeta = max((tau2 - s2e / n) / (1 - 1 / n), 0.0) if tau2 > s2e / n else 0.0
    var_zeta = min(var_zeta, s2e); var_eps = max(s2e - var_zeta, 0.0)
    runs = b[['trainset', 'seed']].drop_duplicates(); runs['zeta'] = rng.normal(0, math.sqrt(var_zeta), len(runs))
    b = b.merge(cells, on=['scene', 'trainset']).merge(runs, on=['trainset', 'seed'])
    b[ycol] = gamma.loc[b['scene']].values + delta.loc[b['trainset']].values + b['gd'].values + b['zeta'].values + rng.normal(0, math.sqrt(var_eps), len(b))
    return b


def boot(d, ycol, comps, n_boot, rng, keys=('share', 'icc', 's2_scene', 's2_train', 's2_inter', 's2_seed', 'mde_K1_S3', 'mde_full_S3')):
    point = stats(d, ycol, comps); draws = {k: [] for k in keys}
    for _ in range(n_boot):
        st = stats(simulate(d, ycol, point, rng), ycol)
        for k in keys: draws[k].append(st[k])
    out = {'point': {k: float(v) for k, v in point.items()}, 'n_boot': n_boot}
    for k, v in draws.items():
        v = np.array(v, dtype=float); v = v[np.isfinite(v)]
        out[k] = {'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], 'median': float(np.median(v)), 'n_finite': int(len(v))}
    return out


def picked(m): return m['reml'] if m['used'] == 'reml' else m['moment']


def pair_frame(df, a, b, ycol):
    da = df[df['arch'] == a].set_index(['scene', 'rep', 'fold', 'seed']); db = df[df['arch'] == b].set_index(['scene', 'rep', 'fold', 'seed'])
    common = da.index.intersection(db.index)
    dd = (db.loc[common, ycol] - da.loc[common, ycol]).reset_index().rename(columns={ycol: 'y'}).dropna()
    dd['trainset'] = 'r' + dd['rep'].astype(str) + 'f' + dd['fold'].astype(str); dd['scene_rep'] = dd['scene'] + '|r' + dd['rep'].astype(str)
    dd['n_frames'] = 0; dd['R_MSE_mean'] = 0.0
    return dd


def boot_pair(dd, pcomp, n_boot, rng):
    """Paired comparison: components of the difference as reported (s2_scene_x_method etc.); gap/MDE under K=1,S=3 and full coverage."""
    comps = {'s2_scene': pcomp['s2_scene_x_method'], 's2_train': pcomp['s2_train_diff'], 's2_inter': pcomp['s2_inter_diff'], 's2_seed': picked(pcomp)['s2_seed']}
    point = stats(dd, 'y', comps); draws = {'gap': [], 'gap_over_mde_K1': [], 'gap_over_mde_full': [], 'share_interaction_K1S3': []}
    def derived(st):
        tot = (1 - N_TEST / U) * st['s2_scene'] / N_TEST + st['s2_train'] + st['s2_inter'] / N_TEST + st['s2_seed_run'] / 3
        return {'gap': st['mean'], 'gap_over_mde_K1': abs(st['mean']) / st['paired_mde_K1_S3'], 'gap_over_mde_full': abs(st['mean']) / st['paired_mde_full_S3'],
                'share_interaction_K1S3': ((1 - N_TEST / U) * st['s2_scene'] / N_TEST) / tot if tot > 0 else np.nan}
    for _ in range(n_boot):
        st = derived(stats(simulate(dd, 'y', point, rng), 'y'))
        for k in draws: draws[k].append(st[k])
    out = {'point': {**{k: float(v) for k, v in point.items()}, **derived(point)}, 'n_boot': n_boot}
    for k, v in draws.items():
        v = np.array(v, dtype=float); v = v[np.isfinite(v)]
        out[k] = {'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], 'median': float(np.median(v)), 'n_finite': int(len(v))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-boot', type=int, default=1000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(R / 'bootstrap_ci.json')); args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    k1 = json.load(open(R / 'killing_v1.json')); m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    out = {'method': 'parametric bootstrap on the actual design from the reported components; scenes fixed (finite population); exact moment estimator refitted',
           'n_boot': args.n_boot, 'seed': args.seed, 'S_pred': S_PRED, 'killing': {}, 'intrinsic': {}, 'flow': {}, 'pairs': {}}
    def report(tag, arch, ycol, r):
        print(f"{tag:9s} {arch:11s} {ycol:24s} share {r['point']['share']:.3f} [{r['share']['ci95'][0]:.3f}, {r['share']['ci95'][1]:.3f}]  icc {r['point']['icc']:.3f} [{r['icc']['ci95'][0]:.3f}, {r['icc']['ci95'][1]:.3f}]  MDE K1 {r['point']['mde_K1_S3']:.3g} [{r['mde_K1_S3']['ci95'][0]:.3g}, {r['mde_K1_S3']['ci95'][1]:.3g}]", flush=True)
    kd = A.load_v1(R / 'results_cv_v1'); kd['trainset'] = 'r' + kd['rep'].astype(str) + 'f' + kd['fold'].astype(str); kd['scene_rep'] = kd['scene'] + '|r' + kd['rep'].astype(str); kd['n_frames'] = 0; kd['R_MSE_mean'] = 0.0
    for arch in sorted(kd['arch'].unique()):
        d = kd[kd['arch'] == arch]; out['killing'][arch] = {}
        for ycol in ['R_PSNR_of_mean_mse'] + list(A.PRIMARY):
            if ycol not in d or d[ycol].isna().all(): continue
            out['killing'][arch][ycol] = boot(d, ycol, picked(k1['archs'][arch]['metrics'][ycol]), args.n_boot, rng); report('killing', arch, ycol, out['killing'][arch][ycol])
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
    for arch in sorted(df['arch'].unique()):
        d = df[df['arch'] == arch]; out['intrinsic'][arch] = {}
        for ycol in A.METRICS:
            if d[ycol].isna().all(): continue
            out['intrinsic'][arch][ycol] = boot(d, ycol, picked(m2['archs'][arch]['metrics'][ycol]), args.n_boot, rng); report('intrinsic', arch, ycol, out['intrinsic'][arch][ycol])
    fdf, _ = F.load(R / 'results_flow_v1')
    for arch in sorted(fdf['arch'].unique()):
        d = fdf[fdf['arch'] == arch]; out['flow'][arch] = {}
        for ycol in F.METRICS:
            if d[ycol].isna().all(): continue
            out['flow'][arch][ycol] = boot(d, ycol, picked(fl['archs'][arch]['metrics'][ycol]), args.n_boot, rng); report('flow', arch, ycol, out['flow'][arch][ycol])
    # headline pairs: all intrinsic pairs under si-MSE, LMSE and PSNR; the flow pair under EPE
    for p in m2['pairs']:
        if p['metric'] not in ('simse_R_gray_mean', 'lmse_R_chen_mean', 'R_PSNR_of_mean_mse'): continue
        dd = pair_frame(df, p['a'], p['b'], p['metric']); key = f"{p['pair']}|{p['metric']}"
        out['pairs'][key] = boot_pair(dd, p, args.n_boot, rng); r = out['pairs'][key]
        print(f"pair {key:45s} gap/MDE K1 {r['point']['gap_over_mde_K1']:.2f} [{r['gap_over_mde_K1']['ci95'][0]:.2f}, {r['gap_over_mde_K1']['ci95'][1]:.2f}]  full {r['point']['gap_over_mde_full']:.2f} [{r['gap_over_mde_full']['ci95'][0]:.2f}, {r['gap_over_mde_full']['ci95'][1]:.2f}]", flush=True)
    for p in fl['pairs']:
        if p['metric'] != 'epe_mean': continue
        dd = pair_frame(fdf, p['a'], p['b'], p['metric']); key = f"{p['pair']}|{p['metric']}"
        out['pairs'][key] = boot_pair(dd, p, args.n_boot, rng); r = out['pairs'][key]
        print(f"pair {key:45s} gap/MDE K1 {r['point']['gap_over_mde_K1']:.2f} [{r['gap_over_mde_K1']['ci95'][0]:.2f}, {r['gap_over_mde_K1']['ci95'][1]:.2f}]  full {r['point']['gap_over_mde_full']:.2f} [{r['gap_over_mde_full']['ci95'][0]:.2f}, {r['gap_over_mde_full']['ci95'][1]:.2f}]", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float)); print('wrote', args.out)


if __name__ == '__main__':
    main()
