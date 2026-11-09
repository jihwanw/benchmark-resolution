#!/usr/bin/env python3
"""calibrate_diagnostics.py — reference distributions of the two diagnostics by simulation of the complete procedure (review round 4, B1).
1. Full-coverage check. For every unit-level fit, the observed variance of the full-coverage mean across the R = 5 realised assignments is
   divided by the variance eq. fullcov predicts from components fitted to the *same* runs. Under the fitted model with the 23 difficulties
   fixed, n_sim data sets are generated on the actual design (bootstrap_components.simulate), components are refitted (moment estimator),
   the ratio is recomputed, and its empirical 2.5 and 97.5 percentiles are recorded together with the real-data ratio and its two-sided
   empirical p-value. The nominal chi2_4/4 band is kept for comparison.
2. Stratification. For DirectCNN and UNet and each intrinsic metric, the statistic of Sec. 7 (within-repeat variance of fold means in the
   stratified arm divided by that in the random arm) is simulated under the fitted random-arm components with the difficulties fixed, on the
   realised random and stratified assignments (seeds and folds as run); the empirical distribution gives the reference for the observed ratio
   (two-sided empirical p-value) in place of the nominal F(8, 20).
Output: results/benchmark_resolution/calibration_diagnostics.json.  Usage: python3 calibrate_diagnostics.py [--n-sim 300] [--seed 21]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
import analyze_cv2 as A
import analyze_flow as F
import bootstrap_components as B

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U = 23


def fullcov_ratio(sim, ycol, st):
    rep_means = sim.groupby(['rep', 'scene'])[ycol].mean().groupby(level=0).mean()
    pred = A.C_T * st['s2_train'] + st['s2_inter'] / U + A.seed_var_design(st['v_zeta'], st['v_eps'], 5, 3, True)
    return float(rep_means.var(ddof=1) / pred) if pred > 0 else float('nan')


def within_rep_fold_var(d, ycol):
    fm = d.groupby(['rep', 'fold', 'seed'])[ycol].mean().groupby(level=[0, 1]).mean()
    return float(fm.groupby(level=0).var(ddof=1).mean())


def emp_p(samples, obs):
    s = np.asarray(samples); s = s[np.isfinite(s)]
    lo = float((s <= obs).mean()); hi = float((s >= obs).mean())
    return float(min(1.0, 2 * min(lo, hi))), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-sim', type=int, default=300); ap.add_argument('--seed', type=int, default=21); args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    df = A.load_runs(R / 'results_cv_v2'); drand = df[df['design'] == 'rand']; fdf, _ = F.load(R / 'results_flow_v1')
    out = {'n_sim': args.n_sim, 'seed': args.seed, 'fullcov': [], 'stratification': []}
    # 1. full-coverage check, every unit-level fit of the main experiment and the flow arm
    settings = [('intrinsic', a, y, drand[drand['arch'] == a], m2['archs'][a]['metrics'][y]) for a in m2['archs'] for y in m2['archs'][a]['metrics']]
    settings += [('flow', a, y, fdf[fdf['arch'] == a], fl['archs'][a]['metrics'][y]) for a in fl['archs'] for y in fl['archs'][a]['metrics']]
    for exp, arch, ycol, d, m in settings:
        comps = B.picked(m); truth = B.stats(d, ycol, comps); obs = m['fullcov_check']['ratio_obs_pred']
        ratios = []
        for i in range(args.n_sim):
            sim = B.simulate(d, ycol, truth, rng); st = B.stats(sim, ycol); ratios.append(fullcov_ratio(sim, ycol, st))
        p, lo, hi = emp_p(ratios, obs)
        out['fullcov'].append({'experiment': exp, 'arch': arch, 'metric': ycol, 'observed_ratio': obs, 'empirical_ci95': [lo, hi], 'empirical_p_two_sided': p,
                               'inside_empirical_band': bool(lo <= obs <= hi), 'inside_nominal_band': bool(0.1211 <= obs <= 2.7858), 'sim_median': float(np.median(ratios))})
        print(f"fullcov {arch:12s} {ycol[:14]:14s} obs {obs:.2f} emp band [{lo:.2f}, {hi:.2f}] p {p:.2f}", flush=True)
    # 2. stratification statistic for DirectCNN and UNet
    for arch in ['direct', 'unet']:
        dr = df[(df['arch'] == arch) & (df['design'] == 'rand')]; ds = df[(df['arch'] == arch) & (df['design'] == 'strat')]
        if not len(ds): continue
        both = pd.concat([dr, ds], ignore_index=True)
        for ycol in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse']:
            m = m2['archs'][arch]['metrics'][ycol]; comps = B.picked(m); truth = B.stats(dr, ycol, comps)
            obs = within_rep_fold_var(ds, ycol) / within_rep_fold_var(dr, ycol)
            ratios = []
            for i in range(args.n_sim):
                sim = B.simulate(both, ycol, truth, rng)
                sim['design'] = np.where(sim['trainset'].str.startswith('strat'), 'strat', 'rand')   # simulate() keeps trainset labels, not the design column
                ratios.append(within_rep_fold_var(sim[sim['design'] == 'strat'], ycol) / within_rep_fold_var(sim[sim['design'] == 'rand'], ycol))
            p, lo, hi = emp_p(ratios, obs)
            out['stratification'].append({'arch': arch, 'metric': ycol, 'observed_ratio_strat_over_rand': obs, 'empirical_ci95': [lo, hi], 'sim_median': float(np.median(ratios)),
                                          'empirical_p_two_sided': p, 'n_sim': args.n_sim})
            print(f"strat {arch:8s} {ycol[:14]:14s} obs {obs:.2f} sim median {np.median(ratios):.2f} band [{lo:.2f}, {hi:.2f}] p {p:.2f}", flush=True)
    (R / 'calibration_diagnostics.json').write_text(json.dumps(out, indent=1, default=float)); print('wrote', R / 'calibration_diagnostics.json')


if __name__ == '__main__':
    main()
