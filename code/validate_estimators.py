#!/usr/bin/env python3
"""validate_estimators.py — recovery and interval-coverage study for the design-exact moment estimator and the parametric bootstrap
(review round 2, B1/B2). For each setting (the reported components of a real fit), n_data synthetic data sets are generated on the
actual design (23 scenes x 5 assignments x 3 seeds, folds 5,5,5,4,4) from known components; each is estimated with the moment
estimator (and, optionally, multi-start REML); for coverage, a parametric bootstrap with n_boot replicates is run inside each data set
and the 95% percentile interval is checked against the known value of the test-side share and of the ICC.
Reported per setting: true components; mean and SD of the estimates; bias of share and ICC; empirical coverage of the 95% intervals.
Output: results/benchmark_resolution/estimator_validation.json
Usage: python3 validate_estimators.py [--n-data 200] [--n-boot 200] [--reml]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import analyze_cv2 as A
import bootstrap_components as B

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N_TEST, S_PRED = 23, 5, 6


def truth_quantities(c, st):
    v_test = (1 - N_TEST / U) * c['s2_scene'] / N_TEST
    share = v_test / (v_test + c['s2_train'] + c['s2_inter'] / N_TEST + A.seed_var_design(st['v_zeta'], st['v_eps'], 1, S_PRED, False))
    icc = c['s2_scene'] / (c['s2_scene'] + c['s2_inter'] + c['s2_seed'])
    return share, icc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-data', type=int, default=200); ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--reml', action='store_true'); ap.add_argument('--seed', type=int, default=1); args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
    import analyze_flow as F; fdf, _ = F.load(R / 'results_flow_v1')
    settings = [('intrinsic', 'direct', 'R_PSNR_of_mean_mse', df[df['arch'] == 'direct'], m2['archs']['direct']['metrics']['R_PSNR_of_mean_mse']),
                ('intrinsic', 'direct', 'simse_R_gray_mean', df[df['arch'] == 'direct'], m2['archs']['direct']['metrics']['simse_R_gray_mean']),
                ('intrinsic', 'unet', 'R_PSNR_of_mean_mse', df[df['arch'] == 'unet'], m2['archs']['unet']['metrics']['R_PSNR_of_mean_mse']),
                ('flow', 'raft_small', 'epe_mean', fdf[fdf['arch'] == 'raft_small'], fl['archs']['raft_small']['metrics']['epe_mean'])]
    out = {'n_data': args.n_data, 'n_boot': args.n_boot, 'seed': args.seed, 'settings': []}
    for exp, arch, ycol, d, m in settings:
        comps = B.picked(m); truth = B.stats(d, ycol, comps)   # truth = reported components + the data's run-level seed variance
        share_t, icc_t = truth_quantities(comps, truth)
        est = {k: [] for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed', 'share', 'icc']}; cover = {'share': 0, 'icc': 0}; reml_est = {k: [] for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']}
        for i in range(args.n_data):
            sim = B.simulate(d, ycol, truth, rng)
            st = B.stats(sim, ycol)
            for k in est: est[k].append(st[k])
            if args.reml:
                r = A.reml_components(sim, ycol)
                if r and 'error' not in r:
                    for k in reml_est: reml_est[k].append(r[k])
            if args.n_boot > 0:
                bs = B.boot(sim, ycol, None, args.n_boot, rng, keys=('share', 'icc'))
                lo, hi = bs['share']['ci95']; cover['share'] += int(lo <= share_t <= hi)
                lo, hi = bs['icc']['ci95']; cover['icc'] += int(lo <= icc_t <= hi)
            if (i + 1) % 25 == 0: print(f"{exp} {arch} {ycol}: {i+1}/{args.n_data}", flush=True)
        rec = {'experiment': exp, 'arch': arch, 'metric': ycol, 'truth': {**{k: float(comps[k]) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']}, 's2_seed_run': truth['s2_seed_run'], 'share': share_t, 'icc': icc_t},
               'moment': {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))} for k, v in est.items()},
               'coverage95': {k: cover[k] / args.n_data for k in cover} if args.n_boot > 0 else None}
        if args.reml and reml_est['s2_scene']:
            rec['reml'] = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)} for k, v in reml_est.items()}
        out['settings'].append(rec)
        print(json.dumps(rec, indent=0)[:600], flush=True)
        (R / 'estimator_validation.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', R / 'estimator_validation.json')


if __name__ == '__main__':
    main()
