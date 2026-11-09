#!/usr/bin/env python3
"""compare_q3_forms.py — conditional versus unconditional third moment equation (review round 2, B2 follow-up, 2026-11-06).
The design-exact moment estimator of analyze_cv2.moment_components uses, as its third quadratic form, the variance across training
sets of the training-set means of the *unit-centred* cell means (M_g Q3* M_g), which annihilates the unit effects. An earlier form used
the plain training-set-mean variance Q3*, whose expectation contains the unit effects through sigma2_scene tr(Q3* Zg Zg'), i.e. their
contribution under random unit effects. With 23 fixed scenes and five realised assignments the actual contribution gamma' Q3* gamma
differs from that expectation and the train-side component absorbs the difference.
This script simulates n_data data sets from the reported components of one fit on the actual design with the difficulties held fixed
(bootstrap_components.simulate) and records the mean of both estimators (raw and truncated), plus REML when requested.
Output: results/benchmark_resolution/estimator_q3_comparison.json
Usage: python3 compare_q3_forms.py [--n-data 300] [--seed 11] [--reml]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import analyze_cv2 as A
import bootstrap_components as B

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')


def moment_unconditional(df, ycol):
    """The round-2a form: Q3 = plain variance of training-set means (expectation taken over random unit effects)."""
    d = df.dropna(subset=[ycol]); g = d.groupby(['scene', 'trainset'])[ycol]
    S = float(d.groupby(['scene', 'trainset'])['seed'].nunique().mean()); s2_seed = float(g.var(ddof=1).mean())
    m = g.mean().reset_index(); y = m[ycol].values.astype(float); N = len(y)
    Zg, Zt = A._onehot(m['scene'].tolist()), A._onehot(m['trainset'].tolist())
    def vg(Z):
        Am = (Z / Z.sum(0)).T; C = np.eye(Z.shape[1]) - 1.0 / Z.shape[1]; return Am.T @ C @ Am / (Z.shape[1] - 1)
    Q1, Q3 = vg(Zg), vg(Zt)
    X = np.hstack([np.ones((N, 1)), Zg, Zt]); P = X @ np.linalg.pinv(X); rank = int(round(np.trace(P))); Q2 = (np.eye(N) - P) / max(1, N - rank)
    G, Tt = Zg @ Zg.T, Zt @ Zt.T
    Cm = np.array([[np.trace(Q @ G), np.trace(Q @ Tt), np.trace(Q)] for Q in (Q1, Q2, Q3)]); b = np.array([y @ Q1 @ y, y @ Q2 @ y, y @ Q3 @ y])
    sol = np.linalg.solve(Cm, b)
    return {'raw': [float(v) for v in sol], 's2_scene': max(0.0, float(sol[0])), 's2_train': max(0.0, float(sol[1])), 's2_inter': max(0.0, float(sol[2]) - s2_seed / S)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-data', type=int, default=300); ap.add_argument('--seed', type=int, default=11); ap.add_argument('--reml', action='store_true')
    args = ap.parse_args(); rng = np.random.RandomState(args.seed)
    m2 = json.load(open(R / 'main_v2.json')); df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
    out = {'n_data': args.n_data, 'seed': args.seed, 'settings': []}
    for arch, ycol in [('direct', 'R_PSNR_of_mean_mse'), ('unet', 'R_PSNR_of_mean_mse'), ('direct', 'simse_R_gray_mean')]:
        d = df[df['arch'] == arch]; m = m2['archs'][arch]['metrics'][ycol]; comps = B.picked(m); truth = B.stats(d, ycol, comps)
        real_c = A.moment_components(d, ycol); real_u = moment_unconditional(d, ycol); real_r = m['reml']
        cond = {k: [] for k in ['s2_scene', 's2_train', 's2_inter']}; unc = {k: [] for k in cond}; cond_raw, unc_raw = [], []; reml = {k: [] for k in cond}
        for i in range(args.n_data):
            sim = B.simulate(d, ycol, truth, rng)
            mc = A.moment_components(sim, ycol); mu = moment_unconditional(sim, ycol)
            for k in cond: cond[k].append(mc[k]); unc[k].append(mu[k])
            cond_raw.append(mc['raw_solution'][1]); unc_raw.append(mu['raw'][1])
            if args.reml:
                r = A.reml_components(sim, ycol)
                if r and 'error' not in r:
                    for k in reml: reml[k].append(r[k])
        n = args.n_data
        rec = {'arch': arch, 'metric': ycol, 'truth': {k: float(comps[k]) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']},
               'real_data': {'conditional': {k: real_c[k] for k in ['s2_scene', 's2_train', 's2_inter']}, 'unconditional': {k: real_u[k] for k in ['s2_scene', 's2_train', 's2_inter']},
                             'reml': {k: real_r[k] for k in ['s2_scene', 's2_train', 's2_inter']} if real_r and 'error' not in real_r else None},
               'sim_conditional': {k: {'mean': float(np.mean(v)), 'se': float(np.std(v, ddof=1) / math.sqrt(n))} for k, v in cond.items()},
               'sim_unconditional': {k: {'mean': float(np.mean(v)), 'se': float(np.std(v, ddof=1) / math.sqrt(n))} for k, v in unc.items()},
               'sim_conditional_train_raw_mean': float(np.mean(cond_raw)), 'sim_unconditional_train_raw_mean': float(np.mean(unc_raw))}
        if args.reml and reml['s2_train']: rec['sim_reml'] = {k: {'mean': float(np.mean(v)), 'se': float(np.std(v, ddof=1) / math.sqrt(len(v))), 'n': len(v)} for k, v in reml.items()}
        out['settings'].append(rec); print(json.dumps(rec, indent=0)[:900], flush=True)
        (R / 'estimator_q3_comparison.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', R / 'estimator_q3_comparison.json')


if __name__ == '__main__':
    main()
