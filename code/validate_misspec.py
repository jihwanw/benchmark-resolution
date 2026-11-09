#!/usr/bin/env python3
"""validate_misspec.py — misspecification study for the design-exact moment estimator and its parametric-bootstrap intervals
(review round 3, B1). The in-model study (validate_estimators.py) generates data from the fitted model with the estimator's own
assumptions. Here data are generated from the fitted DirectCNN PSNR and si-MSE components (and RAFT-small EPE) on the actual design
under three departures, and the *model-based* bootstrap interval (which does not know about the departure) is checked for coverage:
  hetero   seed variance proportional to scene difficulty: Var(eps_u) = s2_seed * w_u with w_u = U * exp(c z_u) / sum exp(c z_u),
           z_u the standardised scene difficulty, c chosen so that the hardest scene has 4x the seed variance of the easiest
           (the mean over scenes equals s2_seed, so the pooled estimate is unchanged in expectation);
  corr     interaction cells of the units of one fold correlated at rho_eta = 0.5 (a shared fold component plus an independent part,
           both scaled so that the marginal variance is s2_inter); the model assumes 0;
  heavy    seed noise and run shift drawn from a Student t with 3 degrees of freedom scaled to the fitted variances;
  assign   an assignment-level effect nu_r (variance v = s2_train, i.e. rho_delta = 0.5 between the training-set effects of the five
           folds of one assignment) added to every cell of assignment r; the model has no such term.
Truth per scenario is defined on the generating process, not on the labels of the fitted model:
  share  = V_test / Var(single random 5-of-23 partition mean, six seeds) under the generating process. Under 'corr' the n units of a
           fold share rho s2_inter, so the interaction contributes s2_inter/n + rho s2_inter (n-1)/n; under 'assign' the training-set
           effect delta_T = delta + nu_r has marginal variance 2 s2_train (a single partition sees the sum), which enters the share.
  icc    = s2_scene / (s2_scene + s2_inter_within + s2_seed) where s2_inter_within is the part of the interaction that varies within
           a run: (1-rho) s2_inter under 'corr' (the fold-shared part is a run-level quantity, indistinguishable from a training-set
           effect at the cell level, and is absorbed by s2_train by both estimators), s2_inter otherwise.
Also recorded: the full-coverage variance (eq. fullcov, S=3) predicted from the mean fitted components versus the true variance of the
full-coverage mean under the generating process ('assign': + v, not reduced by averaging over folds), as ratio true/predicted.
Output: results/benchmark_resolution/estimator_misspec.json (merged across invocations by scenario)
Usage: python3 validate_misspec.py [--n-data 200] [--n-boot 200] [--seed 3] [--scenarios hetero,corr,heavy,assign]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
import analyze_cv2 as A
import bootstrap_components as B
import analyze_flow as F

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N_TEST, S_PRED = 23, 5, 6


def simulate_misspec(d, ycol, st, rng, scenario):
    b = d[['scene', 'rep', 'fold', 'seed', 'trainset', 'scene_rep', 'n_frames', 'R_MSE_mean']].copy()
    g = d.groupby('scene')[ycol].mean(); g = g - g.mean()
    if g.var(ddof=1) > 0 and st['s2_scene'] > 0: g = g * math.sqrt(st['s2_scene'] / g.var(ddof=1))
    else: g = g * 0.0
    gamma = g + st['mean']
    ts = b['trainset'].unique(); delta = pd.Series(rng.normal(0, math.sqrt(max(st['s2_train'], 0)), len(ts)), index=ts)
    cells = b[['scene', 'trainset']].drop_duplicates().copy()
    si = math.sqrt(max(st['s2_inter'], 0))
    if scenario == 'corr':   # shared fold component (same trainset) + independent part; marginal variance s2_inter, within-fold corr 0.5
        rho = 0.5; shared = pd.Series(rng.normal(0, si * math.sqrt(rho), len(ts)), index=ts)
        cells['gd'] = shared.loc[cells['trainset']].values + rng.normal(0, si * math.sqrt(1 - rho), len(cells))
    else:
        cells['gd'] = rng.normal(0, si, len(cells))
    var_zeta, var_eps = max(st['v_zeta'], 0.0), max(st['v_eps'], 0.0)     # fitted run-level decomposition (round 4, A1)
    if scenario == 'assign':   # assignment-level shift shared by the five training sets of one assignment: rho_delta = 0.5
        reps = b['rep'].unique(); nu = pd.Series(rng.normal(0, math.sqrt(max(st['s2_train'], 0)), len(reps)), index=reps)
        delta = delta + pd.Series({t: nu.loc[b.loc[b['trainset'] == t, 'rep'].iloc[0]] for t in ts})
    runs = b[['trainset', 'seed']].drop_duplicates().copy()
    def noise(size, var):
        if scenario == 'heavy':   # t_3 scaled to the given variance (Var t_3 = 3)
            return rng.standard_t(3, size) * math.sqrt(var / 3.0)
        return rng.normal(0, math.sqrt(var), size)
    runs['zeta'] = noise(len(runs), var_zeta)
    b = b.merge(cells, on=['scene', 'trainset']).merge(runs, on=['trainset', 'seed'])
    eps = noise(len(b), var_eps)
    if scenario == 'hetero':   # difficulty-dependent seed variance, mean weight 1 over scenes
        z = (g - g.mean()) / (g.std(ddof=1) if g.std(ddof=1) > 0 else 1.0)
        # hardest/easiest ratio 4: exp(c (zmax - zmin)) = 4
        c = math.log(4.0) / max(float(z.max() - z.min()), 1e-9); w = np.exp(c * z); w = w / w.mean()
        # PSNR: hard = low score; the sign of z does not matter for the test, we make the harder (lower-PSNR / higher-error) scene noisier
        hard_is_low = ycol.startswith('R_PSNR')
        if hard_is_low: w = np.exp(-c * z); w = w / w.mean()
        eps = eps * np.sqrt(w.loc[b['scene']].values)
    b[ycol] = gamma.loc[b['scene']].values + delta.loc[b['trainset']].values + b['gd'].values + b['zeta'].values + eps
    return b


RHO_CORR = 0.5


def truth_quantities(c, st, scenario='none'):
    """st: the generating 'truth' dict of bootstrap_components.stats (carries v_zeta, v_eps)."""
    v_test = (1 - N_TEST / U) * c['s2_scene'] / N_TEST
    inter_part = c['s2_inter'] / N_TEST + (RHO_CORR * c['s2_inter'] * (N_TEST - 1) / N_TEST if scenario == 'corr' else 0.0)
    train_part = c['s2_train'] * (2.0 if scenario == 'assign' else 1.0)   # 'assign': delta_T = delta + nu_r with Var(nu) = s2_train -> marginal 2 s2_train
    share = v_test / (v_test + train_part + inter_part + A.seed_var_design(st['v_zeta'], st['v_eps'], 1, S_PRED, False))
    inter_within = (1 - RHO_CORR) * c['s2_inter'] if scenario == 'corr' else c['s2_inter']
    icc = c['s2_scene'] / (c['s2_scene'] + inter_within + c['s2_seed'])
    return share, icc


def true_fullcov_var(c, st, scenario, S=3):
    """Variance of the full-coverage mean (one assignment, S seeds) under the generating process."""
    base = B.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(st['v_zeta'], st['v_eps'], 5, S, True)
    if scenario == 'corr': base += (B.C_T - 1 / U) * RHO_CORR * c['s2_inter']
    if scenario == 'assign': base += c['s2_train']          # nu_r has variance v = s2_train and is common to the whole assignment
    return base


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-data', type=int, default=200); ap.add_argument('--n-boot', type=int, default=200); ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--scenarios', default='hetero,corr,heavy,assign')
    args = ap.parse_args(); rng = np.random.RandomState(args.seed); scenarios = args.scenarios.split(',')
    OUT = R / 'estimator_misspec.json'; prev = json.load(open(OUT)) if OUT.exists() else None
    m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']; fdf, _ = F.load(R / 'results_flow_v1')
    settings = [('intrinsic', 'direct', 'R_PSNR_of_mean_mse', df[df['arch'] == 'direct'], m2['archs']['direct']['metrics']['R_PSNR_of_mean_mse']),
                ('intrinsic', 'direct', 'simse_R_gray_mean', df[df['arch'] == 'direct'], m2['archs']['direct']['metrics']['simse_R_gray_mean']),
                ('flow', 'raft_small', 'epe_mean', fdf[fdf['arch'] == 'raft_small'], fl['archs']['raft_small']['metrics']['epe_mean'])]
    out = {'n_data': args.n_data, 'n_boot': args.n_boot, 'seed': args.seed, 'scenarios': scenarios, 'rho_corr': RHO_CORR, 'settings': []}
    for exp, arch, ycol, d, m in settings:
        comps = B.picked(m); truth = B.stats(d, ycol, comps)
        rec = {'experiment': exp, 'arch': arch, 'metric': ycol, 'truth': {k: float(comps[k]) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']}, 'scenarios': {}}
        # achieved total seed-variance ratio hardest/easiest under 'hetero' (B3): only the independent part is scaled, (v_zeta + v_eps w_max)/(v_zeta + v_eps w_min)
        gsc = d.groupby('scene')[ycol].mean(); gsc = gsc - gsc.mean(); zz = (gsc - gsc.mean()) / (gsc.std(ddof=1) if gsc.std(ddof=1) > 0 else 1.0)
        cc = math.log(4.0) / max(float(zz.max() - zz.min()), 1e-9); ww = np.exp(cc * zz); ww = ww / ww.mean()
        rec['hetero_total_ratio'] = float((truth['v_zeta'] + truth['v_eps'] * ww.max()) / (truth['v_zeta'] + truth['v_eps'] * ww.min())); rec['hetero_independent_ratio'] = 4.0
        if prev:   # keep scenarios computed in earlier invocations
            old_rec = next((r for r in prev['settings'] if r['arch'] == arch and r['metric'] == ycol), None)
            if old_rec: rec['scenarios'] = {k: v for k, v in old_rec['scenarios'].items() if k not in scenarios}
        for sc in scenarios:
            share_t, icc_t = truth_quantities(comps, truth, sc)
            est = {k: [] for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed', 'share', 'icc']}; cover = {'share': 0, 'icc': 0}; width = {'share': [], 'icc': []}
            for i in range(args.n_data):
                sim = simulate_misspec(d, ycol, truth, rng, sc); st = B.stats(sim, ycol)
                for k in est: est[k].append(st[k])
                if args.n_boot > 0:
                    bs = B.boot(sim, ycol, None, args.n_boot, rng, keys=('share', 'icc'))
                    for k in cover:
                        lo, hi = bs[k]['ci95']; cover[k] += int(lo <= (share_t if k == 'share' else icc_t) <= hi); width[k].append(hi - lo)
                if (i + 1) % 50 == 0: print(f"{arch} {ycol} {sc}: {i+1}/{args.n_data}", flush=True)
            mean_c = {k: float(np.mean(est[k])) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']}
            pred_fc = B.C_T * mean_c['s2_train'] + mean_c['s2_inter'] / U + A.seed_var_design(truth['v_zeta'], truth['v_eps'], 5, 3, True)
            rec['scenarios'][sc] = {'truth': {'share': share_t, 'icc': icc_t}, 'moment': {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))} for k, v in est.items()},
                                    'bias': {'share': float(np.mean(est['share']) - share_t), 'icc': float(np.mean(est['icc']) - icc_t)},
                                    'fullcov_var_true_over_predicted': float(true_fullcov_var(comps, truth, sc) / pred_fc) if pred_fc > 0 else None,
                                    'coverage95': {k: cover[k] / args.n_data for k in cover} if args.n_boot > 0 else None,
                                    'mean_width': {k: float(np.mean(v)) for k, v in width.items()} if args.n_boot > 0 else None}
            print(json.dumps({sc: rec['scenarios'][sc]['bias'], 'cov': rec['scenarios'][sc]['coverage95']}), flush=True)
        out['settings'].append(rec); out['scenarios'] = sorted(set(scenarios) | set(k for r in out['settings'] for k in r['scenarios'])); OUT.write_text(json.dumps(out, indent=1, default=float))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
