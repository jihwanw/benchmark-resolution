#!/usr/bin/env python3
"""validate_power.py — operating characteristics of the normal-approximation MDE for the headline paired comparisons (review round 4, B2).
The MDE of eq. mde is z SE with z = z_0.975 + z_0.8 and SE from the fitted paired components (eq. mdepaired), so it promises power 0.8 at
two-sided level 0.05 for the rule |gap_hat| >= z_0.975 SE *when the gap_hat is normal with the stated SE*. Here the finite-population part of a
paired design is not normal (the mean of five of 23 fixed per-scene gaps) and the SE is estimated. For each headline pair and design we
simulate the design's reported gap under the fitted paired model with the per-scene gaps held fixed (the observed per-scene gap table, centred
and rescaled to the fitted scene-by-method variance), training-set, interaction, run-level and independent seed effects drawn from the fitted
components, and record
  * null rejection rate: true mean gap 0, rule |gap_hat| >= z_0.975 SE_model (SE_model from the pilot components, as a user would apply it);
  * power at the MDE: true mean gap = MDE_design, same rule;
  * the same two with SE estimated from the simulated data of the design itself where the design allows it (random partitions K >= 2:
    the empirical SE of the K partition gaps; full coverage with R >= 2: the empirical SE of the R repeat means), with a t critical value.
Designs: K = 1, 3, 5 random partitions with S = 3 seeds; full coverage with R = 1, 2 repeats and S = 3 seeds.
Output: results/benchmark_resolution/power_validation.json.  Usage: python3 validate_power.py [--n-sim 4000] [--seed 31]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
from scipy.stats import norm, t as tdist
import analyze_cv2 as A

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N, FOLDS = 23, 5, [5, 5, 5, 4, 4]
Z975, Z80 = norm.ppf(0.975), norm.ppf(0.8)


def design_se2(p, K, S, fullcov):
    c = {'s2_train': p['s2_train_diff'], 's2_inter': p['s2_inter_diff'], 'vz': p['v_zeta_diff'], 've': p['v_eps_diff'], 's2_dg': p['s2_scene_x_method']}
    if fullcov:
        Rr = K / 5.0
        return (A.C_T * c['s2_train'] + c['s2_inter'] / U) / Rr + A.seed_var_design(c['vz'], c['ve'], K, S, True)
    return ((1 - N / U) * c['s2_dg'] / N + c['s2_train'] + c['s2_inter'] / N) / K + A.seed_var_design(c['vz'], c['ve'], K, S, False)


def simulate_gap(p, gaps, true_mean, K, S, fullcov, rng):
    """Reported mean gap of one realisation of the design, plus the per-partition (or per-repeat) gaps for an empirical SE."""
    g = gaps - gaps.mean() + true_mean                       # fixed per-scene gaps, centred on the true mean
    st, si, vz, ve = math.sqrt(p['s2_train_diff']), math.sqrt(p['s2_inter_diff']), math.sqrt(max(p['v_zeta_diff'], 0)), math.sqrt(max(p['v_eps_diff'], 0))
    parts = []
    if fullcov:
        Rr = int(round(K / 5))
        for r in range(Rr):
            perm = rng.permutation(U); start = 0; tot = 0.0
            for nk in FOLDS:
                idx = perm[start:start + nk]; start += nk
                delta = rng.normal(0, st); inter = rng.normal(0, si, nk)
                seeds = np.array([rng.normal(0, vz) + rng.normal(0, ve, nk).mean() for _ in range(S)]).mean()   # run-level + independent, averaged over S
                tot += (g[idx] + delta + inter).sum() + nk * seeds
            parts.append(tot / U)
    else:
        for k in range(K):
            idx = rng.choice(U, N, replace=False); delta = rng.normal(0, st); inter = rng.normal(0, si, N)
            seeds = np.array([rng.normal(0, vz) + rng.normal(0, ve, N).mean() for _ in range(S)]).mean()
            parts.append((g[idx] + delta + inter).mean() + seeds)
    parts = np.array(parts)
    return float(parts.mean()), parts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-sim', type=int, default=4000); ap.add_argument('--seed', type=int, default=31); args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    pairs = [('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'simse_R_gray_mean')),
             ('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'R_PSNR_of_mean_mse')),
             ('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'lmse_R_chen_mean')),
             ('flow', next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'))]
    designs = [('K1_S3', 1, 3, False), ('K3_S3', 3, 3, False), ('K5_S3', 5, 3, False), ('full_R1_S3', 5, 3, True), ('full_R2_S3', 10, 3, True)]
    out = {'n_sim': args.n_sim, 'seed': args.seed, 'results': []}
    for exp, p in pairs:
        sc = sorted(p['per_scene_gap']); gaps = np.array([p['per_scene_gap'][s] for s in sc])
        # rescale the fixed per-scene gaps to the fitted scene-by-method variance (as the bootstrap does for the difficulties)
        if gaps.var(ddof=1) > 0 and p['s2_scene_x_method'] > 0: gaps = gaps.mean() + (gaps - gaps.mean()) * math.sqrt(p['s2_scene_x_method'] / gaps.var(ddof=1))
        for name, K, S, fc in designs:
            se2 = design_se2(p, K, S, fc); se = math.sqrt(se2); mde = (Z975 + Z80) * se
            rej_null = rej_mde = rej_null_emp = rej_mde_emp = 0; n_emp = 0
            for i in range(args.n_sim):
                g0, parts0 = simulate_gap(p, gaps, 0.0, K, S, fc, rng); g1, parts1 = simulate_gap(p, gaps, mde, K, S, fc, rng)
                rej_null += abs(g0) >= Z975 * se; rej_mde += abs(g1) >= Z975 * se
                if len(parts0) >= 2:
                    n_emp += 1; dfree = len(parts0) - 1; tc = tdist.ppf(0.975, dfree)
                    rej_null_emp += abs(g0) >= tc * parts0.std(ddof=1) / math.sqrt(len(parts0)); rej_mde_emp += abs(g1) >= tc * parts1.std(ddof=1) / math.sqrt(len(parts1))
            rec = {'experiment': exp, 'pair': p['pair'], 'metric': p['metric'], 'design': name, 'K': K, 'S': S, 'fullcov': fc, 'mde': mde, 'se_model': se,
                   'null_rejection_model_se': rej_null / args.n_sim, 'power_at_mde_model_se': rej_mde / args.n_sim,
                   'null_rejection_empirical_se': rej_null_emp / n_emp if n_emp else None, 'power_at_mde_empirical_se': rej_mde_emp / n_emp if n_emp else None,
                   'n_partitions_or_repeats': len(parts0)}
            out['results'].append(rec)
            print(f"{p['pair']:24s} {p['metric'][:12]:12s} {name:10s} MDE {mde:.4g} null {rec['null_rejection_model_se']:.3f} power {rec['power_at_mde_model_se']:.3f}" +
                  (f" | emp-SE null {rec['null_rejection_empirical_se']:.3f} power {rec['power_at_mde_empirical_se']:.3f}" if n_emp else ''), flush=True)
    (R / 'power_validation.json').write_text(json.dumps(out, indent=1, default=float)); print('wrote', R / 'power_validation.json')


if __name__ == '__main__':
    main()
