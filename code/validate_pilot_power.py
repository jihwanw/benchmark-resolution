#!/usr/bin/env python3
"""validate_pilot_power.py — operating characteristics of the recommended workflow with a *finite pilot* (review round 6, B2).
validate_power.py used the fitted components as if known. Here, for each headline pair, a pilot is simulated on the 75-run design
(5 assignments x 3 seeds; bootstrap_components.simulate with the fitted paired components and fixed per-scene gaps), the paired components
are re-estimated from it with the design-exact moment estimator, the SE of the evaluation design is computed from those estimates, and then
independent evaluation outcomes are simulated under the *true* components: the null rejection rate of |gap_hat| >= z_0.975 SE_hat and the
power at a true gap equal to the true MDE of the design. Averaged over pilots, with the spread across pilots.
Designs: K=1 random partition with 3 seeds; full coverage with R=1,2 repeats and 3 seeds; the searched R=6, S=1 design.
Output: results/benchmark_resolution/pilot_power_validation.json.  Usage: python3 validate_pilot_power.py [--n-pilot 200] [--n-eval 400] [--seed 41]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
from scipy.stats import norm
import analyze_cv2 as A
import bootstrap_components as B
import analyze_flow as F
import validate_power as VP

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
Z975, Z80 = norm.ppf(0.975), norm.ppf(0.8)


def comps_to_pair(st, p):
    """Components estimated from a pilot (bootstrap_components.stats output) in the pair-record form design_se2 expects."""
    return {'s2_scene_x_method': st['s2_scene'], 's2_train_diff': st['s2_train'], 's2_inter_diff': st['s2_inter'], 'v_zeta_diff': st['v_zeta'], 'v_eps_diff': st['v_eps'],
            'mean_gap': p['mean_gap'], 'per_scene_gap': p['per_scene_gap']}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-pilot', type=int, default=200); ap.add_argument('--n-eval', type=int, default=400); ap.add_argument('--seed', type=int, default=41)
    args = ap.parse_args(); rng = np.random.RandomState(args.seed)
    m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']; fdf, _ = F.load(R / 'results_flow_v1')
    pairs = [('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'simse_R_gray_mean'), B.pair_frame(df, 'direct', 'unet', 'simse_R_gray_mean')),
             ('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'R_PSNR_of_mean_mse'), B.pair_frame(df, 'direct', 'unet', 'R_PSNR_of_mean_mse')),
             ('flow', next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'), B.pair_frame(fdf, 'raft_large', 'raft_small', 'epe_mean'))]
    designs = [('K1_S3', 1, 3, False), ('full_R1_S3', 5, 3, True), ('full_R2_S3', 10, 3, True), ('full_R6_S1', 30, 1, True)]
    out = {'n_pilot': args.n_pilot, 'n_eval': args.n_eval, 'seed': args.seed, 'results': []}
    for exp, p, dd in pairs:
        true_c = {'s2_scene': p['s2_scene_x_method'], 's2_train': p['s2_train_diff'], 's2_inter': p['s2_inter_diff'], 's2_seed': B.picked(p)['s2_seed'], 'v_zeta': p['v_zeta_diff'], 'v_eps': p['v_eps_diff']}
        truth = B.stats(dd, 'y', true_c)
        sc = sorted(p['per_scene_gap']); gaps = np.array([p['per_scene_gap'][s] for s in sc])
        if gaps.var(ddof=1) > 0 and p['s2_scene_x_method'] > 0: gaps = gaps.mean() + (gaps - gaps.mean()) * math.sqrt(p['s2_scene_x_method'] / gaps.var(ddof=1))
        # pilots: estimated components
        pilots = []
        for i in range(args.n_pilot):
            sim = B.simulate(dd, 'y', truth, rng); st = B.stats(sim, 'y'); pilots.append(st)
        for name, K, S, fc in designs:
            se_true = math.sqrt(VP.design_se2(p, K, S, fc)); mde_true = (Z975 + Z80) * se_true
            null_rates, powers, se_hats = [], [], []
            for st in pilots:
                se_hat = math.sqrt(max(VP.design_se2(comps_to_pair(st, p), K, S, fc), 1e-30)); se_hats.append(se_hat)
                rej0 = rej1 = 0
                for j in range(args.n_eval):
                    g0, _ = VP.simulate_gap(p, gaps, 0.0, K, S, fc, rng); g1, _ = VP.simulate_gap(p, gaps, mde_true, K, S, fc, rng)
                    rej0 += abs(g0) >= Z975 * se_hat; rej1 += abs(g1) >= Z975 * se_hat
                null_rates.append(rej0 / args.n_eval); powers.append(rej1 / args.n_eval)
            null_rates, powers, se_hats = np.array(null_rates), np.array(powers), np.array(se_hats)
            rec = {'experiment': exp, 'pair': p['pair'], 'metric': p['metric'], 'design': name, 'K': K, 'S': S, 'fullcov': fc, 'se_true': se_true, 'mde_true': mde_true,
                   'se_hat_over_true_median': float(np.median(se_hats / se_true)), 'se_hat_over_true_q05': float(np.percentile(se_hats / se_true, 5)), 'se_hat_over_true_q95': float(np.percentile(se_hats / se_true, 95)),
                   'null_rejection_mean': float(null_rates.mean()), 'null_rejection_q95': float(np.percentile(null_rates, 95)),
                   'power_at_true_mde_mean': float(powers.mean()), 'power_at_true_mde_q05': float(np.percentile(powers, 5)), 'power_at_true_mde_q95': float(np.percentile(powers, 95))}
            out['results'].append(rec)
            print(f"{p['pair']:24s} {p['metric'][:10]:10s} {name:10s} SEhat/SE med {rec['se_hat_over_true_median']:.2f} [{rec['se_hat_over_true_q05']:.2f},{rec['se_hat_over_true_q95']:.2f}] null {rec['null_rejection_mean']:.3f} (q95 {rec['null_rejection_q95']:.3f}) power {rec['power_at_true_mde_mean']:.3f} [{rec['power_at_true_mde_q05']:.2f},{rec['power_at_true_mde_q95']:.2f}]", flush=True)
        (R / 'pilot_power_validation.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', R / 'pilot_power_validation.json')


if __name__ == '__main__':
    main()
