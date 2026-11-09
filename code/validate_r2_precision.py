#!/usr/bin/env python3
"""validate_r2_precision.py — how precisely does the 20-run design (two assignments, two seeds) estimate a pair's components? (round 7, C1)
Under the fitted paired model (per-scene gaps fixed, as in the bootstrap), simulate the 75-run design n_sim times; estimate the paired
components (a) from all five assignments and three seeds and (b) from the first two assignments and seeds 42, 43 (the 20-run design);
report the relative standard deviation (SD / true value) of the scene-by-method component, the interaction component, and the full-coverage
paired MDE under each. Pairs: UNet-DirectCNN si-MSE and PSNR; RAFT-small-RAFT-large EPE.
Output: results/benchmark_resolution/r2_precision.json.  Usage: python3 validate_r2_precision.py [--n-sim 300] [--seed 51]"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import analyze_cv2 as A
import bootstrap_components as B
import analyze_flow as F

warnings.filterwarnings('ignore')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-sim', type=int, default=300); ap.add_argument('--seed', type=int, default=51); args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']; fdf, _ = F.load(R / 'results_flow_v1')
    pairs = [('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'simse_R_gray_mean'), B.pair_frame(df, 'direct', 'unet', 'simse_R_gray_mean')),
             ('intrinsic', next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'R_PSNR_of_mean_mse'), B.pair_frame(df, 'direct', 'unet', 'R_PSNR_of_mean_mse')),
             ('flow', next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'), B.pair_frame(fdf, 'raft_large', 'raft_small', 'epe_mean'))]
    out = {'n_sim': args.n_sim, 'seed': args.seed, 'results': []}
    for exp, p, dd in pairs:
        true_c = {'s2_scene': p['s2_scene_x_method'], 's2_train': p['s2_train_diff'], 's2_inter': p['s2_inter_diff'], 's2_seed': B.picked(p)['s2_seed'], 'v_zeta': p['v_zeta_diff'], 'v_eps': p['v_eps_diff']}
        truth = B.stats(dd, 'y', true_c)
        reps = sorted(dd['rep'].unique())[:2]; seeds = sorted(dd['seed'].unique())[:2]
        est = {'full': {'s2_scene': [], 's2_inter': [], 'mde_full': []}, 'r2s2': {'s2_scene': [], 's2_inter': [], 'mde_full': []}}
        for i in range(args.n_sim):
            sim = B.simulate(dd, 'y', truth, rng)
            st5 = B.stats(sim, 'y'); sub = sim[sim['rep'].isin(reps) & sim['seed'].isin(seeds)]; st2 = B.stats(sub, 'y')
            for k, st in [('full', st5), ('r2s2', st2)]:
                est[k]['s2_scene'].append(st['s2_scene']); est[k]['s2_inter'].append(st['s2_inter']); est[k]['mde_full'].append(st['paired_mde_full_S3'])
        rec = {'experiment': exp, 'pair': p['pair'], 'metric': p['metric'], 'truth': {'s2_scene': true_c['s2_scene'], 's2_inter': true_c['s2_inter'], 'mde_full_S3': truth['paired_mde_full_S3']}, 'designs': {}}
        for k in est:
            rec['designs'][k] = {q: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'rel_sd': float(np.std(v, ddof=1) / (abs(np.mean(v)) if np.mean(v) != 0 else 1.0)),
                                     'rel_sd_vs_truth': float(np.std(v, ddof=1) / rec['truth'][{'s2_scene': 's2_scene', 's2_inter': 's2_inter', 'mde_full': 'mde_full_S3'}[q]]) if rec['truth'][{'s2_scene': 's2_scene', 's2_inter': 's2_inter', 'mde_full': 'mde_full_S3'}[q]] > 0 else None,
                                     'frac_zero': float(np.mean(np.array(v) == 0))} for q, v in est[k].items()}
        rec['designs']['r2s2']['n_runs'] = 5 * len(reps) * len(seeds); rec['designs']['full']['n_runs'] = 75
        out['results'].append(rec)
        for k in est: print(f"{p['pair']:24s} {p['metric'][:10]:10s} {k:5s} scene x method rel SD {rec['designs'][k]['s2_scene']['rel_sd_vs_truth']:.2f} inter rel SD {rec['designs'][k]['s2_inter']['rel_sd_vs_truth']:.2f} MDE_full rel SD {rec['designs'][k]['mde_full']['rel_sd_vs_truth']:.2f} zero-trunc {rec['designs'][k]['s2_inter']['frac_zero']:.2f}", flush=True)
    (R / 'r2_precision.json').write_text(json.dumps(out, indent=1, default=float)); print('wrote', R / 'r2_precision.json')


if __name__ == '__main__':
    main()
