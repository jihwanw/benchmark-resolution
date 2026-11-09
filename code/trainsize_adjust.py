#!/usr/bin/env python3
"""trainsize_adjust.py — sensitivity of the components and of the PSNR design comparison to a fixed training-size effect (round 8, A1).
Folds of 4 and 5 units train on 19 and 18 scenes. A systematic size effect beta (19-scene minus 18-scene training) is a function of the
training set, so at the cell level it lies in the span of the training-set indicators: the primary analysis absorbs it into the train-side
component as if it were exchangeable composition noise, and it shifts the full-coverage target (a mixture of the two sizes) relative to the
single-partition target (18-scene training). Here beta is estimated per architecture and metric as the mean within-scene difference between
the two sizes (scenes scored under both; trainsize_check.py), the scores are adjusted to the 18-scene target (y - beta * 1{19-scene}), the
components are refitted with the same estimator, and the MDE ladder and the equal-cost comparison (15 runs: full coverage x 3 seeds vs five
random partitions x 3 seeds) are recomputed. For the paired fits the method-by-size effect (beta_B - beta_A) is estimated and removed from
the differences. Output: results/benchmark_resolution/trainsize_adjust.json"""
import json, math
from pathlib import Path
import numpy as np
import analyze_cv2 as A
import analyze_flow as F
import bootstrap_components as B

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N = 23, 5


def size_effect(d, ycol):
    cell = d.groupby(['scene', 'trainset'])[ycol].mean().reset_index(); nT = d.groupby('trainset')['scene'].nunique(); cell['fold_size'] = cell['trainset'].map(nT)
    diffs = []
    for sc, g in cell.groupby('scene'):
        a, b = g[g['fold_size'] == 4][ycol], g[g['fold_size'] == 5][ycol]
        if len(a) and len(b): diffs.append(float(a.mean() - b.mean()))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(diffs.std(ddof=1) / math.sqrt(len(diffs))), int(len(diffs))


def adjust(d, ycol, beta):
    nT = d.groupby('trainset')['scene'].nunique(); d = d.copy(); d[ycol] = d[ycol] - beta * (d['trainset'].map(nT) == 4).astype(float)
    return d


def ladder(c, vz, ve):
    v_test = (1 - N / U) * c['s2_scene'] / N
    full3 = A.Z_MDE * math.sqrt(2 * (A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(vz, ve, 5, 3, True)))
    rand5 = A.Z_MDE * math.sqrt(2 * ((v_test + c['s2_train'] + c['s2_inter'] / N) / 5 + A.seed_var_design(vz, ve, 5, 3, False)))
    rand1 = A.Z_MDE * math.sqrt(2 * ((v_test + c['s2_train'] + c['s2_inter'] / N) + A.seed_var_design(vz, ve, 1, 3, False)))
    return {'K1_S3': rand1, 'K5_S3': rand5, 'fullcov_S3': full3}


def main():
    m2 = json.load(open(R / 'main_v2.json')); df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
    out = {'archs': {}, 'pairs': {}}
    for a in sorted(df['arch'].unique()):
        out['archs'][a] = {}
        for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean']:
            d = df[df['arch'] == a]; beta, se, n = size_effect(d, y)
            mom0 = A.moment_components(d, y); mom1 = A.moment_components(adjust(d, y, beta), y)
            out['archs'][a][y] = {'beta_19_minus_18': beta, 'beta_se': se, 'n_scenes_both': n,
                                  'primary': {k: mom0[k] for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed', 'v_zeta', 'v_eps']}, 'ladder_primary': ladder(mom0, mom0['v_zeta'], mom0['v_eps']),
                                  'size_adjusted': {k: mom1[k] for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed', 'v_zeta', 'v_eps']}, 'ladder_size_adjusted': ladder(mom1, mom1['v_zeta'], mom1['v_eps']),
                                  'fullcov_target_shift_dB_mixture_minus_18': beta * 8 / 23}
            print(f"{a:12s} {y[:10]:10s} beta {beta:+.3g} (se {se:.3g}) train {mom0['s2_train']:.3g} -> {mom1['s2_train']:.3g} inter {mom0['s2_inter']:.3g} -> {mom1['s2_inter']:.3g} | full3 {out['archs'][a][y]['ladder_primary']['fullcov_S3']:.3g} -> {out['archs'][a][y]['ladder_size_adjusted']['fullcov_S3']:.3g} rand5 {out['archs'][a][y]['ladder_primary']['K5_S3']:.3g} -> {out['archs'][a][y]['ladder_size_adjusted']['K5_S3']:.3g}", flush=True)
    for (a1, a2) in [('direct', 'unet'), ('direct', 'proposed'), ('direct', 'transformer')]:
        for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean']:
            dd = B.pair_frame(df, a1, a2, y); beta, se, n = size_effect(dd, 'y')
            mom0 = A.moment_components(dd, 'y'); mom1 = A.moment_components(adjust(dd, 'y', beta), 'y'); gap = float(dd['y'].mean())
            def pm(c):
                v_test = (1 - N / U) * c['s2_scene'] / N
                return {'paired_mde_K1_S3': A.Z_MDE * math.sqrt(v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False)),
                        'paired_mde_full_S3': A.Z_MDE * math.sqrt(A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, True))}
            p0, p1 = pm(mom0), pm(mom1)
            out['pairs'][f'{a2} - {a1}|{y}'] = {'method_by_size_effect': beta, 'se': se, 'gap': gap, 'primary': {**{k: mom0[k] for k in ['s2_scene', 's2_train', 's2_inter']}, **p0, 'gap_over_mde_full': abs(gap) / p0['paired_mde_full_S3']},
                                                'size_adjusted': {**{k: mom1[k] for k in ['s2_scene', 's2_train', 's2_inter']}, **p1, 'gap_over_mde_full': abs(gap) / p1['paired_mde_full_S3']}}
            print(f"pair {a2}-{a1} {y[:10]} size-effect diff {beta:+.3g} (se {se:.3g}) gap/MDE_full {abs(gap)/p0['paired_mde_full_S3']:.2f} -> {abs(gap)/p1['paired_mde_full_S3']:.2f}", flush=True)
    (R / 'trainsize_adjust.json').write_text(json.dumps(out, indent=1, default=float)); print('wrote', R / 'trainsize_adjust.json')


if __name__ == '__main__':
    main()
