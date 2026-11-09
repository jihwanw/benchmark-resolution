#!/usr/bin/env python3
"""rgb_simse_control.py — what the channel-mean ("grayscale") projection of si-MSE leaves out (series-2 review, round 1, C3).

The evaluator stores two scale-invariant errors per frame (experiment_cv2.py:154): simse_R_gray = _si_mse(gray(R), gray(Rp)) with
gray = mean over the three channels (line 89), and simse_R_rgb = _si_mse(R, Rp) on the RGB tensor with one scalar gain over all
channels (lines 92-95). Per-scene values are means over frames ('simse_R_gray_mean', 'simse_R_rgb_mean'; aggregation at
experiment_cv2.py:183). The RGB form counts chromatic error that the channel mean averages away, so the pair measures what the
grayscale protocol metric does not see. This script fits the variance components (analyze_cv2.moment_components, the paper's
estimator) to both per-scene scores for the four architectures on the random design, gives the test-side share, the ICC, the
scene-level ratio RGB/gray, the Spearman correlation of the per-scene difficulty tables, and the six paired comparisons under RGB
si-MSE (gap/MDE under one partition and under full coverage) beside the grayscale ones already in the paper.
Field names are read from an actual record and asserted. Output: results/benchmark_resolution/rgb_simse_control.json."""
import glob, json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import analyze_cv2 as A
import bootstrap_components as B

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N = 23, 5
GRAY, RGB = 'simse_R_gray_mean', 'simse_R_rgb_mean'


def main():
    rec = json.load(open(sorted(glob.glob(str(R / 'results_cv_v2' / '*' / 'cv2_*_v2.json')))[0]))
    for f in (GRAY, RGB): assert f in next(iter(rec['per_scene'].values())), f
    A.METRICS[RGB] = 'si-MSE (RGB, one gain over all channels)'                 # load_runs reads the keys of A.METRICS
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
    out = {'definitions': 'experiment_cv2.py:89 (gray = channel mean), 92-95 (_si_mse), 154 (both variants), 183 (per-scene mean over frames)', 'archs': {}, 'pairs': {}}
    for a in sorted(df['arch'].unique()):
        d = df[df['arch'] == a]; rec_a = {}
        for y in (GRAY, RGB):
            c = A.moment_components(d, y); v_test = (1 - N / U) * c['s2_scene'] / N
            v_pred = v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 6, False)
            den = c['s2_scene'] + c['s2_inter'] + c['s2_seed']
            rec_a[y] = {k: float(c[k]) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']}
            rec_a[y].update({'test_share': v_test / v_pred, 'icc': c['s2_scene'] / den, 'mean': float(d[y].mean())})
        tg, tr = d.groupby('scene')[GRAY].mean(), d.groupby('scene')[RGB].mean()
        rec_a['rgb_over_gray_mean_ratio'] = float(tr.mean() / tg.mean()); rec_a['spearman_scene_tables'] = float(spearmanr(tg.values, tr.values)[0])
        rec_a['rgb_over_gray_per_scene_min'] = float((tr / tg).min()); rec_a['rgb_over_gray_per_scene_max'] = float((tr / tg).max())
        out['archs'][a] = rec_a
        print(f"{a:12s} gray: share {rec_a[GRAY]['test_share']:.2f} icc {rec_a[GRAY]['icc']:.2f} | rgb: share {rec_a[RGB]['test_share']:.2f} icc {rec_a[RGB]['icc']:.2f} | rgb/gray mean {rec_a['rgb_over_gray_mean_ratio']:.2f} (per scene {rec_a['rgb_over_gray_per_scene_min']:.2f}-{rec_a['rgb_over_gray_per_scene_max']:.2f}) | spearman {rec_a['spearman_scene_tables']:.3f}", flush=True)
    archs = sorted(df['arch'].unique())
    for i, a1 in enumerate(archs):
        for a2 in archs[i + 1:]:
            rec_p = {}
            for y in (GRAY, RGB):
                dd = B.pair_frame(df, a1, a2, y); c = A.moment_components(dd, 'y'); gap = float(dd['y'].mean()); v_test = (1 - N / U) * c['s2_scene'] / N
                k1 = A.Z_MDE * math.sqrt(v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False))
                fu = A.Z_MDE * math.sqrt(A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, True))
                ishare = v_test / (v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False))
                rec_p[y] = {'gap': gap, 'gap_over_mde_K1': abs(gap) / k1, 'gap_over_mde_full': abs(gap) / fu, 'interaction_share_K1': ishare, 's2_scene_x_method': float(c['s2_scene'])}
            out['pairs'][f'{a2} - {a1}'] = rec_p
            print(f"pair {a2}-{a1}: gray K1 {rec_p[GRAY]['gap_over_mde_K1']:.2f} full {rec_p[GRAY]['gap_over_mde_full']:.2f} | rgb K1 {rec_p[RGB]['gap_over_mde_K1']:.2f} full {rec_p[RGB]['gap_over_mde_full']:.2f}", flush=True)
    (R / 'rgb_simse_control.json').write_text(json.dumps(out, indent=1)); print('wrote', R / 'rgb_simse_control.json')


if __name__ == '__main__':
    main()
