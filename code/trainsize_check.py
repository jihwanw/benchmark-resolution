#!/usr/bin/env python3
"""trainsize_check.py — does the size of the training set (18 or 19 scenes, i.e. a held-out fold of 5 or 4) shift the score? (review round 4, C4)
Under the five assignments every scene is held out in folds of both sizes for most scenes. For each architecture and metric: for scenes
that appear in both a 5-fold and a 4-fold cell, the mean within-scene difference (score under an 18-scene training set minus score under a
19-scene one), its standard error over scenes, and the difference as a fraction of the between-scene standard deviation. Same for flow.
Output: results/benchmark_resolution/trainsize_check.json"""
import json, math
from pathlib import Path
import numpy as np
import analyze_cv2 as A
import analyze_flow as F

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')


def check(d, ycol):
    cell = d.groupby(['scene', 'trainset'])[ycol].mean().reset_index()
    nT = d.groupby('trainset')['scene'].nunique(); cell['fold_size'] = cell['trainset'].map(nT)
    rows = []
    for sc, g in cell.groupby('scene'):
        a, b = g[g['fold_size'] == 4][ycol], g[g['fold_size'] == 5][ycol]
        if len(a) and len(b): rows.append(float(a.mean() - b.mean()))      # 19-scene training set (fold of 4) minus 18-scene (fold of 5)
    rows = np.array(rows); sd_scene = float(d.groupby('scene')[ycol].mean().std(ddof=1))
    return {'n_scenes_both': int(len(rows)), 'mean_diff_19_minus_18': float(rows.mean()), 'se': float(rows.std(ddof=1) / math.sqrt(len(rows))),
            'diff_over_between_scene_sd': float(rows.mean() / sd_scene) if sd_scene > 0 else None, 'between_scene_sd': sd_scene}


out = {'intrinsic': {}, 'flow': {}}
df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
for a in sorted(df['arch'].unique()):
    out['intrinsic'][a] = {y: check(df[df['arch'] == a], y) for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean']}
fdf, _ = F.load(R / 'results_flow_v1')
for a in sorted(fdf['arch'].unique()):
    out['flow'][a] = {'epe_mean': check(fdf[fdf['arch'] == a], 'epe_mean')}
(R / 'trainsize_check.json').write_text(json.dumps(out, indent=1))
for t in out:
    for a in out[t]:
        for y, r in out[t][a].items(): print(t, a, y[:12], 'n', r['n_scenes_both'], 'diff(19-18)', f"{r['mean_diff_19_minus_18']:+.4g}", 'se', f"{r['se']:.3g}", 'over sd_scene', f"{r['diff_over_between_scene_sd']:+.3f}")
