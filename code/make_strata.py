#!/usr/bin/env python3
"""make_strata.py — difficulty strata for the stratified-partition arm (H6) of the main experiment (v2).

Source: killing experiment v1 per-scene si-MSE (gray, Grosse 2009) of DirectCNN, averaged over the 5 random
training compositions x 2 seeds (results_cv_v1). si-MSE is used (not PSNR) because v1 showed per-scene
difficulty is a stable property only under scale-invariant metrics (ICC 0.95 vs 0.65 for PSNR).
Three strata of sizes 8/8/7 (hard -> easy). Output: strata_v2.json (scene table + stratum + provenance).
"""
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np

res_dir = Path(sys.argv[1]); out = Path(sys.argv[2])
acc = {}
files = sorted(glob.glob(str(res_dir / '**' / 'cv_direct_*_v1.json'), recursive=True))
seen = set()
for fp in files:
    n = Path(fp).name
    if n in seen: continue
    seen.add(n)
    d = json.load(open(fp))
    for sc, ps in d['per_scene'].items():
        acc.setdefault(sc, []).append(ps['simse_R_gray_mean'])
scenes = sorted(acc)
assert len(scenes) == 23, len(scenes)
tab = sorted(((sc, float(np.mean(v)), len(v)) for sc, v in acc.items()), key=lambda t: -t[1])  # hardest first
sizes = [8, 8, 7]
strata = {}
i = 0
for k, sz in enumerate(sizes):
    for sc, _, _ in tab[i:i + sz]:
        strata[sc] = k
    i += sz
h = hashlib.sha256(''.join(sorted(seen)).encode()).hexdigest()[:16]
out.write_text(json.dumps({
    'version': 'v2', 'metric': 'simse_R_gray_mean (DirectCNN, mean over 5 reps x 2 seeds, killing v1)',
    'n_source_runs': len(seen), 'source_run_set_sha256_16': h,
    'strata_sizes_hard_to_easy': sizes,
    'table': [{'scene': sc, 'simse_R_gray': v, 'n_runs': n, 'stratum': strata[sc]} for sc, v, n in tab],
    'stratum_of': strata}, indent=1))
print(out, '| hardest:', tab[0][0], f'{tab[0][1]:.5f}', '| easiest:', tab[-1][0], f'{tab[-1][1]:.5f}')
