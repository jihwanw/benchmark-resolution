#!/usr/bin/env python3
"""qual_display_check.py — are the display models of Figs. 6/7 typical? (review round 3, C5)
For the fold used in the qualitative figures (qual_intrinsic_meta.json: rep, fold, scenes shown), compare the display run's scene PSNR
(seed 42, retrained for display) with the main experiment's three seeds on the same (rep, fold) cell, per architecture and scene; same
for the flow figure (qual_flow_meta.json) with scene EPE. Writes results/benchmark_resolution/qual/display_check.json."""
import json
from pathlib import Path
import analyze_cv2 as A
import analyze_flow as F

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
out = {}
mi = json.load(open(R / 'qual/qual_intrinsic_meta.json'))['jobs']['inter']
df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
sub = df[(df['rep'] == mi['rep']) & (df['fold'] == mi['fold'])]
assert set(sub['scene'].unique()) == set(mi['test_scenes']), 'fold mismatch'
out['intrinsic'] = {'rep': mi['rep'], 'fold': mi['fold'], 'scenes': {}}
for s in mi['scenes_shown']:
    out['intrinsic']['scenes'][s] = {}
    for a in ['direct', 'unet']:
        v = sub[(sub['arch'] == a) & (sub['scene'] == s)].sort_values('seed')
        vals = [float(x) for x in v['R_PSNR_of_mean_mse']]
        out['intrinsic']['scenes'][s][a] = {'display_psnr': mi['per_scene_psnr'][f'{a}/{s}'], 'main_seeds': [int(x) for x in v['seed']], 'main_psnr': vals,
                                            'main_min': min(vals), 'main_max': max(vals), 'main_mean': sum(vals) / len(vals),
                                            'display_within_seed_range': bool(min(vals) <= mi['per_scene_psnr'][f'{a}/{s}'] <= max(vals))}
    g = out['intrinsic']['scenes'][s]
    g['gap_unet_minus_direct_display'] = g['unet']['display_psnr'] - g['direct']['display_psnr']
    g['gap_unet_minus_direct_main_cell_mean'] = g['unet']['main_mean'] - g['direct']['main_mean']
# main-experiment per-scene gap (all cells) for the shown scenes
m2 = json.load(open(R / 'main_v2.json')); p = next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'R_PSNR_of_mean_mse')
for s in mi['scenes_shown']: out['intrinsic']['scenes'][s]['gap_unet_minus_direct_main_all_cells'] = p['per_scene_gap'][s]
# flow
mf = json.load(open(R / 'qual/qual_flow_meta.json'))
fdf, _ = F.load(R / 'results_flow_v1')
fsub = fdf[(fdf['rep'] == mf['rep']) & (fdf['fold'] == mf['fold'])]
assert set(fsub['scene'].unique()) == set(mf['test_scenes']), 'flow fold mismatch'
out['flow'] = {'rep': mf['rep'], 'fold': mf['fold'], 'scenes': {}}
for s in mf['scenes_shown']:
    out['flow']['scenes'][s] = {}
    for a in ['raft_small', 'raft_large']:
        v = fsub[(fsub['arch'] == a) & (fsub['scene'] == s)].sort_values('seed'); vals = [float(x) for x in v['epe_mean']]
        disp = mf['models'][a]['scene_epe_finetuned'][s]
        out['flow']['scenes'][s][a] = {'display_epe': disp, 'main_seeds': [int(x) for x in v['seed']], 'main_epe': vals, 'main_min': min(vals), 'main_max': max(vals), 'main_mean': sum(vals) / len(vals),
                                       'display_within_seed_range': bool(min(vals) <= disp <= max(vals))}
# gauge figure (round 5, A1): bamboo_2 under the five compositions in which it is held out, DirectCNN; main-experiment seeds vs display run
mg = json.load(open(R / 'qual/qual_intrinsic_meta.json'))['jobs']['gauge']
import glob
recs = {}
for fp in glob.glob(str(R / 'results_cv_v2' / '**' / f"cv2_{mg['arch']}_rand*_v2.json"), recursive=True):
    r = json.load(open(fp)); ps = r['per_scene'].get(mg['scene'])
    if ps is None: continue
    gain = ps['gain_R_rgb_xy_sum'] / ps['gain_R_rgb_xx_sum'] if ps.get('gain_R_rgb_xx_sum') else None
    recs.setdefault(int(r['rep']), []).append({'seed': int(r['seed']), 'scene_psnr': ps['R_PSNR_of_mean_mse'], 'gain': gain})
out['gauge'] = {'scene': mg['scene'], 'arch': mg['arch'], 'compositions': []}
for c in sorted(mg['compositions'], key=lambda c: c['rep']):
    main = sorted(recs.get(c['rep'], []), key=lambda x: x['seed'])
    out['gauge']['compositions'].append({'rep': c['rep'], 'bamboo_1_in_train': c['bamboo_1_in_train'], 'display_psnr': c['scene_psnr'], 'display_gain': c['optimal_gain'],
                                         'main_psnr': [x['scene_psnr'] for x in main], 'main_gain': [x['gain'] for x in main], 'main_seeds': [x['seed'] for x in main],
                                         'display_psnr_within': bool(main and min(x['scene_psnr'] for x in main) <= c['scene_psnr'] <= max(x['scene_psnr'] for x in main)),
                                         'display_gain_within': bool(main and min(x['gain'] for x in main) <= c['optimal_gain'] <= max(x['gain'] for x in main))})
(R / 'qual/display_check.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
