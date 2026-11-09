#!/usr/bin/env python3
"""make_fig1.py — Figure 1 of the IJCV manuscript (benchmark resolution). All values read from result JSONs.
(a) Per-scene difficulty on the same 23 Sintel scenes for two tasks: intrinsic (DirectCNN si-MSE, main_v2) and optical flow
    (raft_small EPE, flow_v1). Scenes sorted by intrinsic difficulty; flow difficulty does not follow -> task-specific difficulty.
(b) Resolution ladder: minimum detectable effect (MDE, alpha .05, power .80) for the prevailing recipe (1 partition x 3 seeds),
    5 random partitions, and full-coverage 5-fold x 3 seeds, for intrinsic si-MSE (UNet vs DirectCNN) and flow EPE
    (raft_small vs raft_large), normalised by the actual mean method gap so both tasks share one axis (MDE / |gap|; 1 = resolvable).
Greyscale, serif, value labels (user figure style)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution'); OUT = Path(__file__).parent / 'figs'; OUT.mkdir(exist_ok=True)
m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})

# (a) per-scene difficulty
intr = {r['scene']: r['score'] for r in m2['archs']['direct']['metrics']['simse_R_gray_mean']['scene_table']}
flow = {r['scene']: r['score'] for r in fl['archs']['raft_small']['metrics']['epe_mean']['scene_table']}
scenes = sorted(intr, key=lambda s: -intr[s])
xi = np.array([intr[s] for s in scenes]); xf = np.array([flow[s] for s in scenes])
from scipy.stats import spearmanr
rho, p = spearmanr(xi, xf)

# (b) ladder: MDE / |gap|
pi = next(p for p in m2['pairs'] if p['a'] == 'direct' and p['b'] == 'unet' and p['metric'] == 'simse_R_gray_mean')
pf = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean')
designs = [('1 partition\n3 seeds', 'paired_K1_S3'), ('5 random\npartitions', 'paired_K5_S3'), ('all 23 scenes\n5-fold, 3 seeds', 'paired_fullcov_5fold_S3')]
li = [pi['MDE_paired'][k] / abs(pi['mean_gap']) for _, k in designs]; lf = [pf['MDE_paired'][k] / abs(pf['mean_gap']) for _, k in designs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={'width_ratios': [1.6, 1]})
x = np.arange(len(scenes)); w = 0.42
ax1.bar(x - w / 2, xi / xi.max(), w, color='0.25', label='intrinsic: si-MSE (DirectCNN)')
ax1.bar(x + w / 2, xf / xf.max(), w, color='0.72', edgecolor='0.35', linewidth=0.4, label='optical flow: EPE (RAFT-small)')
ax1.set_xticks(x); ax1.set_xticklabels([s.replace('_', ' ') for s in scenes], fontsize=6, rotation=90)
ax1.set_ylabel('scene difficulty / hardest scene'); ax1.set_ylim(0, 1.08)
ax1.set_title('(a) Scene difficulty, same 23 scenes, two tasks', fontsize=8.5, loc='left')
ax1.legend(frameon=False, fontsize=7, loc='upper right')
ax1.text(0.99, 0.80, f'Spearman $\\rho$ = {rho:.2f} (descriptive)\nsi-MSE {xi.min():.4f}–{xi.max():.4f}; EPE {xf.min():.2f}–{xf.max():.1f} px', transform=ax1.transAxes, fontsize=6.5, va='top', ha='right', color='0.2')

xb = np.arange(3); wb = 0.36
b1 = ax2.bar(xb - wb / 2, li, wb, color='0.25', label='intrinsic (UNet − DirectCNN, si-MSE)')
b2 = ax2.bar(xb + wb / 2, lf, wb, color='0.72', edgecolor='0.35', linewidth=0.4, label='flow (RAFT-small − RAFT-large, EPE)')
ax2.axhline(1.0, color='0.0', linestyle='--', linewidth=0.8); ax2.text(-0.48, 1.07, 'resolvable (MDE = gap)', fontsize=6.3, ha='left', va='bottom')
for bars in (b1, b2):
    for b in bars: ax2.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.04, f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=6.5)
ax2.set_yscale('log'); ax2.set_ylim(0.1, 80); ax2.set_xticks(xb); ax2.set_xticklabels([d for d, _ in designs], fontsize=7)
ax2.set_ylabel('MDE / actual method gap'); ax2.set_title('(b) Resolution ladder', fontsize=8.5, loc='left')
ax2.legend(frameon=False, fontsize=6.3, loc='upper center')
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / 'fig1_resolution.pdf'); fig.savefig(OUT / 'fig1_resolution.png', dpi=220)
print(OUT / 'fig1_resolution.pdf', '| intrinsic ladder', [round(v, 2) for v in li], '| flow ladder', [round(v, 2) for v in lf], f'| rho={rho:.3f} p={p:.3f}')
