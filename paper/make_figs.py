#!/usr/bin/env python3
"""make_figs.py — Figures 2–5 of the IJCV manuscript, all values read from result JSONs (greyscale, serif, value labels).
Fig 2: stacked variance shares per model x metric (intrinsic + flow).
Fig 3: gauge — scene x training-set interaction under PSNR vs scene-gain-aligned PSNR, with ICC.
Fig 4: out-of-sample prediction of the JOSA A per-partition method gaps (30 points).
Fig 5: per-scene method gap vs an independently measured scene difficulty (main-experiment UNet-DirectCNN gap vs the Proposed model's PSNR;
       flow RAFT gap vs zero-shot RAFT-large EPE); coupled versions (difficulty = one member of the pair) are recorded in fig_stats.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution'); OUT = Path(__file__).parent / 'figs'; OUT.mkdir(exist_ok=True)
k1 = json.load(open(R / 'killing_v1.json')); m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
AN = {'direct': 'DirectCNN', 'unet': 'UNet', 'transformer': 'TransUNet', 'proposed': 'Proposed', 'raft_small': 'RAFT-small', 'raft_large': 'RAFT-large'}
MN = {'simse_R_gray_mean': 'si-MSE', 'lmse_R_chen_mean': 'LMSE', 'dssim_R_mean': 'DSSIM', 'R_PSNR_of_mean_mse': 'PSNR', 'R_siPSNR_scenegain_rgb': 'PSNR\nscene gain',
      'epe_mean': 'EPE', 'fl_all_mean': 'Fl-all', 'out3_mean': '>3px', 'epe_occ_mean': 'EPE occ', 'epe_noc_mean': 'EPE noc'}
def comp(m): return m['reml'] if m['used'] == 'reml' else m['moment']
GREYS = ['0.15', '0.45', '0.72', '0.92']

# ---------------- Fig 2: stacked shares
cols = [(a, y) for a in ['direct', 'unet', 'transformer', 'proposed'] for y in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse', 'R_siPSNR_scenegain_rgb']]
fcols = [(a, y) for a in ['raft_small', 'raft_large'] for y in ['epe_mean', 'fl_all_mean', 'out3_mean', 'epe_occ_mean', 'epe_noc_mean']]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={'width_ratios': [2, 1]})
def stacked(ax, items, src, title):
    x = np.arange(len(items)); bottoms = np.zeros(len(items))
    comps = [comp(src['archs'][a]['metrics'][y]) for a, y in items]
    tots = [c['s2_scene'] + c['s2_train'] + c['s2_inter'] + c['s2_seed'] for c in comps]
    for k, (key, lab) in enumerate([('s2_scene', 'scene (test-side)'), ('s2_train', 'training set'), ('s2_inter', 'scene × training set'), ('s2_seed', 'seed')]):
        vals = np.array([c[key] / t for c, t in zip(comps, tots)])
        ax.bar(x, vals, 0.8, bottom=bottoms, color=GREYS[k], edgecolor='0.3', linewidth=0.3, label=lab); bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels([MN[y] for a, y in items], fontsize=6.0, rotation=90)
    for a in sorted({a for a, _ in items}, key=lambda a: [i for i, (aa, _) in enumerate(items) if aa == a][0]):
        idx = [i for i, (aa, _) in enumerate(items) if aa == a]; ax.text(np.mean(idx), 1.02, AN[a], ha='center', fontsize=7)
        if idx[0] > 0: ax.axvline(idx[0] - 0.5, color='0.8', linewidth=0.6)
    ax.set_ylim(0, 1.10); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel('share of scene-level variance', fontsize=7.5); ax.set_title(title, loc='left', fontsize=8.5, pad=14)
stacked(ax1, cols, m2, '(a) Intrinsic decomposition'); stacked(ax2, fcols, fl, '(b) Optical flow')
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / 'fig2_shares.pdf'); fig.savefig(OUT / 'fig2_shares.png', dpi=220)

# ---------------- Fig 3: gauge
fig, ax = plt.subplots(figsize=(3.4, 2.5)); archs = ['direct', 'unet', 'transformer', 'proposed']; x = np.arange(4); w = 0.38
ip = [m2['archs'][a]['H4_gauge']['s2_inter_psnr'] for a in archs]; ig = [m2['archs'][a]['H4_gauge']['s2_inter_scenegain_psnr'] for a in archs]
b1 = ax.bar(x - w / 2, ip, w, color='0.25', label='PSNR'); b2 = ax.bar(x + w / 2, ig, w, color='0.75', edgecolor='0.35', linewidth=0.4, label='PSNR after per-scene gain')
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f'{b.get_height():.2f}', ha='center', fontsize=6.3)
for i, a in enumerate(archs):
    h = m2['archs'][a]['H4_gauge']; ax.text(i, max(ip[i], ig[i]) + 0.32, f'ICC {h["icc_psnr"]:.2f}→{h["icc_scenegain_psnr"]:.2f}\n−{100*h["interaction_reduction"]:.0f}%', ha='center', fontsize=6)
ax.set_xticks(x); ax.set_xticklabels([AN[a] for a in archs], fontsize=7); ax.set_ylabel('scene × training-set variance (dB²)'); ax.set_ylim(0, max(ip) * 1.55)
ax.legend(frameon=False, fontsize=6.5, loc='upper right'); fig.tight_layout(); fig.savefig(OUT / 'fig3_gauge.pdf'); fig.savefig(OUT / 'fig3_gauge.png', dpi=220)

# ---------------- Fig 4: out-of-sample prediction
v = m2['H3_josaa_validation']; rows = v['rows']
fig, ax = plt.subplots(figsize=(3.4, 3.0))
dr = [r for r in rows if 'direct' in r['pair']]; ot = [r for r in rows if 'direct' not in r['pair']]
ax.scatter([r['predicted_gap_dB'] for r in dr], [r['observed_gap_dB'] for r in dr], s=22, color='0.15', marker='o', label='pairs involving DirectCNN')
ax.scatter([r['predicted_gap_dB'] for r in ot], [r['observed_gap_dB'] for r in ot], s=22, facecolors='white', edgecolors='0.3', marker='s', label='pairs among UNet-family models')
lim = [min(min(r['predicted_gap_dB'] for r in rows), min(r['observed_gap_dB'] for r in rows)) - 0.2, max(max(r['predicted_gap_dB'] for r in rows), max(r['observed_gap_dB'] for r in rows)) + 0.2]
ax.plot(lim, lim, color='0.5', linestyle='--', linewidth=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
pd_ = np.array([r['predicted_gap_dB'] for r in dr]); od = np.array([r['observed_gap_dB'] for r in dr]); po = np.array([r['predicted_gap_dB'] for r in ot]); oo = np.array([r['observed_gap_dB'] for r in ot])
ax.text(0.03, 0.97, f"all 30: r = {v['pearson_r']:.2f}, RMSE {v['rmse_dB']:.2f} dB\nDirectCNN pairs: r = {np.corrcoef(pd_, od)[0,1]:.2f}\nUNet-family pairs: r = {np.corrcoef(po, oo)[0,1]:.2f}, mean |obs| {np.abs(oo).mean():.2f} dB", transform=ax.transAxes, va='top', fontsize=6.5)
ax.set_xlabel('predicted partition gap (dB)'); ax.set_ylabel('observed gap in the earlier grid (dB)'); ax.legend(frameon=False, fontsize=6.3, loc='lower right')
fig.tight_layout(); fig.savefig(OUT / 'fig4_prediction.pdf'); fig.savefig(OUT / 'fig4_prediction.png', dpi=220)

# ---------------- Fig 5: method gap vs an independently measured difficulty (round 2, C4: no term of the difference appears on the x-axis)
def pfmt(p): return 'p < 0.001' if p < 0.001 else (f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
mp = next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'R_PSNR_of_mean_mse')
ref_tab = {r['scene']: r['score'] for r in m2['archs']['proposed']['metrics']['R_PSNR_of_mean_mse']['scene_table']}
sc = sorted(ref_tab); x = np.array([ref_tab[s] for s in sc]); y = np.array([mp['per_scene_gap'][s] for s in sc]); rho, p = spearmanr(x, y)
ax1.axhline(0, color='0.6', linewidth=0.6); ax1.scatter(x, y, s=20, color='0.15')
for s_, xx, yy in zip(sc, x, y):
    if abs(yy) > 1.8: ax1.annotate(s_.replace('_', ' '), (xx, yy), fontsize=5.6, xytext=(3, 2), textcoords='offset points')
ax1.set_xlabel('reference: Proposed model, reflectance PSNR (dB)\n(higher = easier)'); ax1.set_ylabel('UNet − DirectCNN (dB)'); ax1.set_title(f'(a) Intrinsic: Spearman ρ = {rho:.2f} (descriptive)', loc='left', fontsize=8.5)
fp = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'); zs_tab = fl['zeroshot']['raft_large']['per_scene_epe']
sc2 = sorted(set(zs_tab) & set(fp['per_scene_gap'])); x2 = np.array([zs_tab[s] for s in sc2]); y2 = np.array([fp['per_scene_gap'][s] for s in sc2]); rho2, p2 = spearmanr(x2, y2)
ax2.axhline(0, color='0.6', linewidth=0.6); ax2.scatter(x2, y2, s=20, color='0.15')
for s_, xx, yy in zip(sc2, x2, y2):
    if yy > 2.0: ax2.annotate(s_.replace('_', ' '), (xx, yy), fontsize=5.6, xytext=(3, 2), textcoords='offset points')
ax2.set_xlabel('reference: zero-shot RAFT-large (C+T) EPE (px)\n(higher = harder)'); ax2.set_ylabel('RAFT-small − RAFT-large EPE (px)'); ax2.set_title(f'(b) Flow: Spearman ρ = {rho2:.2f} (descriptive)', loc='left', fontsize=8.5)
fig.tight_layout(w_pad=2); fig.savefig(OUT / 'fig5_gap_vs_difficulty.pdf'); fig.savefig(OUT / 'fig5_gap_vs_difficulty.png', dpi=220)
# the coupled versions (difficulty = one member of the pair) are kept for the record
kp = next(p for p in k1['pairs'] if p['metric'] == 'R_PSNR_of_mean_mse'); ktab = {r['scene']: r['psnr'] for r in k1['archs']['direct']['metrics']['R_PSNR_of_mean_mse']['scene_table']}
ksc = sorted(ktab); rho_k, p_k = spearmanr([ktab[s_] for s_ in ksc], [kp['per_scene_gap'][s_] for s_ in ksc])
ftab = {r['scene']: r['score'] for r in fl['archs']['raft_large']['metrics']['epe_mean']['scene_table']}; fsc = sorted(ftab); rho_f, p_f = spearmanr([ftab[s_] for s_ in fsc], [fp['per_scene_gap'][s_] for s_ in fsc])
json.dump({'fig5_intrinsic_spearman': float(rho), 'fig5_intrinsic_p': float(p), 'fig5_flow_spearman': float(rho2), 'fig5_flow_p': float(p2), 'fig5_intrinsic_n': int(len(sc)), 'fig5_flow_n': int(len(sc2)),
           'coupled_intrinsic_spearman_killing': float(rho_k), 'coupled_intrinsic_p': float(p_k), 'coupled_flow_spearman': float(rho_f), 'coupled_flow_p': float(p_f)}, open(OUT / 'fig_stats.json', 'w'), indent=1)
print('figs written; fig5 rho intrinsic', round(rho, 3), 'flow', round(rho2, 3))
