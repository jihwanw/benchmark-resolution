#!/usr/bin/env python3
"""make_qual_figs.py — qualitative figures for the IJCV manuscript from the saved predictions (qual_intrinsic.npz, qual_flow.npz).
All figures are drawn at the manuscript text width (372 pt = 13.1 cm; sn-jnl) and sized to fill a page with their caption.
  Fig 6  23-scene mosaic (final pass, middle frame), 4 per row, ordered by intrinsic difficulty, with si-MSE and EPE annotations.
  Fig 7  gauge: one held-out scene predicted by DirectCNN under five training compositions (context row) and large crops of the
         automatically selected window (GT + five predictions).
  Fig 8  intrinsic scene x method interaction: hard and easy scene; per scene a context row (input, GT, DirectCNN, UNet) and
         large crops of the automatically selected window (GT, DirectCNN, UNet).
  Fig 9  flow scene x method interaction: hard and easy scene; per scene a context row (frame, GT flow, RAFT-small, RAFT-large)
         and large crops (frame, EPE map of each model).
Crop windows are selected automatically, never by hand: the window maximising the mean absolute difference between the two
quantities compared (mean prediction of the two groups of training compositions in Fig 7; per-pixel error of the two models in
Figs 8 and 9). Intrinsic windows are 128 x 64 pixels of the networks' 256 x 256 space (half the height, a quarter of the width of
the frame, displayed at the original 1024 x 436 aspect ratio); flow windows are 256 x 256 pixels of the full-resolution frame.
Window coordinates and patch errors are written to figs/qual_fig_stats.json and become LaTeX macros (make_numbers.py).
Images: MPI-Sintel (c) Blender Foundation | durian.blender.org, CC BY 3.0 (attribution in captions)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

Q = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution/qual')
R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution'); OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / 'figs'
qi = np.load(Q / 'qual_intrinsic.npz'); mi = json.load(open(Q / 'qual_intrinsic_meta.json'))
qf = np.load(Q / 'qual_flow.npz'); mf = json.load(open(Q / 'qual_flow_meta.json'))
m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
plt.rcParams.update({'font.family': 'serif', 'font.size': 7})
IN = 1 / 2.54
TW = 372 / 72.27                 # text width in inches (372 pt)
ASPECT = 436 / 1024              # networks see 256x256 resized frames; display at the original Sintel aspect ratio
PH, PW = 128, 64                 # intrinsic crop window in 256x256 network space -> displayed aspect 128*ASPECT/64 = 0.85
ZP = 256                         # flow crop window in full-resolution pixels
EDGE = dict(edgecolor='#d62728', facecolor='none', linewidth=1.2)
STATS = {}

def chw(a): return np.clip(np.transpose(a, (1, 2, 0)), 0, 1)
def frame_key(prefix):
    tags = sorted({k[len(prefix):].split('/')[0] for k in qi.files if k.startswith(prefix)}, key=lambda t: int(t[1:]))
    return tags[-1]

def best_window(score_map, ph, pw):
    """Top-left corner of the ph x pw window with the largest mean of score_map (2-D), via a summed-area table."""
    s = np.pad(score_map, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = score_map.shape
    tot = s[ph:H + 1, pw:W + 1] - s[:H - ph + 1, pw:W + 1] - s[ph:H + 1, :W - pw + 1] + s[:H - ph + 1, :W - pw + 1]
    i, j = np.unravel_index(np.argmax(tot), tot.shape)
    return int(i), int(j)

def show(ax, img, title=None, aspect=ASPECT, cmap=None, vmin=None, vmax=None, fs=6.5, interp='bilinear'):
    ax.imshow(img, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp); ax.axis('off')
    if title: ax.set_title(title, fontsize=fs, pad=1.8)

def box(ax, i, j, ph, pw):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), pw, ph, **EDGE))

def grid(nrows_spec, wspace=0.04, hspace_cm=0.55, left_cm=0.0, top_cm=0.45):
    """Figure + gridspec from a list of row heights (cm) on a common 12-column grid; top_cm leaves room for the first row's
    titles and left_cm for rotated row labels."""
    heights = [h + hspace_cm for h in nrows_spec]; H = sum(heights) + top_cm
    fig = plt.figure(figsize=(TW, H * IN))
    gs = fig.add_gridspec(len(heights), 12, height_ratios=heights, hspace=0.0, wspace=wspace,
                          left=left_cm / (TW / IN), right=0.998, top=1 - top_cm / H, bottom=0.002)
    return fig, gs

# ---------------- Fig 6: scene mosaic, 3 per row, labels overlaid on the frames so that 8 rows fit on one page with the caption.
# Reading aid: red border = five hardest scenes for reflectance (DirectCNN si-MSE), blue border = five hardest for flow (RAFT-small EPE);
# each label gives the scene and its rank under both tasks, so the mismatch of the two orderings is visible at a glance.
simse = {r['scene']: r['score'] for r in m2['archs']['direct']['metrics']['simse_R_gray_mean']['scene_table']}
epe = {r['scene']: r['score'] for r in fl['archs']['raft_small']['metrics']['epe_mean']['scene_table']}
scenes = sorted(simse, key=lambda s: -simse[s]); flow_rank = {s: k + 1 for k, s in enumerate(sorted(epe, key=lambda s: -epe[s]))}
TOPK = 5; hard_i = set(scenes[:TOPK]); hard_f = {s for s in epe if flow_rank[s] <= TOPK}
RED, BLUE = '#d62728', '#1f77b4'
ncol = 3; nrow = int(np.ceil((len(scenes) + 1) / ncol)); pw_cm = TW / ncol / IN; ph_cm = pw_cm * ASPECT
fig, gs = grid([ph_cm] * nrow, hspace_cm=0.10, top_cm=0.05)
for k, s in enumerate(scenes):
    r, c = divmod(k, ncol); ax = fig.add_subplot(gs[r, c * 4:(c + 1) * 4]); im = chw(qi[f'thumb/{s}/final']); show(ax, im)
    H, W = im.shape[:2]
    ax.text(0.012, 0.04, f"{s.replace('_', ' ')}  $-$  refl. #{k+1}  $\\cdot$  flow #{flow_rank[s]}", transform=ax.transAxes, fontsize=5.9, color='white',
            va='bottom', ha='left', bbox=dict(boxstyle='square,pad=0.25', facecolor='black', alpha=0.62, edgecolor='none'))
    if s in hard_i: ax.add_patch(Rectangle((-0.5, -0.5), W, H, edgecolor=RED, facecolor='none', linewidth=2.6, clip_on=False))
    if s in hard_f: ax.add_patch(Rectangle((W * 0.015 - 0.5, H * 0.035 - 0.5), W * 0.97, H * 0.93, edgecolor=BLUE, facecolor='none', linewidth=2.2, clip_on=False))
ax = fig.add_subplot(gs[nrow - 1, 8:12]); ax.axis('off')
ax.add_patch(Rectangle((0.03, 0.60), 0.10, 0.22, transform=ax.transAxes, edgecolor=RED, facecolor='none', linewidth=2.6))
ax.text(0.17, 0.71, f'{TOPK} hardest for reflectance\n(DirectCNN si-MSE)', transform=ax.transAxes, fontsize=6.2, va='center')
ax.add_patch(Rectangle((0.03, 0.18), 0.10, 0.22, transform=ax.transAxes, edgecolor=BLUE, facecolor='none', linewidth=2.2))
ax.text(0.17, 0.29, f'{TOPK} hardest for flow\n(RAFT-small end-point error)', transform=ax.transAxes, fontsize=6.2, va='center')
ax.text(0.03, 0.02, 'labels: rank under each task', transform=ax.transAxes, fontsize=6.0, va='bottom')
STATS['fig6'] = {'order': scenes, 'flow_rank': flow_rank, 'hardest_reflectance': scenes[:TOPK], 'hardest_flow': sorted(hard_f, key=lambda s: flow_rank[s]),
                 'both': sorted(hard_i & hard_f), 'top_k': TOPK}
fig.savefig(OUT / 'fig6_scenes.pdf', dpi=220); fig.savefig(OUT / 'fig6_scenes.png', dpi=160); plt.close(fig)

# ---------------- Fig 7: gauge — context rows (4 per row) + large crops (3 per row)
comps = sorted(mi['jobs']['gauge']['compositions'], key=lambda c: c['rep']); sc = mi['jobs']['gauge']['scene']
ftag = frame_key(f"gauge/rep{comps[0]['rep']}/{sc}/"); key0 = f"gauge/rep{comps[0]['rep']}/{sc}/{ftag}/"
inp, gt = chw(qi[key0 + 'input']), chw(qi[key0 + 'gt_R'])
preds = [chw(qi[f"gauge/rep{c['rep']}/{sc}/{ftag}/pred_R"]) for c in comps]
grp_in = np.mean([p for p, c in zip(preds, comps) if c['bamboo_1_in_train']], 0)
grp_out = np.mean([p for p, c in zip(preds, comps) if not c['bamboo_1_in_train']], 0)
diff_map = np.abs(grp_in - grp_out).mean(-1)
i7, j7 = best_window(diff_map, PH, PW)
STATS['fig7'] = {'scene': sc, 'frame_tag': ftag, 'window_rc': [i7, j7], 'window_hw': [PH, PW],
                 'criterion': 'max mean |mean(pred, compositions with bamboo_1 in train) - mean(pred, compositions without)| over 128x64 windows (network space)'}
lab = lambda c: f"training set {c['rep']+1}{'' if c['bamboo_1_in_train'] else '*'}"
ctx = [('input (final pass)', inp, None), ('ground-truth albedo', gt, None)] + \
      [(f"{lab(c)}\n{c['scene_psnr']:.1f} / {c['scene_gain_psnr']:.1f} dB, gain {c['optimal_gain']:.2f}", p, None) for c, p in zip(comps, preds)] + \
      [('|group mean difference|\n(darker = larger)', diff_map, 'gray_r')]
crop_w = TW / 3 / IN; crop_h = crop_w * (PH * ASPECT / PW); ctx_h = TW / 4 / IN * ASPECT
fig, gs = grid([ctx_h, ctx_h, crop_h, crop_h], hspace_cm=0.75)
for k, (t, im, cm) in enumerate(ctx):
    r, c = divmod(k, 4); ax = fig.add_subplot(gs[r, c * 3:(c + 1) * 3])
    show(ax, im, t, cmap=cm, vmin=0 if cm else None, vmax=diff_map.max() if cm else None, fs=6.0); box(ax, i7, j7, PH, PW)
crops = [('ground truth', gt)] + [(lab(c), p) for c, p in zip(comps, preds)]
for k, (t, im) in enumerate(crops):
    r, c = divmod(k, 3); ax = fig.add_subplot(gs[2 + r, c * 4:(c + 1) * 4]); show(ax, im[i7:i7 + PH, j7:j7 + PW], t, fs=7.0)
fig.savefig(OUT / 'fig7_gauge.pdf', dpi=220); fig.savefig(OUT / 'fig7_gauge.png', dpi=160); plt.close(fig)

# ---------------- Fig 8: intrinsic interaction — per scene a context row (4) and a crop row (3)
inter = mi['jobs']['inter']; ps = inter['per_scene_psnr']; STATS['fig8'] = {}
fig, gs = grid([ctx_h, crop_h, ctx_h, crop_h], hspace_cm=0.75, left_cm=0.42)
for r, s in enumerate(inter['scenes_shown']):
    ft = frame_key(f'inter/direct/{s}/')
    inp = chw(qi[f'inter/direct/{s}/{ft}/input']); gt = chw(qi[f'inter/direct/{s}/{ft}/gt_R'])
    pd_, pu = chw(qi[f'inter/direct/{s}/{ft}/pred_R']), chw(qi[f'inter/unet/{s}/{ft}/pred_R'])
    ed, eu = np.linalg.norm(pd_ - gt, axis=-1), np.linalg.norm(pu - gt, axis=-1)
    i8, j8 = best_window(np.abs(ed - eu), PH, PW)
    STATS['fig8'][s] = {'frame_tag': ft, 'window_rc': [i8, j8], 'window_hw': [PH, PW], 'criterion': 'max mean |err_DirectCNN - err_UNet| (RGB L2 per pixel) over 128x64 windows',
                        'patch_rmse_direct': float(np.sqrt((ed[i8:i8 + PH, j8:j8 + PW] ** 2).mean())), 'patch_rmse_unet': float(np.sqrt((eu[i8:i8 + PH, j8:j8 + PW] ** 2).mean()))}
    row = [('input\n ', inp), ('ground-truth albedo\n ', gt), (f"DirectCNN\n{ps[f'direct/{s}']:.1f} dB", pd_), (f"UNet\n{ps[f'unet/{s}']:.1f} dB", pu)]
    for c, (t, im) in enumerate(row):
        ax = fig.add_subplot(gs[2 * r, c * 3:(c + 1) * 3]); show(ax, im, t, fs=6.2); box(ax, i8, j8, PH, PW)
        if c == 0: ax.text(-0.035, 0.5, s.replace('_', ' '), transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=7.5)
    cut = lambda a: a[i8:i8 + PH, j8:j8 + PW]
    for c, (t, im) in enumerate([('ground truth', cut(gt)), ('DirectCNN', cut(pd_)), ('UNet', cut(pu))]):
        ax = fig.add_subplot(gs[2 * r + 1, c * 4:(c + 1) * 4]); show(ax, im, t, fs=7.0)
fig.savefig(OUT / 'fig8_interaction.pdf', dpi=220); fig.savefig(OUT / 'fig8_interaction.png', dpi=160); plt.close(fig)

# ---------------- Fig 9: flow interaction — per scene a context row (4) and a crop row (frame, EPE small, EPE large)
def flow_rgb(f, maxmag):
    u, v = f[0], f[1]; mag = np.sqrt(u ** 2 + v ** 2); ang = np.arctan2(v, u)
    hsv = np.stack([(ang + np.pi) / (2 * np.pi), np.clip(mag / maxmag, 0, 1), np.ones_like(mag)], -1)
    return colors.hsv_to_rgb(hsv)
sm = mf['models']['raft_small']['scene_epe_finetuned']; lg = mf['models']['raft_large']['scene_epe_finetuned']; STATS['fig9'] = {}
crop_h9 = TW / 3 / IN
fig, gs = grid([ctx_h, crop_h9, ctx_h, crop_h9], hspace_cm=0.75, left_cm=0.42)
for r, s in enumerate(mf['scenes_shown']):
    pair = [p for p in mf['pairs_shown'] if p.startswith(s)][-1]; pair = pair.rsplit('.', 1)[0] if pair.endswith('.png') else pair
    bs, bl = f'finetuned/raft_small/{pair}/', f'finetuned/raft_large/{pair}/'
    img = np.transpose(qf[bs + 'img1'], (1, 2, 0)); gt = qf[bs + 'gt']; valid = qf[bs + 'valid']
    es, el = qf[bs + 'epe'] * valid, qf[bl + 'epe'] * valid
    mm = max(np.percentile(np.linalg.norm(gt, axis=0), 99), 1e-6)
    i9, j9 = best_window(np.abs(es - el), ZP, ZP)
    vcount = max(valid[i9:i9 + ZP, j9:j9 + ZP].sum(), 1)
    STATS['fig9'][s] = {'pair': pair, 'window_rc': [i9, j9], 'window_hw': [ZP, ZP], 'criterion': 'max mean |EPE_small - EPE_large| over 256x256 windows (valid pixels)',
                        'patch_epe_small': float(es[i9:i9 + ZP, j9:j9 + ZP].sum() / vcount), 'patch_epe_large': float(el[i9:i9 + ZP, j9:j9 + ZP].sum() / vcount)}
    row = [('frame\n ', img), ('ground-truth flow\n ', flow_rgb(gt, mm)), (f'RAFT-small\nscene EPE {sm[s]:.2f} px', flow_rgb(qf[bs + 'pred'], mm)), (f'RAFT-large\nscene EPE {lg[s]:.2f} px', flow_rgb(qf[bl + 'pred'], mm))]
    for c, (t, im) in enumerate(row):
        ax = fig.add_subplot(gs[2 * r, c * 3:(c + 1) * 3]); show(ax, im, t, aspect='equal', fs=6.2, interp='antialiased'); box(ax, i9, j9, ZP, ZP)
        if c == 0: ax.text(-0.035, 0.5, s.replace('_', ' '), transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=7.5)
    cut = lambda a: a[i9:i9 + ZP, j9:j9 + ZP]
    emax = max(cut(es).max(), cut(el).max(), 1e-6)
    for c, (t, im, cm) in enumerate([('frame', cut(img), None), (f'end-point error, RAFT-small\npatch mean {STATS["fig9"][s]["patch_epe_small"]:.2f} px', cut(es), 'gray_r'),
                                      (f'end-point error, RAFT-large\npatch mean {STATS["fig9"][s]["patch_epe_large"]:.2f} px', cut(el), 'gray_r')]):
        ax = fig.add_subplot(gs[2 * r + 1, c * 4:(c + 1) * 4]); show(ax, im, t, aspect='equal', cmap=cm, vmin=0 if cm else None, vmax=emax if cm else None, fs=6.6, interp='antialiased')
fig.savefig(OUT / 'fig9_flow_interaction.pdf', dpi=220); fig.savefig(OUT / 'fig9_flow_interaction.png', dpi=160); plt.close(fig)

json.dump({'gauge_compositions': comps, 'inter_psnr': ps, 'flow_scene_epe': {'raft_small': sm, 'raft_large': lg}, 'zoom_windows': STATS},
          open(OUT / 'qual_fig_stats.json', 'w'), indent=1)
print('qualitative figures written'); print(json.dumps(STATS, indent=0)[:900])
