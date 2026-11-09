#!/usr/bin/env python3
"""target_consistency_audit.py — the reconstruction terms of the training losses against the supervised targets (series-2 review,
round 1, A1). Interface card: INTERFACE_CARD_sintel_loader_and_losses.md (sources for every definition used here).

The targets are R = A, S = clamp(C/(A+1e-4), 0, 2)/2, H = clamp(F-C, 0, 1) (v7_base.py:104-106) and the reconstruction terms penalise
mse(Rp*Sp, F) (DirectCNN, weight 0.5; v7_base.py:385) and mse(Rp*Sp+Hp, F) (UNet, TransUNet, Proposed, weight 1; v7_base.py:413, 450).
Substituting the targets gives R*S + H ~= C/2 + (F-C)+, not F: the reconstruction term and the supervised terms cannot both be zero.
This script quantifies the conflict on the dataset tensors the networks were trained on, and reads from the per-run records what the
trained networks actually did:
  1. per scene, the reconstruction MSE obtained by substituting the ground-truth targets, for the 2-head and 3-head forms, and for the
     same forms with the factor of two undone (2*R*S), plus the fractions of pixels where the S clamp (C/(A+1e-4) >= 2) and the H clamp
     (F < C) are active;
  2. per architecture, the reconstruction MSE the trained networks achieved (per_scene Recon_MSE_mean, mean over the 75 random runs);
  3. per architecture, the distribution over (scene, training set) cells of the pooled optimal per-scene RGB gain alpha_u = xy_sum/xx_sum
     of Eq. (24): alpha < 1 means the predicted reflectance is brighter than the albedo pass, the direction in which the reconstruction
     term pulls a network whose shading target is halved.
Controls (run first; the script aborts if they fail): a known-answer synthetic check of the two substitution identities, the frame count
(1064), and name assertions on every record field.
Series-2 round 3 (C1): alpha = cos(theta) ||x||/||xhat|| mixes magnitude and direction, so the RMS-magnitude ratio ||xhat||/||x|| = sqrt(xx/tt)
and cos(theta) = xy/sqrt(xx tt) are reported separately; alpha < 1 does not by itself mean a larger prediction.
Output: results/benchmark_resolution/target_consistency_audit.json. CPU only; environment: JEI_IntrinsicUQ/env (torch 2.11.0)."""
import glob, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import v7_base as B                                                     # the training code's own loader

ROOT = Path('/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/data/sintel_real/training')
RES = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
SIZE, N_FRAMES = 256, 1064
FIELDS = ['Recon_MSE_mean', 'gain_R_rgb_xy_sum', 'gain_R_rgb_xx_sum', 'gain_R_rgb_tt_sum', 'n_frames', 'simse_R_rgb_mean', 'simse_R_gray_mean']


def mse(a, b): return float(((a - b) ** 2).mean())


def known_answer_control():
    g = torch.Generator().manual_seed(0)
    A = 0.2 + 0.6 * torch.rand(3, 64, 64, generator=g); s = 0.1 + 0.8 * torch.rand(3, 64, 64, generator=g)
    C = 2 * A * s; F = C.clone()
    S = torch.clamp(C / (A + 1e-4), 0, 2.0) / 2.0; H = torch.clamp(F - C, 0, 1)      # the loader's formulas (v7_base.py:104-106)
    assert float(S.max()) < 1.0 and float(H.abs().max()) == 0.0
    got3, exp3 = mse(A * S + H, F), float(((C / 2) ** 2).mean())
    got3x2 = mse(2 * A * S + H, F)
    # tolerance from the stabiliser: |S - s| <= s*1e-4/(A+1e-4) <= 0.9e-4/0.2 = 4.5e-4, so the squared residual differs from (C/2)^2 by
    # at most ~2*4.5e-4/(1-...) ~ 1e-3 relative; 2e-3 is the bound, not a fitted number (observed 3.6e-4)
    assert abs(got3 - exp3) < 2e-3 * exp3, (got3, exp3)
    assert got3x2 < 1e-6, got3x2
    return {'mse_RS_plus_H_vs_F': got3, 'expected_mean_C_half_sq': exp3, 'mse_2RS_plus_H_vs_F': got3x2}


def main():
    ctrl = known_answer_control(); print('known-answer control passed:', ctrl, flush=True)
    cache = B.SintelDataset._get_global_cache(str(ROOT), SIZE)              # same code path as training (v7_base.py:65-121)
    F, R, S, H, meta = cache['F'], cache['R'], cache['S'], cache['H'], cache['meta']
    assert F.shape[0] == N_FRAMES, F.shape
    # C is not cached; rebuild it with the loader's own transform (v7_base.py:80-83) to measure the clamp fractions
    import os
    from PIL import Image
    from torchvision import transforms
    tfm = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])
    scenes = sorted(set(m[0] for m in meta)); per_scene = {}
    for sc in scenes:
        idx = [i for i, m in enumerate(meta) if m[0] == sc]
        Cs = torch.stack([tfm(Image.open(os.path.join(ROOT, 'clean', sc, meta[i][1])).convert('RGB')) for i in idx])
        Fs, Rs, Ss, Hs = F[idx], R[idx], S[idx], H[idx]
        # reproduction control: the loader's S and H from this C equal the cached ones
        S_re = torch.clamp(Cs / (Rs + 1e-4), 0, 2.0) / 2.0; H_re = torch.clamp(Fs - Cs, 0, 1)
        assert float((S_re - Ss).abs().max()) < 1e-6 and float((H_re - Hs).abs().max()) < 1e-6, sc
        per_scene[sc] = {
            'n_frames': len(idx),
            'recon2_gt_targets': mse(Rs * Ss, Fs),                       # DirectCNN term with the targets substituted
            'recon3_gt_targets': mse(Rs * Ss + Hs, Fs),                  # 3-head term with the targets substituted
            'recon2_gt_targets_x2': mse(2 * Rs * Ss, Fs),                # the same with the halving undone
            'recon3_gt_targets_x2': mse(2 * Rs * Ss + Hs, Fs),
            'mse_C_vs_F': mse(Cs, Fs),                                   # what remains between clean and final passes
            'frac_S_clamped': float((Cs / (Rs + 1e-4) >= 2.0).float().mean()),
            'frac_F_below_C': float((Fs < Cs).float().mean()),
            'mean_F': float(Fs.mean()), 'mean_C': float(Cs.mean()), 'mean_A': float(Rs.mean())}
        print(f"{sc:12s} recon3(gt) {per_scene[sc]['recon3_gt_targets']:.4f}  x2 {per_scene[sc]['recon3_gt_targets_x2']:.4f}  C-vs-F {per_scene[sc]['mse_C_vs_F']:.4f}  S clamped {per_scene[sc]['frac_S_clamped']:.3f}  F<C {per_scene[sc]['frac_F_below_C']:.3f}", flush=True)
    def pooled(key): return float(np.average([v[key] for v in per_scene.values()], weights=[v['n_frames'] for v in per_scene.values()]))
    summary = {k: pooled(k) for k in ['recon2_gt_targets', 'recon3_gt_targets', 'recon2_gt_targets_x2', 'recon3_gt_targets_x2', 'mse_C_vs_F', 'frac_S_clamped', 'frac_F_below_C']}
    summary['scene_mean_recon3_gt_targets'] = float(np.mean([v['recon3_gt_targets'] for v in per_scene.values()]))
    summary['scene_mean_recon3_gt_targets_x2'] = float(np.mean([v['recon3_gt_targets_x2'] for v in per_scene.values()]))
    # what the trained networks did: achieved reconstruction MSE and pooled per-scene gains, from the per-run records (random design)
    archs = {}
    for fp in sorted(glob.glob(str(RES / 'results_cv_v2' / '**' / 'cv2_*_v2.json'), recursive=True)):
        d = json.load(open(fp))
        if d.get('kind') != 'cv' or d.get('design') != 'rand': continue
        a = archs.setdefault(d['arch'], {'recon': [], 'alpha': [], 'alpha_cells': {}, 'norm_ratio': [], 'cos': []})
        for sc, ps in d['per_scene'].items():
            for f in FIELDS: assert f in ps, (fp, sc, f)
            xy, xx, tt = ps['gain_R_rgb_xy_sum'], ps['gain_R_rgb_xx_sum'], ps['gain_R_rgb_tt_sum']      # sum x*xhat, sum xhat^2, sum x^2 (experiment_cv2.py:98-101)
            a['recon'].append(ps['Recon_MSE_mean']); alpha = xy / xx; a['alpha'].append(alpha)
            a['alpha_cells'].setdefault(f"{sc}|r{d['rep']}f{d['fold']}", []).append(alpha)
            # series-2 round 3 (C1): alpha = cos(theta) * ||x|| / ||xhat||; the RMS-magnitude ratio and the angular agreement separately
            a['norm_ratio'].append(np.sqrt(xx / tt)); a['cos'].append(xy / np.sqrt(xx * tt))
            assert abs(alpha - a['cos'][-1] / a['norm_ratio'][-1]) < 1e-9
    out_archs = {}
    for a, v in archs.items():
        al = np.array(v['alpha']); cell_alpha = np.array([np.mean(x) for x in v['alpha_cells'].values()])
        out_archs[a] = {'n_scene_rows': len(al), 'achieved_recon_mse_mean': float(np.mean(v['recon'])), 'achieved_recon_mse_median': float(np.median(v['recon'])),
                        'alpha_median': float(np.median(al)), 'alpha_q05': float(np.percentile(al, 5)), 'alpha_q95': float(np.percentile(al, 95)),
                        'frac_alpha_below_1': float((al < 1).mean()), 'frac_alpha_below_0p9': float((al < 0.9).mean()),
                        'cell_alpha_median': float(np.median(cell_alpha)), 'frac_cells_alpha_below_1': float((cell_alpha < 1).mean()), 'n_cells': int(len(cell_alpha)),
                        'norm_ratio_median': float(np.median(v['norm_ratio'])), 'norm_ratio_q05': float(np.percentile(v['norm_ratio'], 5)), 'norm_ratio_q95': float(np.percentile(v['norm_ratio'], 95)),
                        'frac_norm_ratio_above_1': float((np.array(v['norm_ratio']) > 1).mean()), 'cos_median': float(np.median(v['cos'])), 'cos_min': float(np.min(v['cos']))}
        print(f"{a:12s} achieved recon MSE mean {out_archs[a]['achieved_recon_mse_mean']:.4f} | alpha median {out_archs[a]['alpha_median']:.3f} [5-95%: {out_archs[a]['alpha_q05']:.3f}, {out_archs[a]['alpha_q95']:.3f}] frac<1 {out_archs[a]['frac_alpha_below_1']:.2f} | ||xhat||/||x|| median {out_archs[a]['norm_ratio_median']:.3f} frac>1 {out_archs[a]['frac_norm_ratio_above_1']:.2f} | cos median {out_archs[a]['cos_median']:.3f}", flush=True)
    out = {'known_answer_control': ctrl, 'definitions_source': 'INTERFACE_CARD_sintel_loader_and_losses.md', 'per_scene': per_scene, 'pooled_frame_weighted': summary, 'archs': out_archs}
    (RES / 'target_consistency_audit.json').write_text(json.dumps(out, indent=1)); print('wrote', RES / 'target_consistency_audit.json')
    print('pooled:', json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
