#!/usr/bin/env python3
"""experiment_cv2.py — Benchmark-resolution MAIN experiment (v2), pre-registered 2026-11-02 (user approval same day).

Pre-registration: PhD/AI/CV/ideas/2026-11-01_benchmark_resolution.md (section "본실험 설계 (사전 등록 갱신, 2026-11-02)").
Killing experiment (v1): experiment_cv_v1_killing.py / results_cv_v1.  Parent: JOSA A 621704 (experiment_v8.py, v7_base.py).

Design
  * Random repeated scene-level 5-fold CV: R=5 repeats (fold seeds 201..205, identical to v1), 23 Sintel scenes,
    folds 5/5/5/4/4; S=3 model seeds (42,43,44); 4 architectures = DirectCNN, UNet(physics), TransformerUNet,
    Proposed (heteroscedastic) — the JOSA A protocol-study set, same recipe (60 epochs, batch 16, 256 px).
    -> 4 x 5 x 5 x 3 = 300 runs.  Seeds 42/43 of direct/unet duplicate v1 configurations exactly and serve as a
    GPU non-determinism control (same seed, independent run).
  * Stratified arm (H6): R_strat=2 repeats (fold seeds 301,302) in which folds are a systematic sample from the
    difficulty-sorted scene list (strata_v2.json; within-stratum order shuffled by the fold seed), for DirectCNN and
    UNet, S=2 seeds (42,43) -> 2 x 2 x 5 x 2 = 40 runs.
  * Shard assignment: deterministic longest-processing-time (LPT) packing by expected minutes
    (direct 14.6 / unet 24.1 / transformer 34.8 / proposed 26.1, measured on T4), so shards finish together.
  * Resume: one JSON per run, skip-if-exists; the launcher syncs the shard's S3 prefix down before starting.

Per-frame metrics: as v1 (R/S/H/Recon MSE, PSNR, si-MSE gray+rgb, LMSE Chen & Koltun + Grosse-normalized,
DSSIM) plus gain sufficient statistics (sum x*xhat, sum xhat^2, sum x^2, n_pixels; RGB and gray) so that
PSNR after a *pooled* (per-scene or per-run) optimal gain can be computed post hoc (H4: gauge interaction).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import v7_base as B
from experiment_v8 import TransformerUNet          # unchanged JOSA A definition (sha256-verified copy)

DEVICE = B.DEVICE
VERSION = 'v2'
FOLD_SEED_BASE_RAND = 201
FOLD_SEED_BASE_STRAT = 301
EXPECTED_MIN = {'direct': 14.6, 'unet': 24.1, 'transformer': 34.8, 'proposed': 26.1}


# ----------------------------------------------------------------------------
class SceneSubsetDataset(B.SintelDataset):
    def __init__(self, root, size, scenes_subset):           # noqa: D107
        self.size = size; self.protocol = 'cvfold'; self.split = 'custom'
        cache = self._get_global_cache(root, size); self._cache = cache
        picked = set(scenes_subset)
        self.split_scenes = sorted(picked)
        self.indices = [i for i, (s, _f, _fi) in enumerate(cache['meta']) if s in picked]

    def meta(self):
        return [self._cache['meta'][i] for i in self.indices]


def all_scenes(root, size):
    return list(B.SintelDataset._get_global_cache(root, size)['scenes'])


def build_folds_random(scenes, n_folds, fold_seed):
    rng = np.random.RandomState(fold_seed)
    order = list(scenes); rng.shuffle(order)
    return [sorted(p.tolist()) for p in np.array_split(np.array(order, dtype=object), n_folds)]


def build_folds_stratified(scenes, n_folds, fold_seed, strata):
    """Systematic sample from the difficulty-sorted list: shuffle within stratum, concatenate hard->easy,
    fold f takes positions f, f+n_folds, ... -> every fold spans all strata; sizes 5/5/5/4/4 for 23 scenes."""
    rng = np.random.RandomState(fold_seed)
    order = []
    for k in sorted(set(strata['stratum_of'].values())):
        members = sorted(s for s in scenes if strata['stratum_of'][s] == k)
        rng.shuffle(members); order.extend(members)
    return [sorted(order[f::n_folds]) for f in range(n_folds)]


def make_cv_loaders(root, size, test_scenes, batch_size, num_workers):
    scenes = all_scenes(root, size)
    train_scenes = [s for s in scenes if s not in set(test_scenes)]
    tr = SceneSubsetDataset(root, size, train_scenes); te = SceneSubsetDataset(root, size, test_scenes)
    trl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    tel = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trl, tel, tr, te


# ----------------------------------------------------------------------------
def _gray(x): return x.mean(1, keepdim=True)


def _si_mse(x, xh, eps=1e-12):
    b = x.shape[0]; xf, xhf = x.reshape(b, -1), xh.reshape(b, -1)
    a = (xf * xhf).sum(1) / (xhf.pow(2).sum(1) + eps)
    return ((xf - a[:, None] * xhf) ** 2).mean(1)


def _gain_stats(x, xh):
    """Sufficient statistics for pooled optimal-gain MSE: (sum x*xh, sum xh^2, sum x^2, n)."""
    b = x.shape[0]; xf, xhf = x.reshape(b, -1), xh.reshape(b, -1)
    return (xf * xhf).sum(1), xhf.pow(2).sum(1), xf.pow(2).sum(1), torch.full((b,), xf.shape[1], device=x.device)


def _lmse_chen(x, xh, frac=0.10, eps=1e-12):
    b, _, h, w = x.shape
    k = max(2, int(round(frac * min(h, w)))); step = max(1, k // 2)
    px = Fn.unfold(x, kernel_size=k, stride=step); pxh = Fn.unfold(xh, kernel_size=k, stride=step)
    a = (px * pxh).sum(1) / (pxh.pow(2).sum(1) + eps)
    return ((px - a[:, None, :] * pxh) ** 2).mean(1).mean(1), k, step


def _lmse_grosse(x, xh, k=20, eps=1e-12):
    step = max(1, k // 2)
    px = Fn.unfold(x, kernel_size=k, stride=step); pxh = Fn.unfold(xh, kernel_size=k, stride=step)
    a = (px * pxh).sum(1) / (pxh.pow(2).sum(1) + eps)
    return ((px - a[:, None, :] * pxh) ** 2).sum((1, 2)) / ((px ** 2).sum((1, 2)) + eps)


def _gaussian_window(size=11, sigma=1.5, device='cpu'):
    g = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _ssim(x, xh, data_range=1.0, k1=0.01, k2=0.03):
    win = _gaussian_window(device=x.device); c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    mu_x = Fn.conv2d(x, win); mu_y = Fn.conv2d(xh, win)
    sxx = Fn.conv2d(x * x, win) - mu_x ** 2; syy = Fn.conv2d(xh * xh, win) - mu_y ** 2
    sxy = Fn.conv2d(x * xh, win) - mu_x * mu_y
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean((1, 2, 3))


def _per_sample_mse(a, b): return ((a - b) ** 2).reshape(a.shape[0], -1).mean(1)
def _psnr(mse): return -10.0 * torch.log10(mse + 1e-10)


@torch.no_grad()
def eval_perframe(model, loader, ds, arch):
    model.eval().to(DEVICE)
    meta = ds.meta(); rows = []; batch_rmse = []; i0 = 0; lmse_k = lmse_step = None
    for batch in loader:
        I, R, S, H = [t.to(DEVICE) for t in batch]
        out = model(I)
        if arch == 'direct':
            Rp, Sp = out; Hp = None; recon = Rp * Sp
        else:
            Rp, Sp, Hp = out[0], out[1], out[2]        # proposed returns (R,S,H,logvar)
            recon = Rp * Sp + Hp
        bsz = I.shape[0]
        batch_rmse.append(nn.functional.mse_loss(Rp, R).item())
        r_mse = _per_sample_mse(Rp, R); s_mse = _per_sample_mse(Sp, S); rc_mse = _per_sample_mse(recon, I)
        h_mse = _per_sample_mse(Hp, H) if Hp is not None else None
        gR, gRp, gS, gSp = _gray(R), _gray(Rp), _gray(S), _gray(Sp)
        simse_R_gray = _si_mse(gR, gRp); simse_R_rgb = _si_mse(R, Rp); simse_S_gray = _si_mse(gS, gSp)
        lmse_R, lmse_k, lmse_step = _lmse_chen(gR, gRp); lmse_S, _, _ = _lmse_chen(gS, gSp)
        lmseg_R = _lmse_grosse(gR, gRp)
        dssim_R = (1.0 - _ssim(gR, gRp)) / 2.0; dssim_S = (1.0 - _ssim(gS, gSp)) / 2.0
        xy_rgb, xx_rgb, tt_rgb, n_rgb = _gain_stats(R, Rp)
        xy_g, xx_g, tt_g, n_g = _gain_stats(gR, gRp)
        for j in range(bsz):
            scene, frame, fi = meta[i0 + j]
            row = {'scene': scene, 'frame': frame, 'frame_idx': int(fi),
                   'R_MSE': float(r_mse[j]), 'R_PSNR': float(_psnr(r_mse[j])),
                   'S_MSE': float(s_mse[j]), 'S_PSNR': float(_psnr(s_mse[j])), 'Recon_MSE': float(rc_mse[j]),
                   'simse_R_gray': float(simse_R_gray[j]), 'simse_R_rgb': float(simse_R_rgb[j]),
                   'simse_S_gray': float(simse_S_gray[j]),
                   'lmse_R_chen': float(lmse_R[j]), 'lmse_S_chen': float(lmse_S[j]), 'lmse_R_grosse': float(lmseg_R[j]),
                   'dssim_R': float(dssim_R[j]), 'dssim_S': float(dssim_S[j]),
                   'gain_R_rgb_xy': float(xy_rgb[j]), 'gain_R_rgb_xx': float(xx_rgb[j]), 'gain_R_rgb_tt': float(tt_rgb[j]),
                   'gain_R_rgb_n': int(n_rgb[j]), 'gain_R_gray_xy': float(xy_g[j]), 'gain_R_gray_xx': float(xx_g[j]),
                   'gain_R_gray_tt': float(tt_g[j]), 'gain_R_gray_n': int(n_g[j])}
            if h_mse is not None: row['H_MSE'] = float(h_mse[j])
            rows.append(row)
        i0 += bsz
    assert i0 == len(meta), f"frame count mismatch {i0} vs {len(meta)}"
    skip = {'scene', 'frame', 'frame_idx'}
    mean_keys = [k for k in rows[0] if k not in skip and not k.startswith('gain_')]
    gain_keys = [k for k in rows[0] if k.startswith('gain_')]
    per_scene = {}
    for sc in sorted({r['scene'] for r in rows}):
        rs = [r for r in rows if r['scene'] == sc]
        d = {'n_frames': len(rs)}
        for k in mean_keys: d[k + '_mean'] = float(np.mean([r[k] for r in rs]))
        for k in gain_keys: d[k + '_sum'] = float(np.sum([r[k] for r in rs]))
        d['R_PSNR_of_mean_mse'] = float(-10 * np.log10(d['R_MSE_mean'] + 1e-10))
        d['S_PSNR_of_mean_mse'] = float(-10 * np.log10(d['S_MSE_mean'] + 1e-10))
        # pooled optimal gain over the scene (RGB): MSE_pooled = (tt - xy^2/xx)/n
        for sp in ('rgb', 'gray'):
            xy, xx, tt, n = (d[f'gain_R_{sp}_{q}_sum'] for q in ('xy', 'xx', 'tt', 'n'))
            mse_pooled = max((tt - xy * xy / max(xx, 1e-12)) / max(n, 1), 0.0)
            d[f'R_siPSNR_scenegain_{sp}'] = float(-10 * np.log10(mse_pooled + 1e-10))
        per_scene[sc] = d
    agg = {'n_frames': len(rows), 'n_scenes': len(per_scene)}
    for k in mean_keys: agg[k + '_frame_mean'] = float(np.mean([r[k] for r in rows]))
    agg['R_PSNR_of_frame_mean_mse'] = float(-10 * np.log10(agg['R_MSE_frame_mean'] + 1e-10))
    agg['R_PSNR_batchconv'] = float(B.psnr_from_mse(np.mean(batch_rmse)))
    agg['R_PSNR_scene_equal_weight'] = float(np.mean([d['R_PSNR_of_mean_mse'] for d in per_scene.values()]))
    agg['lmse_window_k'] = int(lmse_k); agg['lmse_window_step'] = int(lmse_step)
    return {'agg': agg, 'per_scene': per_scene, 'per_frame': rows}


# ----------------------------------------------------------------------------
def train_model(arch, trl, epochs):
    if arch == 'direct':
        m = B.DirectCNN(); B.train_2head(m, trl, 'direct', epochs=epochs)
    elif arch == 'unet':
        m = B.UNetPhysics(); B.train_3head(m, trl, 'unet', epochs=epochs)
    elif arch == 'transformer':
        m = TransformerUNet(); B.train_3head(m, trl, 'transformer', epochs=epochs)
    elif arch == 'proposed':
        m = B.ProposedFull(); B.train_proposed(m, trl, 'proposed', epochs=epochs, warmup=int(epochs * 0.7))
    else:
        raise ValueError(arch)
    return m


def build_grid(args):
    archs = [a for a in args.archs.split(',') if a]; seeds = [int(s) for s in args.seeds.split(',') if s]
    s_archs = [a for a in args.strat_archs.split(',') if a]; s_seeds = [int(s) for s in args.strat_seeds.split(',') if s]
    runs = []
    for r in range(args.reps):
        for f in range(args.folds):
            for s in seeds:
                for a in archs:
                    runs.append(dict(design='rand', arch=a, rep=r, fold=f, seed=s))
    for r in range(args.strat_reps):
        for f in range(args.folds):
            for s in s_seeds:
                for a in s_archs:
                    runs.append(dict(design='strat', arch=a, rep=r, fold=f, seed=s))
    return runs


def lpt_assign(runs, n_shards):
    """Deterministic longest-processing-time packing by expected minutes; returns shard index per run."""
    order = sorted(range(len(runs)), key=lambda i: (-EXPECTED_MIN[runs[i]['arch']], i))
    load = [0.0] * n_shards; assign = [None] * len(runs)
    for i in order:
        k = min(range(n_shards), key=lambda j: (load[j], j))
        assign[i] = k; load[k] += EXPECTED_MIN[runs[i]['arch']]
    return assign, load


def tag_of(r):
    return f"cv2_{r['arch']}_{r['design']}r{r['rep']}_f{r['fold']}_s{r['seed']}_{VERSION}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/home/ubuntu/data/training')
    p.add_argument('--out-dir', default='/home/ubuntu/results')
    p.add_argument('--strata', default='strata_v2.json')
    p.add_argument('--archs', default='direct,unet,transformer,proposed')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--strat-archs', default='direct,unet')
    p.add_argument('--strat-reps', type=int, default=2)
    p.add_argument('--strat-seeds', default='42,43')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--s3-prefix', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--plan-only', action='store_true', help='print the shard plan and exit')
    args = p.parse_args()

    grid = build_grid(args)
    assign, load = lpt_assign(grid, args.num_shards)
    mine = [r for r, k in zip(grid, assign) if k == args.shard]
    if args.limit: mine = mine[:args.limit]
    if args.plan_only:
        print(f"{len(grid)} runs, {args.num_shards} shards, expected total {sum(load)/60:.1f} GPU-h, "
              f"max shard {max(load)/60:.2f} h, min {min(load)/60:.2f} h")
        from collections import Counter
        print(Counter(r['arch'] for r in grid), Counter(r['design'] for r in grid))
        return

    strata = json.load(open(args.strata))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scenes = all_scenes(args.data_root, args.size)
    assert set(scenes) == set(strata['stratum_of']), 'strata file does not match the dataset scenes'
    fdoc = out / f'cv_folds_{VERSION}.json'
    if not fdoc.exists():
        doc = {'version': VERSION, 'n_scenes': len(scenes), 'scenes': scenes, 'strata_source': strata.get('metric'),
               'random': {f'rep{r}': {'fold_seed': FOLD_SEED_BASE_RAND + r,
                                      'folds': build_folds_random(scenes, args.folds, FOLD_SEED_BASE_RAND + r)}
                          for r in range(args.reps)},
               'stratified': {f'rep{r}': {'fold_seed': FOLD_SEED_BASE_STRAT + r,
                                          'folds': build_folds_stratified(scenes, args.folds, FOLD_SEED_BASE_STRAT + r, strata)}
                              for r in range(args.strat_reps)}}
        fdoc.write_text(json.dumps(doc, indent=1))
        if args.s3_prefix: os.system(f"aws s3 cp {fdoc} {args.s3_prefix}/{fdoc.name} --quiet")

    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)}/{len(grid)} runs, expected {load[args.shard]/60:.2f} h; "
          f"{len(scenes)} scenes; device={DEVICE}", flush=True)
    for i, r in enumerate(mine):
        tag = tag_of(r); fp = out / f"{tag}.json"
        if fp.exists():
            print(f"skip {tag}", flush=True); continue
        print(f"[{i+1}/{len(mine)}] {tag}", flush=True)
        try:
            if r['design'] == 'rand':
                fold_seed = FOLD_SEED_BASE_RAND + r['rep']; folds = build_folds_random(scenes, args.folds, fold_seed)
            else:
                fold_seed = FOLD_SEED_BASE_STRAT + r['rep']; folds = build_folds_stratified(scenes, args.folds, fold_seed, strata)
            test_scenes = folds[r['fold']]
            B.set_seed(r['seed'])
            trl, tel, tr, te = make_cv_loaders(args.data_root, args.size, test_scenes, args.batch_size, args.num_workers)
            t0 = time.time()
            m = train_model(r['arch'], trl, args.epochs)
            res = eval_perframe(m, tel, te, r['arch'])
            res.update(dict(kind='cv', version=VERSION, design=r['design'], arch=r['arch'], rep=r['rep'], fold=r['fold'],
                            fold_seed=fold_seed, seed=r['seed'], test_scenes=test_scenes, train_scenes=tr.split_scenes,
                            train_n=len(tr), test_n=len(te), epochs=args.epochs, minutes=(time.time() - t0) / 60,
                            device=str(DEVICE), torch=torch.__version__))
            tmp = fp.with_suffix('.json.tmp'); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, fp)
            if args.s3_prefix: os.system(f"aws s3 cp {fp} {args.s3_prefix}/{fp.name} --quiet")
            a = res['agg']
            print(f"   done {tag}: PSNR_bc={a['R_PSNR_batchconv']:.3f} scene_eq={a['R_PSNR_scene_equal_weight']:.3f} "
                  f"simse={a['simse_R_gray_frame_mean']:.5f} lmse={a['lmse_R_chen_frame_mean']:.5f} "
                  f"dssim={a['dssim_R_frame_mean']:.4f} ({res['minutes']:.1f} min)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAILED {tag}: {e}", flush=True)
    print("shard done", flush=True)


if __name__ == '__main__':
    main()
