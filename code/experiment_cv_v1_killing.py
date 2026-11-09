#!/usr/bin/env python3
"""experiment_cv.py — Benchmark-resolution follow-up (pre-registered 2026-11-01).

Pre-registration: PhD/AI/CV/ideas/2026-11-01_benchmark_resolution.md
Parent study:     JOSA A submission 621704 (code: experiment_v8.py / v7_base.py).

Design (killing experiment, v1)
  * Repeated scene-level K-fold cross-validation on MPI Sintel (23 scenes, 1,064 frames).
    R repeats (fold-assignment seeds 201+r) x F folds (np.array_split -> 5,5,5,4,4 scenes)
    x S model seeds x architectures. Every scene is a test scene exactly once per repeat,
    i.e. each scene is scored under R different training compositions.
  * Training recipe identical to the JOSA A protocol study: v7_base.train_2head / train_3head,
    Adam 5e-4 cosine, batch 16, 256x256, 60 epochs, same AMI / T4 hardware.
  * Parity runs: the JOSA A reference configuration (scene partition seed 101, model seed 42)
    is re-run for DirectCNN and UNet so the new per-frame pipeline can be checked against
    results_t4/protocol_{direct,unet}_scene_p101_s42.json.

Per-frame metrics (all computed on the stored 8-bit encoding, as in JOSA A)
  * R_MSE, R_PSNR (= -10 log10 MSE), S_MSE, H_MSE (3-head only), Recon_MSE  — JOSA A conventions.
  * si-MSE  — Grosse et al., ICCV 2009, Sec. 4.2 eq. (7): MSE(x, x_hat) = ||x - a x_hat||^2 with
              a = argmin_a ||x - a x_hat||^2. Implemented as the per-frame *mean* squared error after
              the optimal scalar gain; computed on grayscale (Grosse: gray = mean over channels) and,
              secondarily, on RGB with one shared gain.
  * LMSE (Chen & Koltun, ICCV 2013, Sec. 6 text): "cover the image by overlapping windows of size 10%
              of the image in every dimension, adjust the brightness separately for each window to fit
              the corresponding part of the ground truth image, compute the MSE for each window, and
              average the results."  -> window k = round(0.1*side), stride k//2, mean over windows.
  * LMSE_grosse (Grosse 2009 eq. (8)-(9)): sum over k x k windows (k=20, step k/2) of the
              scale-invariant squared error, normalized by the same sum for the all-zero estimate,
              so that an all-zero estimate scores 1. Kept as the MIT-style variant.
  * DSSIM = (1 - SSIM)/2 (Chen & Koltun 2013). SSIM on grayscale with the Wang et al. 2004 defaults
              (11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03, data range 1); Chen & Koltun do not
              state SSIM parameters, so the defaults are a stated choice, not their convention.

Output: one JSON per run (atomic, skip-if-exists), name cv_<arch>_r<rep>_f<fold>_s<seed>_<version>.json,
containing fold definition, run-level aggregates (incl. the JOSA A batch-mean convention),
per-scene aggregates and the full per-frame table. Fold definitions are also written once to
cv_folds_<version>.json.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import v7_base as B

DEVICE = B.DEVICE
VERSION = 'v1'
FOLD_SEED_BASE = 201          # repeat r uses fold-assignment seed 201 + r
PARITY_PARTITION = 101        # JOSA A reference scene partition
PARITY_SEED = 42              # JOSA A reference model seed


# ----------------------------------------------------------------------------
# Data: explicit scene-list dataset built on the v7_base global cache
# ----------------------------------------------------------------------------
class SceneSubsetDataset(B.SintelDataset):
    """Frames of an explicit scene list, in cache order (scene, frame name)."""

    def __init__(self, root, size, scenes_subset):           # noqa: D107
        self.size = size
        self.protocol = 'cvfold'
        self.split = 'custom'
        cache = self._get_global_cache(root, size)
        self._cache = cache
        picked = set(scenes_subset)
        self.split_scenes = sorted(picked)
        self.indices = [i for i, (s, _f, _fi) in enumerate(cache['meta'])
                        if s in picked]

    def meta(self):
        return [self._cache['meta'][i] for i in self.indices]


def build_folds(scenes, n_folds, fold_seed):
    rng = np.random.RandomState(fold_seed)
    order = list(scenes)
    rng.shuffle(order)
    parts = np.array_split(np.array(order, dtype=object), n_folds)
    return [sorted(p.tolist()) for p in parts]


def all_scenes(root, size):
    return list(B.SintelDataset._get_global_cache(root, size)['scenes'])


def make_cv_loaders(root, size, test_scenes, batch_size, num_workers):
    scenes = all_scenes(root, size)
    train_scenes = [s for s in scenes if s not in set(test_scenes)]
    tr = SceneSubsetDataset(root, size, train_scenes)
    te = SceneSubsetDataset(root, size, test_scenes)
    trl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)
    tel = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)
    return trl, tel, tr, te


# ----------------------------------------------------------------------------
# Metrics (batched, torch)
# ----------------------------------------------------------------------------
def _gray(x):
    return x.mean(1, keepdim=True)


def _si_mse(x, xh, eps=1e-12):
    """Per-sample mean squared error after the optimal scalar gain (Grosse 2009 eq. 7)."""
    b = x.shape[0]
    xf, xhf = x.reshape(b, -1), xh.reshape(b, -1)
    a = (xf * xhf).sum(1) / (xhf.pow(2).sum(1) + eps)
    return ((xf - a[:, None] * xhf) ** 2).mean(1)


def _lmse_chen(x, xh, frac=0.10, eps=1e-12):
    """Chen & Koltun 2013 LMSE: windows of 10% of each dimension, per-window gain, mean over windows.
    x, xh: [B,1,H,W] grayscale."""
    b, _, h, w = x.shape
    k = max(2, int(round(frac * min(h, w))))
    step = max(1, k // 2)
    px = Fn.unfold(x, kernel_size=k, stride=step)        # [B, k*k, nW]
    pxh = Fn.unfold(xh, kernel_size=k, stride=step)
    a = (px * pxh).sum(1) / (pxh.pow(2).sum(1) + eps)   # [B, nW]
    mse_w = ((px - a[:, None, :] * pxh) ** 2).mean(1)    # [B, nW]
    return mse_w.mean(1), k, step


def _lmse_grosse(x, xh, k=20, eps=1e-12):
    """Grosse 2009 eq. (8)-(9): sum_w ||x_w - a_w xh_w||^2 / sum_w ||x_w||^2 (all-zero estimate -> 1)."""
    step = max(1, k // 2)
    px = Fn.unfold(x, kernel_size=k, stride=step)
    pxh = Fn.unfold(xh, kernel_size=k, stride=step)
    a = (px * pxh).sum(1) / (pxh.pow(2).sum(1) + eps)
    num = ((px - a[:, None, :] * pxh) ** 2).sum((1, 2))
    den = (px ** 2).sum((1, 2)) + eps
    return num / den


def _gaussian_window(size=11, sigma=1.5, device='cpu'):
    g = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]        # [1,1,size,size]


def _ssim(x, xh, data_range=1.0, k1=0.01, k2=0.03):
    """Wang et al. 2004 SSIM, grayscale [B,1,H,W], valid convolution, mean over map."""
    win = _gaussian_window(device=x.device)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    mu_x = Fn.conv2d(x, win)
    mu_y = Fn.conv2d(xh, win)
    sxx = Fn.conv2d(x * x, win) - mu_x ** 2
    syy = Fn.conv2d(xh * xh, win) - mu_y ** 2
    sxy = Fn.conv2d(x * xh, win) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return (num / den).mean((1, 2, 3))


def _per_sample_mse(a, b):
    return ((a - b) ** 2).reshape(a.shape[0], -1).mean(1)


def _psnr(mse):
    return -10.0 * torch.log10(mse + 1e-10)


@torch.no_grad()
def eval_perframe(model, loader, ds, arch):
    """Per-frame metric table + per-scene aggregates + run aggregates (incl. JOSA A batch convention)."""
    model.eval().to(DEVICE)
    meta = ds.meta()
    rows = []
    batch_rmse = []                      # JOSA A convention: mean over batch-mean MSEs -> PSNR
    i0 = 0
    lmse_k = lmse_step = None
    for batch in loader:
        I, R, S, H = [t.to(DEVICE) for t in batch]
        if arch == 'direct':
            Rp, Sp = model(I)
            Hp = None
            recon = Rp * Sp
        else:
            Rp, Sp, Hp = model(I)
            recon = Rp * Sp + Hp
        bsz = I.shape[0]
        batch_rmse.append(nn.functional.mse_loss(Rp, R).item())
        r_mse = _per_sample_mse(Rp, R)
        s_mse = _per_sample_mse(Sp, S)
        rc_mse = _per_sample_mse(recon, I)
        h_mse = _per_sample_mse(Hp, H) if Hp is not None else None
        gR, gRp = _gray(R), _gray(Rp)
        gS, gSp = _gray(S), _gray(Sp)
        simse_R_gray = _si_mse(gR, gRp)
        simse_R_rgb = _si_mse(R, Rp)
        simse_S_gray = _si_mse(gS, gSp)
        lmse_R, lmse_k, lmse_step = _lmse_chen(gR, gRp)
        lmse_S, _, _ = _lmse_chen(gS, gSp)
        lmseg_R = _lmse_grosse(gR, gRp)
        dssim_R = (1.0 - _ssim(gR, gRp)) / 2.0
        dssim_S = (1.0 - _ssim(gS, gSp)) / 2.0
        for j in range(bsz):
            scene, frame, fi = meta[i0 + j]
            row = {
                'scene': scene, 'frame': frame, 'frame_idx': int(fi),
                'R_MSE': float(r_mse[j]), 'R_PSNR': float(_psnr(r_mse[j])),
                'S_MSE': float(s_mse[j]), 'S_PSNR': float(_psnr(s_mse[j])),
                'Recon_MSE': float(rc_mse[j]),
                'simse_R_gray': float(simse_R_gray[j]), 'simse_R_rgb': float(simse_R_rgb[j]),
                'simse_S_gray': float(simse_S_gray[j]),
                'lmse_R_chen': float(lmse_R[j]), 'lmse_S_chen': float(lmse_S[j]),
                'lmse_R_grosse': float(lmseg_R[j]),
                'dssim_R': float(dssim_R[j]), 'dssim_S': float(dssim_S[j]),
            }
            if h_mse is not None:
                row['H_MSE'] = float(h_mse[j])
            rows.append(row)
        i0 += bsz
    assert i0 == len(meta), f"frame count mismatch {i0} vs {len(meta)}"

    keys = [k for k in rows[0].keys() if k not in ('scene', 'frame', 'frame_idx')]
    per_scene = {}
    for sc in sorted({r['scene'] for r in rows}):
        rs = [r for r in rows if r['scene'] == sc]
        d = {'n_frames': len(rs)}
        for k in keys:
            d[k + '_mean'] = float(np.mean([r[k] for r in rs]))
        d['R_PSNR_of_mean_mse'] = float(-10.0 * np.log10(d['R_MSE_mean'] + 1e-10))
        d['S_PSNR_of_mean_mse'] = float(-10.0 * np.log10(d['S_MSE_mean'] + 1e-10))
        per_scene[sc] = d
    agg = {'n_frames': len(rows), 'n_scenes': len(per_scene)}
    for k in keys:
        agg[k + '_frame_mean'] = float(np.mean([r[k] for r in rows]))
    agg['R_PSNR_of_frame_mean_mse'] = float(-10.0 * np.log10(agg['R_MSE_frame_mean'] + 1e-10))
    agg['R_PSNR_batchconv'] = float(B.psnr_from_mse(np.mean(batch_rmse)))   # JOSA A convention
    agg['R_PSNR_scene_equal_weight'] = float(np.mean([d['R_PSNR_of_mean_mse'] for d in per_scene.values()]))
    agg['lmse_window_k'] = int(lmse_k)
    agg['lmse_window_step'] = int(lmse_step)
    return {'agg': agg, 'per_scene': per_scene, 'per_frame': rows}


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
def build_grid(archs, reps, folds, seeds, parity):
    runs = []
    for r in range(reps):
        for f in range(folds):
            for s in seeds:
                for a in archs:
                    runs.append(dict(kind='cv', arch=a, rep=r, fold=f, seed=s))
    if parity:
        for a in archs:
            runs.append(dict(kind='parity', arch=a, rep=-1, fold=-1, seed=PARITY_SEED,
                             partition=PARITY_PARTITION))
    return runs


def train_model(arch, trl, epochs):
    if arch == 'direct':
        m = B.DirectCNN(); B.train_2head(m, trl, 'direct', epochs=epochs)
    elif arch == 'unet':
        m = B.UNetPhysics(); B.train_3head(m, trl, 'unet', epochs=epochs)
    else:
        raise ValueError(arch)
    return m


def run_cv_one(r, args, scenes):
    folds = build_folds(scenes, args.folds, FOLD_SEED_BASE + r['rep'])
    test_scenes = folds[r['fold']]
    B.set_seed(r['seed'])
    trl, tel, tr, te = make_cv_loaders(args.data_root, args.size, test_scenes,
                                       args.batch_size, args.num_workers)
    t0 = time.time()
    m = train_model(r['arch'], trl, args.epochs)
    res = eval_perframe(m, tel, te, r['arch'])
    res.update(dict(kind='cv', version=VERSION, arch=r['arch'], rep=r['rep'], fold=r['fold'],
                    fold_seed=FOLD_SEED_BASE + r['rep'], seed=r['seed'],
                    test_scenes=test_scenes, train_scenes=tr.split_scenes,
                    train_n=len(tr), test_n=len(te), epochs=args.epochs,
                    minutes=(time.time() - t0) / 60, device=str(DEVICE)))
    return res


def run_parity_one(r, args):
    """JOSA A reference configuration through the original loaders + the new per-frame evaluator."""
    B.set_seed(r['seed'])
    trl, tel, ntr, nte = B.make_loaders(args.data_root, 'scene', r['partition'],
                                        batch_size=args.batch_size, size=args.size,
                                        num_workers=args.num_workers)
    te = tel.dataset
    t0 = time.time()
    m = train_model(r['arch'], trl, args.epochs)
    orig = B.eval_2head(m, tel, r['arch']) if r['arch'] == 'direct' else B.eval_3head(m, tel, r['arch'])
    te.meta = lambda: [te._cache['meta'][i] for i in te.indices]      # same interface as SceneSubsetDataset
    res = eval_perframe(m, tel, te, r['arch'])
    res.update(dict(kind='parity', version=VERSION, arch=r['arch'], partition=r['partition'],
                    seed=r['seed'], original_eval=orig, test_scenes=sorted(te.split_scenes),
                    train_n=ntr, test_n=nte, epochs=args.epochs,
                    minutes=(time.time() - t0) / 60, device=str(DEVICE)))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/home/ubuntu/data/training')
    p.add_argument('--out-dir', default='/home/ubuntu/results')
    p.add_argument('--archs', default='direct,unet')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', default='42,43')
    p.add_argument('--no-parity', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--s3-prefix', default='')
    p.add_argument('--limit', type=int, default=0, help='debug: run at most N runs')
    args = p.parse_args()

    archs = [a for a in args.archs.split(',') if a]
    seeds = [int(s) for s in args.seeds.split(',') if s]
    grid = build_grid(archs, args.reps, args.folds, seeds, parity=not args.no_parity)
    mine = [r for i, r in enumerate(grid) if i % args.num_shards == args.shard]
    if args.limit:
        mine = mine[:args.limit]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    scenes = all_scenes(args.data_root, args.size)
    folds_doc = {f'rep{r}': {'fold_seed': FOLD_SEED_BASE + r,
                             'folds': build_folds(scenes, args.folds, FOLD_SEED_BASE + r)}
                 for r in range(args.reps)}
    fdoc = out / f'cv_folds_{VERSION}.json'
    if not fdoc.exists():
        fdoc.write_text(json.dumps({'version': VERSION, 'n_scenes': len(scenes), 'scenes': scenes,
                                    'repeats': folds_doc}, indent=1))
        if args.s3_prefix:
            os.system(f"aws s3 cp {fdoc} {args.s3_prefix}/{fdoc.name} --quiet")

    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)}/{len(grid)} runs; "
          f"{len(scenes)} scenes; device={DEVICE}", flush=True)
    for i, r in enumerate(mine):
        if r['kind'] == 'cv':
            tag = f"cv_{r['arch']}_r{r['rep']}_f{r['fold']}_s{r['seed']}_{VERSION}"
        else:
            tag = f"parity_{r['arch']}_scene_p{r['partition']}_s{r['seed']}_{VERSION}"
        fp = out / f"{tag}.json"
        if fp.exists():
            print(f"skip {tag}", flush=True); continue
        print(f"[{i+1}/{len(mine)}] {tag}", flush=True)
        try:
            res = run_cv_one(r, args, scenes) if r['kind'] == 'cv' else run_parity_one(r, args)
            tmp = fp.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(res, indent=1))
            os.replace(tmp, fp)
            if args.s3_prefix:
                os.system(f"aws s3 cp {fp} {args.s3_prefix}/{fp.name} --quiet")
            a = res['agg']
            print(f"   done {tag}: R_PSNR_batchconv={a['R_PSNR_batchconv']:.3f} "
                  f"scene_eq={a['R_PSNR_scene_equal_weight']:.3f} simse={a['simse_R_gray_frame_mean']:.5f} "
                  f"lmse={a['lmse_R_chen_frame_mean']:.5f} dssim={a['dssim_R_frame_mean']:.4f} "
                  f"({res['minutes']:.1f} min)", flush=True)
        except Exception as e:            # keep the shard alive; failures are visible in the log
            import traceback; traceback.print_exc()
            print(f"FAILED {tag}: {e}", flush=True)
    print("shard done", flush=True)


if __name__ == '__main__':
    main()
