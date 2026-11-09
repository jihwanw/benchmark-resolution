#!/usr/bin/env python3
"""qual_intrinsic.py — retrain representative intrinsic configurations and SAVE PREDICTIONS for qualitative figures.
Quantitative claims rest on the grid (main_v2.json); these runs only produce images. Same recipe as experiment_cv2.py.

Jobs (all seed 42, 60 epochs, fold-assignment seeds 201+rep as in the grid):
  gauge : DirectCNN trained on each of the 5 training compositions in which SCENE_GAUGE is held out -> its reflectance
          under 5 training sets (illustrates the per-scene gain interaction, H4).
  inter : DirectCNN and UNet trained on rep 0 fold 3 (held-out: ambush_4, ambush_5, sleeping_2, temple_2) -> hard vs easy scene.
  thumbs: one clean/final frame per scene (middle frame) for the 23-scene mosaic.
Output: qual_intrinsic.npz (+ qual_intrinsic_meta.json) uploaded to --s3-prefix.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch
import v7_base as B
from experiment_cv2 import SceneSubsetDataset, build_folds_random, all_scenes, make_cv_loaders, train_model, FOLD_SEED_BASE_RAND

SCENE_GAUGE = 'bamboo_2'
INTER_REP, INTER_FOLD = 0, 3
INTER_SCENES = ['ambush_5', 'sleeping_2']
FRAMES_PER_SCENE = [10, 30]          # frame index within scene (0-based in cache order)


def predict_scene(model, arch, ds, scene, frame_idxs, device):
    """Return dict of arrays for selected frames of `scene` from dataset `ds` (test fold)."""
    model.eval().to(device); out = {}
    meta = ds.meta(); idx_by_scene = [i for i, (s, _f, _fi) in enumerate(meta) if s == scene]
    for k in frame_idxs:
        k = min(k, len(idx_by_scene) - 1); i = idx_by_scene[k]; F, R, S, H = ds[i]
        with torch.no_grad():
            o = model(F[None].to(device))
            Rp, Sp = o[0][0].cpu(), o[1][0].cpu()
        out[f'{scene}/f{k}/input'] = F.numpy(); out[f'{scene}/f{k}/gt_R'] = R.numpy(); out[f'{scene}/f{k}/gt_S'] = S.numpy()
        out[f'{scene}/f{k}/pred_R'] = Rp.numpy(); out[f'{scene}/f{k}/pred_S'] = Sp.numpy(); out[f'{scene}/f{k}/frame'] = np.array(meta[i][1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/home/ubuntu/data/training'); p.add_argument('--out-dir', default='/home/ubuntu/results_qual')
    p.add_argument('--s3-prefix', default=''); p.add_argument('--epochs', type=int, default=60); p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--num-workers', type=int, default=4); p.add_argument('--jobs', default='thumbs,gauge,inter')
    args = p.parse_args(); device = B.DEVICE
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scenes = all_scenes(args.data_root, args.size); arrays = {}; meta = {'jobs': {}, 'seed': 42, 'epochs': args.epochs}
    cache = B.SintelDataset._get_global_cache(args.data_root, args.size)
    if 'thumbs' in args.jobs:
        for sc in scenes:
            idx = [i for i, (s, _f, _fi) in enumerate(cache['meta']) if s == sc]; i = idx[len(idx) // 2]
            arrays[f'thumb/{sc}/final'] = cache['F'][i].numpy(); arrays[f'thumb/{sc}/albedo'] = cache['R'][i].numpy(); arrays[f'thumb/{sc}/frame'] = np.array(cache['meta'][i][1])
        meta['jobs']['thumbs'] = {'n_scenes': len(scenes), 'frame': 'middle frame of each scene, final pass + albedo pass, 256x256'}
    if 'gauge' in args.jobs:
        meta['jobs']['gauge'] = {'scene': SCENE_GAUGE, 'arch': 'direct', 'compositions': []}
        for rep in range(5):
            folds = build_folds_random(scenes, 5, FOLD_SEED_BASE_RAND + rep); f = next(i for i, fo in enumerate(folds) if SCENE_GAUGE in fo)
            B.set_seed(42); trl, tel, tr, te = make_cv_loaders(args.data_root, args.size, folds[f], args.batch_size, args.num_workers)
            t0 = time.time(); m = train_model('direct', trl, args.epochs)
            pred = predict_scene(m, 'direct', te, SCENE_GAUGE, FRAMES_PER_SCENE, device)
            for k, v in pred.items(): arrays[f'gauge/rep{rep}/' + k] = v
            # scene-level numbers for the caption: PSNR and scene-gain PSNR over all frames of the scene
            idx = [i for i, (s, _f, _fi) in enumerate(te.meta()) if s == SCENE_GAUGE]; mse = []; xy = xx = tt = n = 0.0
            m.eval()
            with torch.no_grad():
                for i in idx:
                    F, R, S, H = te[i]; Rp = m(F[None].to(device))[0][0].cpu(); mse.append(float(((Rp - R) ** 2).mean()))
                    xy += float((R * Rp).sum()); xx += float((Rp * Rp).sum()); tt += float((R * R).sum()); n += R.numel()
            psnr = -10 * np.log10(np.mean(mse) + 1e-10); gain_psnr = -10 * np.log10(max((tt - xy * xy / max(xx, 1e-12)) / n, 0) + 1e-10)
            meta['jobs']['gauge']['compositions'].append({'rep': rep, 'fold': f, 'test_scenes': folds[f], 'train_scenes': tr.split_scenes,
                                                           'bamboo_1_in_train': 'bamboo_1' in tr.split_scenes, 'scene_psnr': psnr, 'scene_gain_psnr': gain_psnr,
                                                           'optimal_gain': xy / max(xx, 1e-12), 'minutes': (time.time() - t0) / 60})
            print(f"gauge rep{rep} fold{f}: PSNR {psnr:.2f} gain-aligned {gain_psnr:.2f} gain {xy/max(xx,1e-12):.3f} bamboo_1_in_train={'bamboo_1' in tr.split_scenes} ({(time.time()-t0)/60:.1f} min)", flush=True)
    if 'inter' in args.jobs:
        folds = build_folds_random(scenes, 5, FOLD_SEED_BASE_RAND + INTER_REP); test_scenes = folds[INTER_FOLD]
        meta['jobs']['inter'] = {'rep': INTER_REP, 'fold': INTER_FOLD, 'test_scenes': test_scenes, 'scenes_shown': INTER_SCENES, 'per_scene_psnr': {}}
        for arch in ['direct', 'unet']:
            B.set_seed(42); trl, tel, tr, te = make_cv_loaders(args.data_root, args.size, test_scenes, args.batch_size, args.num_workers)
            t0 = time.time(); m = train_model(arch, trl, args.epochs)
            for sc in INTER_SCENES:
                pred = predict_scene(m, arch, te, sc, FRAMES_PER_SCENE, device)
                for k, v in pred.items(): arrays[f'inter/{arch}/' + k] = v
                idx = [i for i, (s, _f, _fi) in enumerate(te.meta()) if s == sc]; mse = []
                m.eval()
                with torch.no_grad():
                    for i in idx:
                        F, R, S, H = te[i]; Rp = m(F[None].to(device))[0][0].cpu(); mse.append(float(((Rp - R) ** 2).mean()))
                meta['jobs']['inter']['per_scene_psnr'][f'{arch}/{sc}'] = float(-10 * np.log10(np.mean(mse) + 1e-10))
            print(f"inter {arch}: " + ', '.join(f"{k}={v:.2f}" for k, v in meta['jobs']['inter']['per_scene_psnr'].items() if k.startswith(arch)) + f" ({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez_compressed(out / 'qual_intrinsic.npz', **arrays); (out / 'qual_intrinsic_meta.json').write_text(json.dumps(meta, indent=1))
    if args.s3_prefix:
        os.system(f"aws s3 cp {out / 'qual_intrinsic.npz'} {args.s3_prefix}/qual_intrinsic.npz --quiet"); os.system(f"aws s3 cp {out / 'qual_intrinsic_meta.json'} {args.s3_prefix}/qual_intrinsic_meta.json --quiet")
    print('QUAL_INTRINSIC_DONE', len(arrays), 'arrays', flush=True)


if __name__ == '__main__':
    main()
