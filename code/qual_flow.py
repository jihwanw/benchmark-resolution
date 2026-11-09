#!/usr/bin/env python3
"""qual_flow.py — retrain RAFT-small and RAFT-large on one grid composition (rep 0, fold 3: held-out ambush_4, ambush_5,
sleeping_2, temple_2) with the grid recipe and SAVE PREDICTIONS for two scenes (hard: ambush_4, easy: sleeping_2) for a
qualitative figure. Also stores zero-shot predictions. Output: qual_flow.npz + qual_flow_meta.json -> --s3-prefix."""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch
from experiment_flow import list_pairs, build_folds_random, SintelFlowPairs, make_model, train, pad8, FOLD_SEED_BASE_RAND

REP, FOLD = 0, 3
SCENES_SHOWN = ['ambush_4', 'sleeping_2']
PAIR_IDX = [10, 30]        # pair index within scene


@torch.no_grad()
def predict_pairs(model, root, pairs, device, flow_updates=12):
    model.eval().to(device); out = {}
    ds = SintelFlowPairs(root, pairs, train=False)
    for i, (sc, a, b) in enumerate(pairs):
        img1, img2, flow, valid, occ, _ = ds[i]
        (p1, pad), (p2, _) = pad8(img1[None].to(device)), pad8(img2[None].to(device))
        pred = model(p1, p2, num_flow_updates=flow_updates)[-1][0, :, :img1.shape[-2], :img1.shape[-1]].cpu()
        key = f'{sc}/{Path(a).stem}'
        out[key + '/img1'] = ((img1 + 1) * 127.5).clamp(0, 255).byte().numpy(); out[key + '/gt'] = flow.numpy(); out[key + '/valid'] = valid.numpy(); out[key + '/occ'] = occ.numpy()
        out[key + '/pred'] = pred.numpy(); out[key + '/epe'] = (pred - flow).norm(dim=0).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/home/ubuntu/data/training'); p.add_argument('--out-dir', default='/home/ubuntu/results_qual')
    p.add_argument('--s3-prefix', default=''); p.add_argument('--iters', type=int, default=3000); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--crop', default='320,448'); p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(args.data_root); scenes = sorted({s for s, _, _ in pairs}); crop = tuple(int(x) for x in args.crop.split(','))
    folds = build_folds_random(scenes, 5, FOLD_SEED_BASE_RAND + REP); test_scenes = folds[FOLD]
    tr_pairs = [q for q in pairs if q[0] not in set(test_scenes)]
    shown = []
    for sc in SCENES_SHOWN:
        sp = [q for q in pairs if q[0] == sc]; shown += [sp[min(k, len(sp) - 1)] for k in PAIR_IDX]
    arrays = {}; meta = {'rep': REP, 'fold': FOLD, 'test_scenes': test_scenes, 'scenes_shown': SCENES_SHOWN, 'pairs_shown': [f'{s}/{a}' for s, a, b in shown], 'seed': 42, 'iters': args.iters, 'models': {}}
    for arch in ['raft_small', 'raft_large']:
        zs = make_model(arch, pretrained=True)
        for k, v in predict_pairs(zs, args.data_root, shown, device).items(): arrays[f'zeroshot/{arch}/{k}'] = v
        torch.manual_seed(42); np.random.seed(42); g = torch.Generator(); g.manual_seed(42)
        ds = SintelFlowPairs(args.data_root, tr_pairs, train=True, crop=crop, seed=42)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, generator=g, pin_memory=True)
        t0 = time.time(); m = make_model(arch, pretrained=True); train(m, dl, args.iters, device)
        pr = predict_pairs(m, args.data_root, shown, device)
        for k, v in pr.items(): arrays[f'finetuned/{arch}/{k}'] = v
        # scene-level EPE on all pairs of the shown scenes (for the caption)
        per_scene = {}
        for sc in SCENES_SHOWN:
            sp = [q for q in pairs if q[0] == sc]; e = predict_pairs(m, args.data_root, sp, device)
            per_scene[sc] = float(np.mean([e[f'{s}/{Path(a).stem}/epe'][e[f'{s}/{Path(a).stem}/valid']].mean() for s, a, b in sp]))
        meta['models'][arch] = {'minutes': (time.time() - t0) / 60, 'scene_epe_finetuned': per_scene}
        print(f"{arch}: " + ', '.join(f"{k} EPE {v:.3f}" for k, v in per_scene.items()) + f" ({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez_compressed(out / 'qual_flow.npz', **arrays); (out / 'qual_flow_meta.json').write_text(json.dumps(meta, indent=1))
    if args.s3_prefix:
        os.system(f"aws s3 cp {out / 'qual_flow.npz'} {args.s3_prefix}/qual_flow.npz --quiet"); os.system(f"aws s3 cp {out / 'qual_flow_meta.json'} {args.s3_prefix}/qual_flow_meta.json --quiet")
    print('QUAL_FLOW_DONE', len(arrays), 'arrays', flush=True)


if __name__ == '__main__':
    main()
