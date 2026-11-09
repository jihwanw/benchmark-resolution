#!/usr/bin/env python3
"""experiment_flow.py — second benchmark arm: Sintel optical flow, scene-level repeated 5-fold CV (v1).

Pre-registration: PhD/AI/CV/ideas/2026-11-01_benchmark_resolution.md ("두 번째 벤치마크 arm — Sintel optical flow").
Folds: identical to the intrinsic arm (build_folds_random with fold seeds 201..205 over the sorted scene list).

Data layout expected under --data-root (MPI-Sintel training):
  clean/<scene>/frame_XXXX.png        (from MPI-Sintel-training_images.zip; same tar as the intrinsic arm)
  flow/<scene>/frame_XXXX.flo         (MPI-Sintel-training_extras.zip)
  invalid/<scene>/frame_XXXX.png      (extras; nonzero = invalid pixel)
  occlusions/<scene>/frame_XXXX.png   (extras; nonzero = occluded)
Pair t uses frame_t (img1), frame_{t+1} (img2), flow/frame_t.flo, invalid/frame_t.png, occlusions/frame_t.png.

Model: torchvision raft_small / raft_large initialised from C_T_V2 weights (FlyingChairs+FlyingThings; no Sintel),
fine-tuned for a fixed budget. Minimal augmentation (random crop, horizontal flip). Sequence loss gamma=0.8, max_flow=400,
invalid pixels excluded (conventions of the torchvision optical-flow reference script).

Output: one JSON per run with per-pair metrics (EPE, outlier rates, Fl-all, occ/noc EPE, GT magnitude, valid fraction),
per-scene aggregates (pair-weighted and pixel-weighted), run aggregates, fold definition, timing.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

VERSION = 'v1'
FOLD_SEED_BASE_RAND = 201
MAX_FLOW = 400.0
EXPECTED_MIN = {'raft_small': 25.0, 'raft_large': 60.0}     # placeholders; overwritten by --expected-minutes


# ----------------------------------------------------------------------------
def read_flo(path):
    """Middlebury .flo: 4-byte tag (float 202021.25 == 'PIEH'), int32 W, int32 H, then H*W*2 float32 (u,v interleaved)."""
    with open(path, 'rb') as f:
        tag = np.frombuffer(f.read(4), dtype=np.float32)[0]
        assert abs(tag - 202021.25) < 1e-3, f'bad .flo tag {tag} in {path}'
        w = int(np.frombuffer(f.read(4), dtype=np.int32)[0]); h = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        data = np.frombuffer(f.read(h * w * 2 * 4), dtype=np.float32).reshape(h, w, 2)
    return data.copy()


def list_pairs(root):
    """Return sorted list of (scene, frame_name_t, frame_name_t1) for which img1, img2, flow exist."""
    clean = Path(root) / 'clean'; flow = Path(root) / 'flow'
    pairs = []
    for sc in sorted(p.name for p in clean.iterdir() if p.is_dir()):
        frames = sorted(p.name for p in (clean / sc).iterdir() if p.suffix == '.png')
        for a, b in zip(frames[:-1], frames[1:]):
            if (flow / sc / (Path(a).stem + '.flo')).exists():
                pairs.append((sc, a, b))
    return pairs


class SintelFlowPairs(torch.utils.data.Dataset):
    def __init__(self, root, pairs, train, crop=(320, 448), seed=0):
        self.root = Path(root); self.pairs = pairs; self.train = train; self.crop = crop
        self.rng = np.random.RandomState(seed)

    def __len__(self): return len(self.pairs)

    def _load(self, sc, a, b):
        img1 = np.asarray(Image.open(self.root / 'clean' / sc / a).convert('RGB'), dtype=np.float32)
        img2 = np.asarray(Image.open(self.root / 'clean' / sc / b).convert('RGB'), dtype=np.float32)
        flow = read_flo(self.root / 'flow' / sc / (Path(a).stem + '.flo'))
        inv_p = self.root / 'invalid' / sc / a
        invalid = (np.asarray(Image.open(inv_p).convert('L')) > 0) if inv_p.exists() else np.zeros(flow.shape[:2], bool)
        occ_p = self.root / 'occlusions' / sc / a
        occ = (np.asarray(Image.open(occ_p).convert('L')) > 0) if occ_p.exists() else np.zeros(flow.shape[:2], bool)
        return img1, img2, flow, invalid, occ

    def __getitem__(self, i):
        sc, a, b = self.pairs[i]
        img1, img2, flow, invalid, occ = self._load(sc, a, b)
        if self.train:
            H, W = flow.shape[:2]; ch, cw = self.crop
            y0 = self.rng.randint(0, H - ch + 1); x0 = self.rng.randint(0, W - cw + 1)
            img1 = img1[y0:y0 + ch, x0:x0 + cw]; img2 = img2[y0:y0 + ch, x0:x0 + cw]
            flow = flow[y0:y0 + ch, x0:x0 + cw]; invalid = invalid[y0:y0 + ch, x0:x0 + cw]
            if self.rng.rand() < 0.5:                          # horizontal flip: negate u
                img1 = img1[:, ::-1]; img2 = img2[:, ::-1]; flow = flow[:, ::-1] * np.array([-1.0, 1.0], np.float32); invalid = invalid[:, ::-1]
        t = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1) / 127.5 - 1.0     # [-1,1] as torchvision RAFT expects
        fl = torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1)
        valid = torch.from_numpy(np.ascontiguousarray(~invalid)) & (fl.norm(dim=0) < MAX_FLOW)
        return t(img1), t(img2), fl, valid, torch.from_numpy(np.ascontiguousarray(occ)), i


def build_folds_random(scenes, n_folds, fold_seed):
    rng = np.random.RandomState(fold_seed); order = list(scenes); rng.shuffle(order)
    return [sorted(p.tolist()) for p in np.array_split(np.array(order, dtype=object), n_folds)]


# ----------------------------------------------------------------------------
def make_model(arch, pretrained=True):
    from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
    if arch == 'raft_small':
        return raft_small(weights=Raft_Small_Weights.C_T_V2 if pretrained else None)
    if arch == 'raft_large':
        return raft_large(weights=Raft_Large_Weights.C_T_V2 if pretrained else None)
    raise ValueError(arch)


def sequence_loss(preds, gt, valid, gamma=0.8):
    n = len(preds); loss = 0.0
    for i, p in enumerate(preds):
        w = gamma ** (n - 1 - i)
        l1 = (p - gt).abs().sum(1)                     # [B,H,W]
        loss = loss + w * (l1 * valid).sum() / valid.sum().clamp(min=1)
    return loss


def pad8(x):
    h, w = x.shape[-2:]; ph = (8 - h % 8) % 8; pw = (8 - w % 8) % 8
    return Fn.pad(x, (0, pw, 0, ph), mode='replicate'), (ph, pw)


def train(model, loader, iters, device, lr=1e-4, wd=1e-5, log_every=250, flow_updates=12):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters + 10, pct_start=0.05, anneal_strategy='linear')
    model.train().to(device); it = 0; t0 = time.time(); run_loss = 0.0
    while it < iters:
        for img1, img2, flow, valid, _occ, _i in loader:
            img1, img2, flow, valid = img1.to(device), img2.to(device), flow.to(device), valid.to(device)
            opt.zero_grad(set_to_none=True)
            preds = model(img1, img2, num_flow_updates=flow_updates)
            loss = sequence_loss(preds, flow, valid.float())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            it += 1; run_loss += loss.item()
            if it % log_every == 0:
                print(f"  it {it}/{iters} loss {run_loss/log_every:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True); run_loss = 0.0
            if it >= iters: break
    return model


@torch.no_grad()
def evaluate(model, loader, pairs, device, flow_updates=12):
    model.eval().to(device); rows = []
    for img1, img2, flow, valid, occ, idx in loader:
        img1, img2 = img1.to(device), img2.to(device)
        (p1, pad), (p2, _) = pad8(img1), pad8(img2)
        pred = model(p1, p2, num_flow_updates=flow_updates)[-1]
        ph, pw = pad; H, W = img1.shape[-2:]; pred = pred[..., :H, :W].cpu()
        for j in range(img1.shape[0]):
            e = (pred[j] - flow[j]).norm(dim=0); v = valid[j]; o = occ[j] & v; noc = (~occ[j]) & v
            mag = flow[j].norm(dim=0)
            ev = e[v]; nv = int(v.sum())
            sc, a, b = pairs[int(idx[j])]
            rows.append({'scene': sc, 'frame': a, 'frame2': b, 'n_valid': nv, 'valid_frac': float(v.float().mean()),
                         'epe': float(ev.mean()) if nv else float('nan'), 'epe_sum': float(ev.sum()),
                         'out1': float((ev > 1).float().mean()) if nv else float('nan'),
                         'out3': float((ev > 3).float().mean()) if nv else float('nan'),
                         'out5': float((ev > 5).float().mean()) if nv else float('nan'),
                         'fl_all': float(((ev > 3) & (ev > 0.05 * mag[v])).float().mean()) if nv else float('nan'),
                         'epe_occ': float(e[o].mean()) if int(o.sum()) else float('nan'), 'n_occ': int(o.sum()),
                         'epe_noc': float(e[noc].mean()) if int(noc.sum()) else float('nan'),
                         'gt_mag_mean': float(mag[v].mean()) if nv else float('nan')})
    per_scene = {}
    for sc in sorted({r['scene'] for r in rows}):
        rs = [r for r in rows if r['scene'] == sc]
        d = {'n_pairs': len(rs)}
        for k in ['epe', 'out1', 'out3', 'out5', 'fl_all', 'epe_occ', 'epe_noc', 'gt_mag_mean', 'valid_frac']:
            d[k + '_mean'] = float(np.nanmean([r[k] for r in rs]))
        d['epe_pixel_weighted'] = float(sum(r['epe_sum'] for r in rs) / max(1, sum(r['n_valid'] for r in rs)))
        per_scene[sc] = d
    agg = {'n_pairs': len(rows), 'n_scenes': len(per_scene),
           'epe_pair_mean': float(np.nanmean([r['epe'] for r in rows])),
           'epe_pixel_weighted': float(sum(r['epe_sum'] for r in rows) / max(1, sum(r['n_valid'] for r in rows))),
           'epe_scene_equal_weight': float(np.mean([d['epe_mean'] for d in per_scene.values()])),
           'fl_all_pair_mean': float(np.nanmean([r['fl_all'] for r in rows])),
           'out3_pair_mean': float(np.nanmean([r['out3'] for r in rows]))}
    return {'agg': agg, 'per_scene': per_scene, 'per_pair': rows}


# ----------------------------------------------------------------------------
def build_grid(args):
    archs = [a for a in args.archs.split(',') if a]; seeds = [int(s) for s in args.seeds.split(',') if s]
    runs = [dict(kind='zeroshot', arch=a, rep=-1, fold=-1, seed=0) for a in archs] if not args.no_zeroshot else []
    for r in range(args.reps):
        for f in range(args.folds):
            for s in seeds:
                for a in archs:
                    runs.append(dict(kind='cv', arch=a, rep=r, fold=f, seed=s))
    return runs


def lpt_assign(runs, n_shards, minutes):
    order = sorted(range(len(runs)), key=lambda i: (-minutes[runs[i]['arch']], i))
    load = [0.0] * n_shards; assign = [None] * len(runs)
    for i in order:
        k = min(range(n_shards), key=lambda j: (load[j], j)); assign[i] = k; load[k] += minutes[runs[i]['arch']]
    return assign, load


def tag_of(r):
    return f"flow_{r['arch']}_zeroshot_{VERSION}" if r['kind'] == 'zeroshot' else f"flow_{r['arch']}_r{r['rep']}_f{r['fold']}_s{r['seed']}_{VERSION}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/home/ubuntu/data/training')
    p.add_argument('--out-dir', default='/home/ubuntu/results_flow')
    p.add_argument('--archs', default='raft_small,raft_large')
    p.add_argument('--reps', type=int, default=5); p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--iters', type=int, default=3000); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--crop', default='320,448'); p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--eval-batch', type=int, default=4)
    p.add_argument('--no-zeroshot', action='store_true'); p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--expected-minutes', default='', help='e.g. raft_small=22,raft_large=55 (for LPT)')
    p.add_argument('--s3-prefix', default=''); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--plan-only', action='store_true'); p.add_argument('--calibrate', type=int, default=0, help='run N iterations per arch and report minutes')
    args = p.parse_args()

    minutes = dict(EXPECTED_MIN)
    for kv in [x for x in args.expected_minutes.split(',') if x]:
        k, v = kv.split('='); minutes[k] = float(v)
    grid = build_grid(args); assign, load = lpt_assign(grid, args.num_shards, minutes)
    mine = [r for r, k in zip(grid, assign) if k == args.shard]
    if args.limit: mine = mine[:args.limit]
    if args.plan_only:
        print(f"{len(grid)} runs, {args.num_shards} shards, total {sum(load)/60:.1f} GPU-h, max shard {max(load)/60:.2f} h"); return

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    pairs = list_pairs(args.data_root); scenes = sorted({sc for sc, _, _ in pairs})
    crop = tuple(int(x) for x in args.crop.split(','))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)}/{len(grid)} runs; {len(pairs)} pairs, {len(scenes)} scenes; device={device}", flush=True)

    if args.calibrate:
        for a in [x for x in args.archs.split(',') if x]:
            torch.manual_seed(0)
            ds = SintelFlowPairs(args.data_root, pairs[:200], train=True, crop=crop, seed=0)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
            m = make_model(a, pretrained=not args.no_pretrained); t0 = time.time()
            train(m, dl, args.calibrate, device, log_every=max(1, args.calibrate // 2))
            tr_min = (time.time() - t0) / 60
            te = SintelFlowPairs(args.data_root, pairs[:40], train=False); tl = torch.utils.data.DataLoader(te, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers)
            t0 = time.time(); res = evaluate(m, tl, pairs[:40], device); ev_min = (time.time() - t0) / 60
            est = tr_min / args.calibrate * args.iters + ev_min / 40 * 220
            rec = {'arch': a, 'calib_iters': args.calibrate, 'train_min_per_iter': tr_min / args.calibrate, 'eval_min_per_pair': ev_min / 40,
                   'estimated_run_minutes_at_budget': est, 'epe_pair_mean_after_calib': res['agg']['epe_pair_mean'],
                   'peak_mem_GB': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None}
            print('CALIB', json.dumps(rec), flush=True)
            (out / f'calib_{a}.json').write_text(json.dumps(rec, indent=1))
            if args.s3_prefix: os.system(f"aws s3 cp {out / f'calib_{a}.json'} {args.s3_prefix}/calib_{a}.json --quiet")
        return

    fdoc = out / f'flow_folds_{VERSION}.json'
    if not fdoc.exists():
        fdoc.write_text(json.dumps({'version': VERSION, 'scenes': scenes, 'n_pairs': len(pairs),
                                    'random': {f'rep{r}': {'fold_seed': FOLD_SEED_BASE_RAND + r, 'folds': build_folds_random(scenes, args.folds, FOLD_SEED_BASE_RAND + r)} for r in range(args.reps)}}, indent=1))
        if args.s3_prefix: os.system(f"aws s3 cp {fdoc} {args.s3_prefix}/{fdoc.name} --quiet")

    for i, r in enumerate(mine):
        tag = tag_of(r); fp = out / f'{tag}.json'
        if fp.exists(): print(f"skip {tag}", flush=True); continue
        print(f"[{i+1}/{len(mine)}] {tag}", flush=True)
        try:
            t0 = time.time()
            if r['kind'] == 'zeroshot':
                m = make_model(r['arch'], pretrained=True)
                te = SintelFlowPairs(args.data_root, pairs, train=False)
                tl = torch.utils.data.DataLoader(te, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers)
                res = evaluate(m, tl, pairs, device)
                res.update(dict(kind='zeroshot', version=VERSION, arch=r['arch'], test_scenes=scenes, minutes=(time.time() - t0) / 60))
            else:
                folds = build_folds_random(scenes, args.folds, FOLD_SEED_BASE_RAND + r['rep']); test_scenes = folds[r['fold']]
                tr_pairs = [pq for pq in pairs if pq[0] not in set(test_scenes)]; te_pairs = [pq for pq in pairs if pq[0] in set(test_scenes)]
                torch.manual_seed(r['seed']); np.random.seed(r['seed'])
                ds = SintelFlowPairs(args.data_root, tr_pairs, train=True, crop=crop, seed=r['seed'])
                g = torch.Generator(); g.manual_seed(r['seed'])
                dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, generator=g, pin_memory=True)
                m = make_model(r['arch'], pretrained=not args.no_pretrained)
                train(m, dl, args.iters, device)
                te = SintelFlowPairs(args.data_root, te_pairs, train=False)
                tl = torch.utils.data.DataLoader(te, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers)
                res = evaluate(m, tl, te_pairs, device)
                res.update(dict(kind='cv', version=VERSION, arch=r['arch'], rep=r['rep'], fold=r['fold'], fold_seed=FOLD_SEED_BASE_RAND + r['rep'],
                                seed=r['seed'], test_scenes=test_scenes, train_scenes=sorted({pq[0] for pq in tr_pairs}), train_pairs=len(tr_pairs),
                                test_pairs=len(te_pairs), iters=args.iters, batch_size=args.batch_size, crop=list(crop), pretrained=not args.no_pretrained,
                                minutes=(time.time() - t0) / 60, device=str(device), torch=torch.__version__))
            tmp = fp.with_suffix('.json.tmp'); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, fp)
            if args.s3_prefix: os.system(f"aws s3 cp {fp} {args.s3_prefix}/{fp.name} --quiet")
            a = res['agg']; print(f"   done {tag}: EPE pair {a['epe_pair_mean']:.3f} scene_eq {a['epe_scene_equal_weight']:.3f} fl_all {a['fl_all_pair_mean']:.4f} ({res['minutes']:.1f} min)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"FAILED {tag}: {e}", flush=True)
    print("shard done", flush=True)


if __name__ == '__main__':
    main()
