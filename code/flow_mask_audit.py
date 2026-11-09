#!/usr/bin/env python3
"""flow_mask_audit.py — what the evaluation mask of the flow arm actually excludes (series-2 review, round 3, A1).

experiment_flow.py:87 defines the per-pixel validity used in training AND evaluation as
    valid = ~invalid & (||gt flow|| < MAX_FLOW),  MAX_FLOW = 400.0 (line 31),
where `invalid` is the dataset's invalid mask (invalid/<scene>/<frame>.png > 0; line 66) and the flow is read by read_flo (lines 36-43:
PIEH tag, W, H, interleaved float32 u,v). Table 4 of the manuscript described the scores as computed "over valid pixels" without the
magnitude cutoff. This script reads every ground-truth flow and invalid mask of the 1,041 pairs the arm scored (list_pairs: consecutive
clean frames with a .flo file, experiment_flow.py:46-54) with the same functions, and counts, per scene and in total, the pixels that the
cutoff removes beyond the dataset's own mask, the pixels the dataset mask removes, and the largest ground-truth magnitude per scene.
Controls: the .flo tag is asserted by read_flo itself; the pair count is asserted to equal 1,041 (Sec. 4); the invalid-mask convention
(> 0 = invalid) is the loader's. Interface card: INTERFACE_CARD_sintel_loader_and_losses.md (flow section added 2026-11-10).
Output: results/benchmark_resolution/flow_mask_audit.json."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import experiment_flow as EF                                 # the arm's own read_flo / list_pairs / MAX_FLOW

CLEAN_ROOT = Path('/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/data/sintel_real/training')      # clean frames (same files as the arm's root/clean)
EXTRAS = Path('/Users/jihwanw/PhD/AI/CV/data/sintel_flow_extras')                             # flow/, invalid/, occlusions/ from MPI-Sintel-training_extras
RES = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
N_PAIRS = 1041


class Root:
    """Minimal stand-in so that list_pairs sees clean/ (local copy) and flow/ (extras) under one root."""
    pass


def main():
    # list_pairs(root) expects root/clean and root/flow; build the pair list with the arm's own logic on the two directories
    clean, flow = CLEAN_ROOT / 'clean', EXTRAS / 'flow'
    pairs = []
    for sc in sorted(p.name for p in clean.iterdir() if p.is_dir()):
        frames = sorted(p.name for p in (clean / sc).iterdir() if p.suffix == '.png')
        for a, b in zip(frames[:-1], frames[1:]):
            if (flow / sc / (Path(a).stem + '.flo')).exists(): pairs.append((sc, a, b))
    assert len(pairs) == N_PAIRS, len(pairs)
    per_scene = {}
    for sc, a, b in pairs:
        fl = EF.read_flo(flow / sc / (Path(a).stem + '.flo')); mag = np.linalg.norm(fl, axis=-1)
        inv_p = EXTRAS / 'invalid' / sc / a
        invalid = (np.asarray(Image.open(inv_p).convert('L')) > 0) if inv_p.exists() else np.zeros(mag.shape, bool)
        assert inv_p.exists(), inv_p
        cut = (~invalid) & (mag >= EF.MAX_FLOW)                 # removed by the cutoff beyond the dataset mask
        d = per_scene.setdefault(sc, {'n_pairs': 0, 'n_pixels': 0, 'n_invalid': 0, 'n_cut_beyond_mask': 0, 'n_ge400_total': 0, 'max_mag': 0.0, 'pairs_with_cut': 0})
        d['n_pairs'] += 1; d['n_pixels'] += int(mag.size); d['n_invalid'] += int(invalid.sum()); d['n_cut_beyond_mask'] += int(cut.sum())
        d['n_ge400_total'] += int((mag >= EF.MAX_FLOW).sum()); d['max_mag'] = max(d['max_mag'], float(mag.max())); d['pairs_with_cut'] += int(cut.any())
    tot = {k: sum(v[k] for v in per_scene.values()) for k in ['n_pairs', 'n_pixels', 'n_invalid', 'n_cut_beyond_mask', 'n_ge400_total', 'pairs_with_cut']}
    tot['max_mag'] = max(v['max_mag'] for v in per_scene.values())
    tot['frac_cut_of_dataset_valid'] = tot['n_cut_beyond_mask'] / max(tot['n_pixels'] - tot['n_invalid'], 1)
    tot['frac_invalid'] = tot['n_invalid'] / tot['n_pixels']
    for v in per_scene.values(): v['frac_cut_of_dataset_valid'] = v['n_cut_beyond_mask'] / max(v['n_pixels'] - v['n_invalid'], 1)
    out = {'max_flow_cutoff': EF.MAX_FLOW, 'definition_source': 'experiment_flow.py:31, 66, 87; read_flo 36-43', 'total': tot, 'per_scene': per_scene,
           'scenes_with_cut': sorted((sc, v['n_cut_beyond_mask'], v['frac_cut_of_dataset_valid'], v['max_mag']) for sc, v in per_scene.items() if v['n_cut_beyond_mask'] > 0)}
    (RES / 'flow_mask_audit.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(tot, indent=1)); print('scenes with pixels removed by the cutoff beyond the dataset mask:', out['scenes_with_cut'])
    print('max |gt| per scene:', {sc: round(v['max_mag'], 1) for sc, v in per_scene.items()})


if __name__ == '__main__':
    main()
