#!/usr/bin/env python3
"""Series-2 round 9 (A1): are the KR = 25 training compositions of the random design distinct, as Proposition 1(b) now assumes?

Why: the rank statement U + KR - c counts one indicator per training composition; a fold repeated in two assignments repeats its
composition and its (u, T) cells, which would then have to be consolidated (runs becoming seed replicates of the same cells). The
loader labels training sets by (design, rep, fold) — analyze_cv2.py:load_runs, trainset=f"{design}r{rep}f{fold}" — not by membership,
so distinctness must be checked, not assumed.

Inputs (definitions read from the files that wrote them):
  fold file   results_cv_v2/shard*/cv_folds_v2.json  — keys 'scenes' (23 names), 'random' -> 'rep{r}' -> {'fold_seed', 'folds': [list of
              held-out scene names per fold]}, 'stratified' likewise (structure printed and read 2026-11-10; all 16 shard copies identical,
              asserted below). A training composition is the complement of a held-out fold.
  run records results_cv_v2/**/cv2_*_v2.json — fields 'design', 'rep', 'fold', 'per_scene' (keys = held-out scenes of that run), as read
              by analyze_cv2.load_runs. Second route: the held-out scenes recorded in the runs must equal the fold file's folds.
Controls: (i) all shard copies of the fold file identical; (ii) every fold's scenes are a subset of the 23 and the folds of one assignment
partition the 23; (iii) run-record held-out sets equal the fold-file folds for every (rep, fold); (iv) a deliberately repeated fold
(assignment 0 copied) is detected as a repeat by the same code path.
Output: results/benchmark_resolution/composition_distinctness.json
"""
import glob, json, itertools
from pathlib import Path

ROOT = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
fps = sorted(glob.glob(str(ROOT / 'results_cv_v2' / 'shard*' / 'cv_folds_v2.json')))
F = json.load(open(fps[0])); assert all(json.load(open(f)) == F for f in fps), 'shard fold files differ'
scenes = F['scenes']; U = len(scenes); assert U == 23 and len(set(scenes)) == U
assert set(F.keys()) >= {'scenes', 'random', 'stratified'}, F.keys()


def compositions(design):
    """(assignment index, fold index) -> frozenset of training scenes (complement of the held-out fold)."""
    out = {}
    for r, rep in enumerate(sorted(F[design], key=lambda k: int(k[3:]))):
        folds = F[design][rep]['folds']; flat = [s for f in folds for s in f]
        assert sorted(flat) == sorted(scenes), f'{design} {rep}: folds do not partition the units'   # control (ii)
        for k, held in enumerate(folds): out[(r, k)] = frozenset(set(scenes) - set(held))
    return out


def summarise(comp, K):
    sets = list(comp.values()); distinct = len(set(sets)); R = len({r for r, _ in comp})
    # repeated held-out folds across assignments = repeated compositions
    held = {key: frozenset(set(scenes) - s) for key, s in comp.items()}
    max_shared_heldout = max(len(held[a] & held[b]) for a, b in itertools.combinations(held, 2) if a[0] != b[0])
    # connected components of the unit-composition graph (units joined to the compositions under which they are scored)
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    for key, s in comp.items():
        for u in set(scenes) - s: union(('u', u), ('T', key))
    c = len({find(x) for x in list(parent)})
    return {'R': R, 'K': K, 'n_compositions': len(sets), 'n_distinct': distinct, 'all_distinct': distinct == len(sets),
            'max_shared_heldout_units_between_assignments': max_shared_heldout, 'connected_components': c,
            'rank_U_plus_KR_minus_c': U + K * R - c}


rand = compositions('random'); strat = compositions('stratified')
out = {'U': U, 'random': summarise(rand, 5), 'stratified': summarise(strat, 5)}

# control (iii): run records agree with the fold file
runs = {}
for fp in glob.glob(str(ROOT / 'results_cv_v2' / '**' / 'cv2_*_v2.json'), recursive=True):
    d = json.load(open(fp))
    if d.get('kind') != 'cv': continue
    runs.setdefault((d['design'], d['rep'], d['fold']), set()).add(frozenset(d['per_scene'].keys()))
DESIGN_KEY = {'rand': 'random', 'strat': 'stratified'}   # experiment_cv2.py:225,230 write design='rand'/'strat'; the fold file uses 'random'/'stratified'
assert set(d for d, _, _ in runs) == set(DESIGN_KEY), set(d for d, _, _ in runs)
mism = []
for (design, r, k), held_sets in runs.items():
    assert len(held_sets) == 1, (design, r, k)
    fold_file_held = frozenset(F[DESIGN_KEY[design]][f'rep{r}']['folds'][k])
    if next(iter(held_sets)) != fold_file_held: mism.append((design, r, k))
out['run_records_checked'] = len(runs); out['run_records_mismatching_fold_file'] = mism
assert not mism, mism

# control (iv): a repeated fold is detected
dup = dict(rand); dup[(5, 0)] = rand[(0, 0)]
assert summarise(dup, 5)['all_distinct'] is False and summarise(dup, 5)['max_shared_heldout_units_between_assignments'] == 5
out['control_repeated_fold_detected'] = True

json.dump(out, open(ROOT / 'composition_distinctness.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
