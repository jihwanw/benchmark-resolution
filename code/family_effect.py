#!/usr/bin/env python3
"""family_effect.py — structure of the scene-by-training-set interaction: units that share assets (review round 9, A1/C1).

MPI Sintel's 23 training scenes are rendered in families from shared assets (alley_1/2, ambush_2/4/5/6/7, bamboo_1/2, bandage_1/2,
cave_2/4, market_2/5/6, mountain_1, shaman_2/3, sleeping_1/2, temple_2/3; the family is the name before the underscore). For a
held-out scene u and training composition T the fold files say whether a sibling (another member of u's family) was in T. The
primary analysis treats the interaction (gamma delta)_{u,T} as an exchangeable random effect over cells; Fig. 4 (bamboo_2 turns
orange when bamboo_1 is also held out) suggests that under PSNR it is a function of one identifiable event, a unit held out
together with its only sibling. This script

  1. for every scene of a two-member family, computes the within-scene difference of the seed-averaged cell score between
     compositions with the sibling in training and compositions without it (the five-member ambush family and the three-member
     market family are never without a sibling; mountain_1 has none);
  2. estimates the bamboo-family sibling effect beta (pooled over bamboo_1 and bamboo_2), removes it as a fixed effect (the
     sibling-absent cells of the two bamboo scenes are shifted by +beta, i.e. to the sibling-present target), and refits the
     design-exact moment estimator (analyze_cv2.moment_components); also refits with the two bamboo scenes dropped (U = 21);
  3. states the realised-draw conditionality: under random 5-fold assignments of 5,5,5,4,4 units the probability that a unit's
     single sibling falls in the same fold is sum_k n_k (n_k - 1) / (U (U - 1)) = 84/506 per assignment; the realised number of
     co-hold-outs of each two-member family is counted, and the distribution of the fitted PSNR interaction over random
     five-assignment designs is simulated from the fixed-effect-removed components with the bamboo shift as a deterministic
     fixed effect (gamma_u fixed at the adjusted unit means, delta, interaction and the two seed terms Gaussian as in
     bootstrap_components.simulate, 3 seeds, R = 5 assignments);
  4. repeats 1-2 for the paired differences (six intrinsic pairs; method-by-sibling effect beta_B - beta_A) and for the flow arm
     (same fold files; end-point error).

Output: results/benchmark_resolution/family_effect.json. CPU only; no new training."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_cv2 as A
import analyze_flow as F
import bootstrap_components as B

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N = 23, 5
FOLD_SIZES = [5, 5, 5, 4, 4]
METRICS_I = ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_siPSNR_scenegain_rgb']
METRICS_F = ['epe_mean']
FOCAL_FAMILY = 'bamboo'
N_SIM = 300
COMP_KEYS = ['s2_scene', 's2_train', 's2_inter', 's2_seed', 'v_zeta', 'v_eps']


def families(scenes):
    fam = {s: s.rsplit('_', 1)[0] for s in scenes}
    members = {}
    for s, f in fam.items(): members.setdefault(f, []).append(s)
    return fam, members


def heldout_map(folds, prefix):
    out = {}
    for rep, v in folds['random'].items():
        r = int(rep[3:])
        for f, fs in enumerate(v['folds']): out[f'{prefix}r{r}f{f}'] = set(fs)
    return out


def add_sibling_columns(d, fam, members, heldout):
    d = d.copy()
    def held(T):
        return heldout[T] if T in heldout else heldout['rand' + T]      # pair frames key training sets as r{rep}f{fold}
    def n_sib_in_train(row):
        u, T = row['scene'], row['trainset']
        return sum(1 for s in members[fam[u]] if s != u and s not in held(T))
    d['n_sib'] = d['scene'].map(lambda u: len(members[fam[u]]) - 1)
    d['n_sib_train'] = d.apply(n_sib_in_train, axis=1)
    d['sib_absent'] = ((d['n_sib'] > 0) & (d['n_sib_train'] == 0)).astype(float)
    return d


def within_scene_differences(d, ycol):
    """Per scene of a two-member family: mean cell score with the sibling in training minus without; cell counts."""
    cell = d.groupby(['scene', 'trainset']).agg(y=(ycol, 'mean'), sa=('sib_absent', 'first'), ns=('n_sib', 'first')).reset_index()
    out = {}
    for sc, g in cell.groupby('scene'):
        if int(g['ns'].iloc[0]) != 1: continue
        p, a = g[g['sa'] == 0]['y'], g[g['sa'] == 1]['y']
        out[sc] = {'n_cells_sibling_present': int(len(p)), 'n_cells_sibling_absent': int(len(a)),
                   'present_minus_absent': float(p.mean() - a.mean()) if len(p) and len(a) else None,
                   'mean_present': float(p.mean()) if len(p) else None, 'mean_absent': float(a.mean()) if len(a) else None}
    return out


def focal_beta(wsd, members):
    vals = [wsd[s]['present_minus_absent'] for s in members[FOCAL_FAMILY] if s in wsd and wsd[s]['present_minus_absent'] is not None]
    return float(np.mean(vals)), vals


def remove_focal_effect(d, ycol, beta, members):
    d = d.copy(); mask = d['scene'].isin(members[FOCAL_FAMILY]) & (d['sib_absent'] == 1.0)
    d.loc[mask, ycol] = d.loc[mask, ycol] + beta
    return d, int(mask.sum())


def comps(d, ycol):
    m = A.moment_components(d, ycol); return {k: float(m[k]) for k in COMP_KEYS}


def shares(c, S_pred=6):
    v_test = (1 - N / U) * c['s2_scene'] / N
    v_pred = v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, S_pred, False)
    den = c['s2_scene'] + c['s2_inter'] + c['s2_seed']
    return {'test_share': v_test / v_pred if v_pred > 0 else float('nan'), 'icc': c['s2_scene'] / den if den > 0 else float('nan'),
            'mde_abs_K1_S3': A.Z_MDE * math.sqrt(2 * ((v_test + c['s2_train'] + c['s2_inter'] / N) + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False))),
            'mde_abs_full_S3': A.Z_MDE * math.sqrt(2 * (A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, True)))}


def p_cohold(fold_sizes=FOLD_SIZES, U=U):
    return sum(n * (n - 1) for n in fold_sizes) / (U * (U - 1))


def random_assignment(rng, scenes, fold_sizes=FOLD_SIZES):
    perm = list(rng.permutation(scenes)); folds, i = [], 0
    for n in fold_sizes: folds.append(perm[i:i + n]); i += n
    return folds


def simulate_random_design(rng, scenes, gamma, c, beta, members, R=5, S=3):
    """Cells of a random R-assignment design with a deterministic -beta on the focal family's cells when co-held-out."""
    focal = set(members[FOCAL_FAMILY]); rows = []
    for r in range(R):
        folds = random_assignment(rng, scenes)
        for f, fs in enumerate(folds):
            T = f'simr{r}f{f}'; delta = rng.normal(0, math.sqrt(max(c['s2_train'], 0.0)))
            zetas = rng.normal(0, math.sqrt(max(c['v_zeta'], 0.0)), S)
            both = focal.issubset(set(fs))
            for u in fs:
                gd = rng.normal(0, math.sqrt(max(c['s2_inter'], 0.0)))
                shift = -beta if (both and u in focal) else 0.0
                for s in range(S):
                    rows.append({'scene': u, 'trainset': T, 'seed': s, 'y': gamma[u] + delta + gd + zetas[s] + rng.normal(0, math.sqrt(max(c['v_eps'], 0.0))) + shift})
    return pd.DataFrame(rows), int(sum(focal.issubset(set(fs)) for fs in folds))


def draw_conditionality(d, ycol, c_fixed, beta, members, scenes, rng, n_sim=N_SIM):
    """Distribution over random five-assignment designs of the primary (unadjusted) fit when the focal shift is a fixed effect."""
    adj, _ = remove_focal_effect(d, ycol, beta, members)
    gamma = adj.groupby('scene')[ycol].mean().to_dict()
    inter, train, scene, share, counts = [], [], [], [], []
    for _ in range(n_sim):
        sim, k = simulate_random_design(rng, scenes, gamma, c_fixed, beta, members)
        # k counts co-hold-outs in the last assignment only; recount over all R assignments
        k = int(sim.groupby('trainset').apply(lambda g: set(members[FOCAL_FAMILY]).issubset(set(g['scene']))).sum())
        cs = comps(sim, 'y'); sh = shares(cs)
        inter.append(cs['s2_inter']); train.append(cs['s2_train']); scene.append(cs['s2_scene']); share.append(sh['test_share']); counts.append(k)
    inter, train, scene, share, counts = map(np.array, (inter, train, scene, share, counts))
    def q(v): return {'median': float(np.median(v)), 'p5': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)), 'mean': float(np.mean(v))}
    by_count = {int(k): {'n': int((counts == k).sum()), 'inter_median': float(np.median(inter[counts == k]))} for k in sorted(set(counts.tolist()))}
    return {'n_sim': n_sim, 's2_inter': q(inter), 's2_train': q(train), 's2_scene': q(scene), 'test_share': q(share),
            'cohold_count': {'mean': float(counts.mean()), 'dist': {int(k): int((counts == k).sum()) for k in sorted(set(counts.tolist()))}},
            'inter_by_cohold_count': by_count, '_inter_draws': inter.tolist()}


def percentile_of(value, draws):
    draws = np.asarray(draws); return float((draws < value).mean())


def main():
    folds = json.load(open(R / 'results_cv_v2' / 'shard0' / 'cv_folds_v2.json')); scenes = folds['scenes']
    fam, members = families(scenes)
    two_member = sorted(f for f, m in members.items() if len(m) == 2)
    held_i = heldout_map(folds, 'rand'); held_f = heldout_map(folds, '')
    p_co = p_cohold()
    out = {'families': members, 'two_member_families': two_member, 'p_cohold_per_assignment': p_co, 'expected_coholds_in_5': 5 * p_co,
           'realised_coholds': {}, 'intrinsic': {}, 'pairs': {}, 'flow': {}, 'draw_conditionality': {}}
    # realised co-hold-outs per two-member family over the five random assignments
    for f in two_member:
        a, b = members[f]
        out['realised_coholds'][f] = int(sum(1 for k, hs in held_i.items() if a in hs and b in hs))
    tot = sum(out['realised_coholds'].values()); out['realised_coholds_total'] = tot; out['realised_cohold_rate'] = tot / (5 * len(two_member))
    from scipy.stats import binom
    out['p_focal_coholds_ge_realised'] = float(binom.sf(out['realised_coholds'][FOCAL_FAMILY] - 1, 5, p_co))
    print(f"p(co-hold-out per assignment) = {p_co:.3f}; expected in 5 = {5*p_co:.2f}; realised: {out['realised_coholds']} (rate {out['realised_cohold_rate']:.2f}); "
          f"P(bamboo >= {out['realised_coholds'][FOCAL_FAMILY]}) = {out['p_focal_coholds_ge_realised']:.3f}", flush=True)

    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']
    df = add_sibling_columns(df, fam, members, held_i)
    rng = np.random.default_rng(20261106)
    for a in sorted(df['arch'].unique()):
        out['intrinsic'][a] = {}
        for y in METRICS_I:
            d = df[df['arch'] == a].dropna(subset=[y]); wsd = within_scene_differences(d, y); beta, vals = focal_beta(wsd, members)
            others = [v['present_minus_absent'] for s, v in wsd.items() if fam[s] != FOCAL_FAMILY and v['present_minus_absent'] is not None]
            c0 = comps(d, y); d1, ncells = remove_focal_effect(d, y, beta, members); c1 = comps(d1, y)
            d2 = d[~d['scene'].isin(members[FOCAL_FAMILY])]; c2 = comps(d2, y)
            red = 1 - c1['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else float('nan')
            rec = {'within_scene_differences': wsd, 'beta_focal_present_minus_absent': beta, 'beta_focal_per_scene': vals, 'n_cells_shifted': ncells,
                   'other_two_member_scenes': {'n': len(others), 'min': float(min(others)), 'max': float(max(others)), 'max_abs': float(max(abs(v) for v in others))},
                   'primary': {**c0, **shares(c0)}, 'focal_fixed_effect_removed': {**c1, **shares(c1)}, 'focal_family_dropped_U21': c2,
                   'interaction_reduction_fixed_effect': red, 'interaction_reduction_dropped': 1 - c2['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else float('nan')}
            out['intrinsic'][a][y] = rec
            print(f"{a:12s} {y[:14]:14s} beta {beta:+.3g} others |max| {rec['other_two_member_scenes']['max_abs']:.3g} | inter {c0['s2_inter']:.3g} -> {c1['s2_inter']:.3g} ({red:+.0%}) -> U21 {c2['s2_inter']:.3g} | "
                  f"train {c0['s2_train']:.3g}->{c1['s2_train']:.3g} scene {c0['s2_scene']:.3g}->{c1['s2_scene']:.3g} | test share {rec['primary']['test_share']:.2f}->{rec['focal_fixed_effect_removed']['test_share']:.2f} "
                  f"icc {rec['primary']['icc']:.2f}->{rec['focal_fixed_effect_removed']['icc']:.2f} mde_full {rec['primary']['mde_abs_full_S3']:.3g}->{rec['focal_fixed_effect_removed']['mde_abs_full_S3']:.3g}", flush=True)
        # draw conditionality for PSNR (and DSSIM, the other metric with a visible effect)
        for y in ['R_PSNR_of_mean_mse', 'dssim_R_mean']:
            rec = out['intrinsic'][a][y]; d = df[df['arch'] == a].dropna(subset=[y])
            dc = draw_conditionality(d, y, rec['focal_fixed_effect_removed'], rec['beta_focal_present_minus_absent'], members, scenes, rng)
            dc['realised_s2_inter'] = rec['primary']['s2_inter']; dc['realised_percentile'] = percentile_of(rec['primary']['s2_inter'], dc.pop('_inter_draws'))
            dc['realised_test_share'] = rec['primary']['test_share']
            out['draw_conditionality'][f'{a}|{y}'] = dc
            print(f"  draw-conditionality {a} {y[:10]}: inter median {dc['s2_inter']['median']:.3g} [{dc['s2_inter']['p5']:.3g}, {dc['s2_inter']['p95']:.3g}] vs realised {dc['realised_s2_inter']:.3g} (pct {dc['realised_percentile']:.2f}); "
                  f"test share median {dc['test_share']['median']:.2f} vs realised {dc['realised_test_share']:.2f}; co-hold-out counts {dc['cohold_count']['dist']}; inter by count {{{', '.join(f'{k}: {v['inter_median']:.3g}' for k, v in dc['inter_by_cohold_count'].items())}}}", flush=True)
    # pairs
    archs = sorted(df['arch'].unique())
    for i, a1 in enumerate(archs):
        for a2 in archs[i + 1:]:
            for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean']:
                dd = B.pair_frame(df, a1, a2, y); dd = add_sibling_columns(dd, fam, members, held_i)
                wsd = within_scene_differences(dd, 'y'); beta, _ = focal_beta(wsd, members); c0 = comps(dd, 'y'); d1, _ = remove_focal_effect(dd, 'y', beta, members); c1 = comps(d1, 'y'); gap = float(dd['y'].mean())
                def pm(c):
                    v_test = (1 - N / U) * c['s2_scene'] / N
                    k1 = A.Z_MDE * math.sqrt(v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False))
                    fu = A.Z_MDE * math.sqrt(A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, True))
                    return {'paired_mde_K1_S3': k1, 'paired_mde_full_S3': fu, 'gap_over_mde_K1': abs(gap) / k1, 'gap_over_mde_full': abs(gap) / fu}
                out['pairs'][f'{a2} - {a1}|{y}'] = {'method_by_sibling_effect_focal': beta, 'gap': gap, 'primary': {**c0, **pm(c0)}, 'focal_fixed_effect_removed': {**c1, **pm(c1)},
                                                    'interaction_reduction_fixed_effect': 1 - c1['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else float('nan'),
                                                    'scene_by_method_change': c1['s2_scene'] - c0['s2_scene']}
                print(f"pair {a2}-{a1} {y[:10]}: beta_B-beta_A {beta:+.3g} | S2_dgamma {c0['s2_scene']:.3g}->{c1['s2_scene']:.3g} inter {c0['s2_inter']:.3g}->{c1['s2_inter']:.3g} | gap/MDE_full {pm(c0)['gap_over_mde_full']:.2f}->{pm(c1)['gap_over_mde_full']:.2f}", flush=True)
    # flow
    fdf, _ = F.load(R / 'results_flow_v1'); fdf = add_sibling_columns(fdf, fam, members, held_f)
    for a in sorted(fdf['arch'].unique()):
        out['flow'][a] = {}
        for y in METRICS_F:
            d = fdf[fdf['arch'] == a].dropna(subset=[y]); wsd = within_scene_differences(d, y); beta, vals = focal_beta(wsd, members)
            others = [v['present_minus_absent'] for s, v in wsd.items() if fam[s] != FOCAL_FAMILY and v['present_minus_absent'] is not None]
            c0 = comps(d, y); d1, _ = remove_focal_effect(d, y, beta, members); c1 = comps(d1, y)
            sd_scene = math.sqrt(c0['s2_scene'])
            out['flow'][a][y] = {'within_scene_differences': wsd, 'beta_focal_present_minus_absent': beta, 'beta_focal_per_scene': vals,
                                 'all_two_member_scenes_max_abs': float(max(abs(v['present_minus_absent']) for v in wsd.values() if v['present_minus_absent'] is not None)),
                                 'max_abs_over_between_scene_sd': float(max(abs(v['present_minus_absent']) for v in wsd.values() if v['present_minus_absent'] is not None) / sd_scene),
                                 'primary': c0, 'focal_fixed_effect_removed': c1, 'interaction_reduction_fixed_effect': 1 - c1['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else float('nan')}
            print(f"flow {a:11s} {y}: bamboo beta {beta:+.3g} px; |max| over two-member scenes {out['flow'][a][y]['all_two_member_scenes_max_abs']:.3g} px = {out['flow'][a][y]['max_abs_over_between_scene_sd']:.2f} between-scene SD | inter {c0['s2_inter']:.3g} -> {c1['s2_inter']:.3g}", flush=True)
    (R / 'family_effect.json').write_text(json.dumps(out, indent=1, default=float)); print('wrote', R / 'family_effect.json')


if __name__ == '__main__':
    main()
