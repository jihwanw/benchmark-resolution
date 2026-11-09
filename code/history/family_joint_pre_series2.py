#!/usr/bin/env python3
"""family_joint.py — training-size and sibling-family effects fitted jointly; uncertainty of the family quantities; design variance and
decision rates under the structured (non-exchangeable) interaction (review round 10, A1 / B1 / C1 / C2).

Round 9 found that under PSNR the scene-by-training-set interaction is concentrated on the bamboo pair when both scenes are held out
(family_effect.py). Round 10 observed that both realised co-hold-outs sit in five-scene folds (18-scene training sets), so the
training-size effect of round 8 (trainsize_adjust.py: within-scene 19-scene minus 18-scene contrast averaged over 20 scenes) absorbs
(1/2 + 2/3) beta_bamboo / 20 = 7 beta / 120 of the bamboo effect, i.e. about the whole of the +0.39 to +0.53 dB it reported. Here:

  1. JOINT FIT. On the seed-averaged cell means ybar_{u,T} we fit scene fixed effects + beta_size * 1{19-scene training set}
     + beta_bam * 1{u in bamboo, sibling held out with u} by least squares (the size indicator is a function of the training set, so
     training-set effects cannot be fixed effects alongside it; they stay in the residual and the standard errors are cluster-robust by
     training set). A GLS fit with the training-set and cell covariance taken from the adjusted variance components is reported as a
     check. The cell scores are then adjusted to the 18-scene, sibling-present target (y - beta_size*size4 + beta_bam*bam_abs) and the
     design-exact moment estimator (analyze_cv2.moment_components) is refitted; the same for size-only and bamboo-only adjustments.
     Paired differences: the method-by-size and method-by-sibling effects, adjusted components, gap/MDE with the numerator held at the
     observed gap (denominator-only sensitivity) and with the adjusted gap.
  2. UNCERTAINTY. Parametric bootstrap on the actual design (300 replicates): data generated from the jointly adjusted components
     (unit difficulties fixed at the adjusted unit means rescaled to the fitted S^2_gamma, training-set, cell, run-level and independent
     seed effects Gaussian, as in bootstrap_components.simulate) plus the two fixed effects on the realised cells; in each replicate the
     fixed effects are re-estimated, the scores re-adjusted and the components refitted, giving intervals for beta_size, beta_bam, the
     bamboo interaction reduction, and the adjusted test-side share and ICC. Leave-one-assignment-out estimates of beta_bam and of the
     reduction show the dependence on either of the two co-hold-out events.
  3. STRUCTURED DESIGN VARIANCE AND DECISIONS (PSNR). Scores of random designs (one partition of five with three seeds; five random
     partitions; one full-coverage 5,5,5,4,4 assignment) are simulated from the jointly adjusted exchangeable components plus the
     deterministic bamboo shift (-beta_bam on both bamboo cells whenever the pair shares a fold). The empirical variance of each design's
     score is compared with eqs. (7)/(12)/(18) evaluated at (i) the primary components the paper reports and (ii) the adjusted exchangeable
     components without the shift; the closed-form contribution of the shift, (2 beta/U)^2 p(1-p) under full coverage and
     (2 beta/n)^2 q(1-q) for one partition (p = P(pair shares a fold), q = P(both in the held-out five)), is given alongside. For the
     paired PSNR comparisons the same generator (difference components, method-by-sibling shift) gives the null rejection rate and the
     power at the reported MDE of the normal rule with the reported SE, as in validate_power.py.

Output: results/benchmark_resolution/family_joint.json. CPU only; no new training."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from math import comb
from scipy.stats import norm
import analyze_cv2 as A
import bootstrap_components as B
import family_effect as FE

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
U, N, FOLDS = 23, 5, [5, 5, 5, 4, 4]
METRICS_I = ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_siPSNR_scenegain_rgb']
COMP_KEYS = ['s2_scene', 's2_train', 's2_inter', 's2_seed', 'v_zeta', 'v_eps']
N_BOOT, N_DESIGN, N_DEC = 300, 20000, 4000
Z975, Z80 = norm.ppf(0.975), norm.ppf(0.8)


def comps(d, ycol):
    m = A.moment_components(d, ycol); return {k: float(m[k]) for k in COMP_KEYS}


def with_indicators(d, members):
    d = d.copy(); nT = d.groupby('trainset')['scene'].nunique(); d['size4'] = (d['trainset'].map(nT) == 4).astype(float)
    d['bam_abs'] = (d['scene'].isin(members[FE.FOCAL_FAMILY]) & (d['sib_absent'] == 1.0)).astype(float)
    return d


def cell_table(d, ycol):
    return d.groupby(['scene', 'trainset']).agg(y=(ycol, 'mean'), size4=('size4', 'first'), bam_abs=('bam_abs', 'first')).reset_index()


def joint_fit(cell, cols=('size4', 'bam_abs')):
    """Scene fixed effects + the indicator columns, least squares; cluster-robust (by training set) standard errors."""
    scenes = sorted(cell['scene'].unique()); Zg = np.array([[1.0 if s == sc else 0.0 for sc in scenes] for s in cell['scene']])
    X = np.hstack([Zg, cell[list(cols)].values.astype(float)]); y = cell['y'].values.astype(float)
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; res = y - X @ b
    # cluster-robust by training set
    meat = np.zeros((X.shape[1], X.shape[1]))
    for T, idx in cell.groupby('trainset').indices.items():
        s = X[idx].T @ res[idx]; meat += np.outer(s, s)
    G = cell['trainset'].nunique(); n, k = X.shape
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    out = {}
    for j, c in enumerate(cols): out[c] = {'est': float(b[len(scenes) + j]), 'se_cluster': float(math.sqrt(max(V[len(scenes) + j, len(scenes) + j], 0.0)))}
    return out


def gls_fit(cell, c, S, cols=('size4', 'bam_abs')):
    """GLS with V = s2_train Z_T Z_T' + (s2_inter + v_e/S) I on the cell means (components from the adjusted fit)."""
    scenes = sorted(cell['scene'].unique()); ts = sorted(cell['trainset'].unique())
    Zg = np.array([[1.0 if s == sc else 0.0 for sc in scenes] for s in cell['scene']]); Zt = np.array([[1.0 if t == tt else 0.0 for tt in ts] for t in cell['trainset']])
    X = np.hstack([Zg, cell[list(cols)].values.astype(float)]); y = cell['y'].values.astype(float)
    V = (c['s2_train'] + c['v_zeta'] / S) * Zt @ Zt.T + (c['s2_inter'] + c['v_eps'] / S) * np.eye(len(y))
    Vi = np.linalg.pinv(V); A_ = np.linalg.pinv(X.T @ Vi @ X); b = A_ @ X.T @ Vi @ y
    return {col: {'est': float(b[len(scenes) + j]), 'se_model': float(math.sqrt(max(A_[len(scenes) + j, len(scenes) + j], 0.0)))} for j, col in enumerate(cols)}


def adjust(d, ycol, b_size, b_bam):
    """Remove the fitted coefficients (coefficient convention: b_bam is the coefficient on the sibling-absent indicator, negative under PSNR)."""
    d = d.copy(); d[ycol] = d[ycol] - b_size * d['size4'] - b_bam * d['bam_abs']; return d


def shares(c, S_pred=6):
    return FE.shares(c, S_pred)


def ladder(c):
    v_test = (1 - N / U) * c['s2_scene'] / N
    return {'K1_S3': A.Z_MDE * math.sqrt(2 * ((v_test + c['s2_train'] + c['s2_inter'] / N) + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False))),
            'K5_S3': A.Z_MDE * math.sqrt(2 * ((v_test + c['s2_train'] + c['s2_inter'] / N) / 5 + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, False))),
            'fullcov_S3': A.Z_MDE * math.sqrt(2 * (A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, True)))}


def paired_mde(c):
    v_test = (1 - N / U) * c['s2_scene'] / N
    return {'K1_S3': A.Z_MDE * math.sqrt(v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, 3, False)),
            'full_S3': A.Z_MDE * math.sqrt(A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, 3, True))}


def fit_all(d, ycol, members):
    """Joint fit, the three adjustments, components and ladders."""
    cell = cell_table(d, ycol); jf = joint_fit(cell); b_size, b_bam = jf['size4']['est'], jf['bam_abs']['est']
    size_only = joint_fit(cell, cols=('size4',))['size4']['est']; bam_only = joint_fit(cell, cols=('bam_abs',))['bam_abs']['est']
    c0 = comps(d, ycol); cj = comps(adjust(d, ycol, b_size, b_bam), ycol); cs = comps(adjust(d, ycol, size_only, 0.0), ycol); cb = comps(adjust(d, ycol, 0.0, bam_only), ycol)
    S = float(d.groupby(['scene', 'trainset'])['seed'].nunique().mean()); gls = gls_fit(cell, cj, S)
    return {'joint_ols': jf, 'joint_gls': gls, 'size_only_est': size_only, 'bam_only_est': bam_only,
            'primary': {**c0, **shares(c0), 'ladder': ladder(c0)}, 'size_adjusted': {**cs, **shares(cs), 'ladder': ladder(cs)},
            'bamboo_adjusted': {**cb, **shares(cb), 'ladder': ladder(cb)}, 'joint_adjusted': {**cj, **shares(cj), 'ladder': ladder(cj)},
            'inter_reduction_joint': 1 - cj['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else float('nan'),
            'inter_reduction_bamboo_only': 1 - cb['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else float('nan'),
            'target_shift_mixture_minus_18_joint': b_size * 8 / 23, 'target_shift_mixture_minus_18_size_only': size_only * 8 / 23}


def simulate_actual_design(d, ycol, c, gamma_adj, b_size, b_bam, rng):
    """One parametric replicate on the actual cells: exchangeable parts from c, plus the two fixed effects."""
    st = {**c, 'mean': float(gamma_adj.mean())}
    base = d.copy(); base[ycol] = 0.0
    g = gamma_adj - gamma_adj.mean()
    if g.var(ddof=1) > 0 and c['s2_scene'] > 0: g = g * math.sqrt(c['s2_scene'] / g.var(ddof=1))
    sim = B.simulate(base, ycol, {**st, 's2_scene': 0.0}, rng)          # delta, interaction, seeds; gamma added below (rescaled adjusted means)
    sim[ycol] = sim[ycol] + (g + gamma_adj.mean()).loc[sim['scene']].values
    sim = sim.merge(d[['scene', 'trainset', 'seed', 'size4', 'bam_abs']].drop_duplicates(), on=['scene', 'trainset', 'seed'], how='left')
    sim[ycol] = sim[ycol] + b_size * sim['size4'] + b_bam * sim['bam_abs']           # coefficient convention
    return sim


def bootstrap_family(d, ycol, fit, members, rng, n_boot=N_BOOT):
    cj = fit['joint_adjusted']; b_size, b_bam = fit['joint_ols']['size4']['est'], fit['joint_ols']['bam_abs']['est']
    gamma_adj = adjust(d, ycol, b_size, b_bam).groupby('scene')[ycol].mean()
    keys = ['b_size', 'b_bam', 'reduction', 'share_adj', 'icc_adj', 'inter_adj', 'inter_primary', 'share_primary', 'full_mde_adj']; draws = {k: [] for k in keys}
    for _ in range(n_boot):
        sim = simulate_actual_design(d, ycol, cj, gamma_adj, b_size, b_bam, rng)
        cell = cell_table(sim, ycol); jf = joint_fit(cell); bs, bb = jf['size4']['est'], jf['bam_abs']['est']
        c0 = comps(sim, ycol); c1 = comps(adjust(sim, ycol, bs, bb), ycol); sh1 = shares(c1); sh0 = shares(c0)
        for k, v in [('b_size', bs), ('b_bam', bb), ('reduction', 1 - c1['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else np.nan), ('share_adj', sh1['test_share']), ('icc_adj', sh1['icc']),
                     ('inter_adj', c1['s2_inter']), ('inter_primary', c0['s2_inter']), ('share_primary', sh0['test_share']), ('full_mde_adj', sh1['mde_abs_full_S3'])]: draws[k].append(v)
    out = {'n_boot': n_boot}
    for k, v in draws.items():
        v = np.array(v, dtype=float); v = v[np.isfinite(v)]
        out[k] = {'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], 'median': float(np.median(v)), 'mean': float(v.mean()), 'n_finite': int(len(v))}
    return out


def leave_one_assignment_out(d, ycol, members):
    out = {}
    for r in sorted(d['rep'].unique()):
        dd = d[d['rep'] != r]; cell = cell_table(dd, ycol)
        if cell['bam_abs'].sum() == 0: out[int(r)] = {'b_bam': None, 'reduction': None, 'note': 'no co-hold-out left'}; continue
        jf = joint_fit(cell); c0 = comps(dd, ycol); c1 = comps(adjust(dd, ycol, jf['size4']['est'], jf['bam_abs']['est']), ycol)
        out[int(r)] = {'b_bam': jf['bam_abs']['est'], 'b_size': jf['size4']['est'], 'reduction': 1 - c1['s2_inter'] / c0['s2_inter'] if c0['s2_inter'] > 0 else None, 'n_cohold_cells': int(cell['bam_abs'].sum())}
    return out


# ----------------------------------------------------------------------------------------------------------------------- part 3
def p_share_fold(): return sum(n * (n - 1) for n in FOLDS) / (U * (U - 1))
def q_both_in_five(): return comb(U - 2, N - 2) / comb(U, N)


def structured_design_scores(c, gamma, coef_bam, bam_idx, rng, n_sim, S=3):
    """Scores of three designs under exchangeable components c (no bamboo), fixed gamma (length U), and the shift +coef_bam (the fitted
    coefficient on the sibling-absent indicator, negative under PSNR) on both bamboo units when they share a fold. Returns arrays: one partition (n=5), mean of five partitions, one full-coverage assignment."""
    st, si, vz, ve = math.sqrt(max(c['s2_train'], 0)), math.sqrt(max(c['s2_inter'], 0)), math.sqrt(max(c['v_zeta'], 0)), math.sqrt(max(c['v_eps'], 0))
    one, five, full = np.empty(n_sim), np.empty(n_sim), np.empty(n_sim)
    bam = np.zeros(U); bam[list(bam_idx)] = 1.0
    for i in range(n_sim):
        parts = []
        for k in range(5):
            idx = rng.choice(U, N, replace=False); both = bam[idx].sum() == 2
            seeds = np.mean([rng.normal(0, vz) + rng.normal(0, ve, N).mean() for _ in range(S)])
            parts.append((gamma[idx] + rng.normal(0, st) + rng.normal(0, si, N) + (coef_bam * bam[idx] if both else 0.0)).mean() + seeds)
        one[i] = parts[0]; five[i] = np.mean(parts)
        perm = rng.permutation(U); start = 0; tot = 0.0
        for nk in FOLDS:
            idx = perm[start:start + nk]; start += nk; both = bam[idx].sum() == 2
            seeds = np.mean([rng.normal(0, vz) + rng.normal(0, ve, nk).mean() for _ in range(S)])
            tot += (gamma[idx] + rng.normal(0, st) + rng.normal(0, si, nk) + (coef_bam * bam[idx] if both else 0.0)).sum() + nk * seeds
        full[i] = tot / U
    return one, five, full


def predicted_var(c, S=3):
    v_test = (1 - N / U) * c['s2_scene'] / N
    return {'one': v_test + c['s2_train'] + c['s2_inter'] / N + A.seed_var_design(c['v_zeta'], c['v_eps'], 1, S, False),
            'five': (v_test + c['s2_train'] + c['s2_inter'] / N) / 5 + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, S, False),
            'full': A.C_T * c['s2_train'] + c['s2_inter'] / U + A.seed_var_design(c['v_zeta'], c['v_eps'], 5, S, True)}


def structured_absolute(d, ycol, fit, members, rng, n_sim=N_DESIGN):
    cj = fit['joint_adjusted']; c0 = fit['primary']; b_bam = fit['joint_ols']['bam_abs']['est']; b_size = fit['joint_ols']['size4']['est']
    gam = adjust(d, ycol, b_size, b_bam).groupby('scene')[ycol].mean().sort_index()
    g = gam.values - gam.values.mean(); g = g * math.sqrt(cj['s2_scene'] / g.var(ddof=1)) if g.var(ddof=1) > 0 and cj['s2_scene'] > 0 else g * 0.0
    scenes = list(gam.index); bam_idx = [scenes.index(s) for s in members[FE.FOCAL_FAMILY]]
    one, five, full = structured_design_scores(cj, g, b_bam, bam_idx, rng, n_sim)
    emp = {'one': float(one.var(ddof=1)), 'five': float(five.var(ddof=1)), 'full': float(full.var(ddof=1))}
    p, q = p_share_fold(), q_both_in_five()
    closed = {'full': (2 * b_bam / U) ** 2 * p * (1 - p), 'one': (2 * b_bam / N) ** 2 * q * (1 - q), 'five': (2 * b_bam / N) ** 2 * q * (1 - q) / 5}
    pr0, prj = predicted_var(c0), predicted_var(cj)
    out = {'n_sim': n_sim, 'empirical_var': emp, 'predicted_var_primary': pr0, 'predicted_var_joint_exchangeable': prj,
           'closed_form_bamboo_term': closed, 'predicted_var_joint_plus_closed_form': {k: prj[k] + closed[k] for k in prj},
           'ratio_empirical_over_primary': {k: emp[k] / pr0[k] for k in emp}, 'ratio_empirical_over_joint_exchangeable': {k: emp[k] / prj[k] for k in emp},
           'ratio_empirical_over_joint_plus_closed': {k: emp[k] / (prj[k] + closed[k]) for k in emp},
           'mde_factor_vs_primary': {k: math.sqrt(emp[k] / pr0[k]) for k in emp}, 'p_share_fold': p, 'q_both_in_five': q,
           'mean_shift_one_partition': float(one.mean() - g.mean() - gam.values.mean())}
    return out


def structured_paired(dd, pfit, b_bam_pair, rng, n_sim=N_DEC):
    """Null rejection and power of the normal rule with the reported (primary) paired SE, under the structured paired generator."""
    cj, c0 = pfit['joint_adjusted'], pfit['primary']
    gaps = adjust(dd, 'y', pfit['joint_ols']['size4']['est'], b_bam_pair).groupby('scene')['y'].mean().sort_index()
    g = gaps.values - gaps.values.mean(); g = g * math.sqrt(cj['s2_scene'] / g.var(ddof=1)) if g.var(ddof=1) > 0 and cj['s2_scene'] > 0 else g * 0.0
    scenes = list(gaps.index); bam_idx = [scenes.index(s) for s in FE.families(scenes)[1][FE.FOCAL_FAMILY]]
    se0 = {'one': math.sqrt(predicted_var(c0)['one']), 'full': math.sqrt(predicted_var(c0)['full'])}
    mde0 = {k: A.Z_MDE * v for k, v in se0.items()}
    out = {'n_sim': n_sim, 'se_primary': se0, 'mde_primary': mde0, 'method_by_sibling_effect': b_bam_pair}
    for true_mean, label in [(0.0, 'null'), (None, 'power')]:
        res = {}
        for design in ['one', 'full']:
            tm = mde0[design] if true_mean is None else 0.0
            one, _, full = structured_design_scores(cj, g + tm, b_bam_pair, bam_idx, rng, n_sim)
            val = one if design == 'one' else full
            res[design] = float((np.abs(val) >= Z975 * se0[design]).mean())
        out[label] = res
    return out


def main():
    folds = json.load(open(R / 'results_cv_v2' / 'shard0' / 'cv_folds_v2.json')); scenes = folds['scenes']; fam, members = FE.families(scenes)
    held_i = FE.heldout_map(folds, 'rand')
    df = A.load_runs(R / 'results_cv_v2'); df = df[df['design'] == 'rand']; df = with_indicators(FE.add_sibling_columns(df, fam, members, held_i), members)
    rng = np.random.default_rng(20261107)
    out = {'intrinsic': {}, 'pairs': {}, 'bootstrap': {}, 'loo': {}, 'structured_absolute': {}, 'structured_paired': {}}
    # 1. joint fits
    for a in sorted(df['arch'].unique()):
        out['intrinsic'][a] = {}
        for y in METRICS_I:
            d = df[df['arch'] == a].dropna(subset=[y]); f = fit_all(d, y, members); out['intrinsic'][a][y] = f
            jf = f['joint_ols']
            print(f"{a:12s} {y[:12]:12s} size: alone {f['size_only_est']:+.3g} joint {jf['size4']['est']:+.3g} (se {jf['size4']['se_cluster']:.3g}; gls {f['joint_gls']['size4']['est']:+.3g}) | bamboo: alone {f['bam_only_est']:+.3g} joint {jf['bam_abs']['est']:+.3g} (se {jf['bam_abs']['se_cluster']:.3g}) | "
                  f"inter {f['primary']['s2_inter']:.3g} -> joint {f['joint_adjusted']['s2_inter']:.3g} ({f['inter_reduction_joint']:+.0%}) | train {f['primary']['s2_train']:.3g}->{f['joint_adjusted']['s2_train']:.3g} | full3 {f['primary']['ladder']['fullcov_S3']:.3g}->{f['joint_adjusted']['ladder']['fullcov_S3']:.3g} rand5 {f['primary']['ladder']['K5_S3']:.3g}->{f['joint_adjusted']['ladder']['K5_S3']:.3g} | shift {f['target_shift_mixture_minus_18_size_only']:+.3g}->{f['target_shift_mixture_minus_18_joint']:+.3g}", flush=True)
    # pairs
    archs = sorted(df['arch'].unique())
    for i, a1 in enumerate(archs):
        for a2 in archs[i + 1:]:
            for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean']:
                dd = B.pair_frame(df, a1, a2, y); dd = with_indicators(FE.add_sibling_columns(dd, fam, members, held_i), members)
                f = fit_all(dd, 'y', members); gap = float(dd['y'].mean()); gap_adj = float(adjust(dd, 'y', f['joint_ols']['size4']['est'], f['joint_ols']['bam_abs']['est'])['y'].mean())
                pm0, pmj = paired_mde(f['primary']), paired_mde(f['joint_adjusted'])
                rec = {'joint_ols': f['joint_ols'], 'gap_observed': gap, 'gap_adjusted': gap_adj,
                       'primary': {**{k: f['primary'][k] for k in COMP_KEYS}, 'paired_mde': pm0, 'gap_over_mde_full': abs(gap) / pm0['full_S3'], 'gap_over_mde_K1': abs(gap) / pm0['K1_S3']},
                       'joint_adjusted': {**{k: f['joint_adjusted'][k] for k in COMP_KEYS}, 'paired_mde': pmj,
                                          'gap_over_mde_full_fixed_numerator': abs(gap) / pmj['full_S3'], 'gap_over_mde_full_adjusted_numerator': abs(gap_adj) / pmj['full_S3'],
                                          'gap_over_mde_K1_fixed_numerator': abs(gap) / pmj['K1_S3'], 'gap_over_mde_K1_adjusted_numerator': abs(gap_adj) / pmj['K1_S3']}}
                out['pairs'][f'{a2} - {a1}|{y}'] = rec
                print(f"pair {a2}-{a1} {y[:10]}: dsize {f['joint_ols']['size4']['est']:+.3g} dbam {f['joint_ols']['bam_abs']['est']:+.3g} | gap {gap:+.4g} -> adj {gap_adj:+.4g} | ratio full {rec['primary']['gap_over_mde_full']:.2f} -> fixed-num {rec['joint_adjusted']['gap_over_mde_full_fixed_numerator']:.2f} / adj-num {rec['joint_adjusted']['gap_over_mde_full_adjusted_numerator']:.2f}", flush=True)
    # 2. uncertainty (PSNR and DSSIM, four architectures) + leave-one-assignment-out
    for a in archs:
        for y in ['R_PSNR_of_mean_mse', 'dssim_R_mean']:
            d = df[df['arch'] == a].dropna(subset=[y]); f = out['intrinsic'][a][y]
            bs = bootstrap_family(d, y, f, members, rng); out['bootstrap'][f'{a}|{y}'] = bs; out['loo'][f'{a}|{y}'] = leave_one_assignment_out(d, y, members)
            print(f"  boot {a} {y[:10]}: b_bam {f['joint_ols']['bam_abs']['est']:+.3g} [{bs['b_bam']['ci95'][0]:.3g}, {bs['b_bam']['ci95'][1]:.3g}] b_size {f['joint_ols']['size4']['est']:+.3g} [{bs['b_size']['ci95'][0]:+.3g}, {bs['b_size']['ci95'][1]:+.3g}] "
                  f"reduction {f['inter_reduction_joint']:.2f} [{bs['reduction']['ci95'][0]:.2f}, {bs['reduction']['ci95'][1]:.2f}] share_adj {f['joint_adjusted']['test_share']:.2f} [{bs['share_adj']['ci95'][0]:.2f}, {bs['share_adj']['ci95'][1]:.2f}] | "
                  f"loo b_bam {[round(v['b_bam'], 2) if v['b_bam'] is not None else None for v in out['loo'][f'{a}|{y}'].values()]} loo red {[round(v['reduction'], 2) if v['reduction'] is not None else None for v in out['loo'][f'{a}|{y}'].values()]}", flush=True)
    # 3. structured design variance (PSNR, four architectures) and paired decisions (DirectCNN pairs, PSNR)
    for a in archs:
        d = df[df['arch'] == a]; sa = structured_absolute(d, 'R_PSNR_of_mean_mse', out['intrinsic'][a]['R_PSNR_of_mean_mse'], members, rng); out['structured_absolute'][a] = sa
        print(f"  structured {a} PSNR: emp var one {sa['empirical_var']['one']:.3g} five {sa['empirical_var']['five']:.3g} full {sa['empirical_var']['full']:.3g} | ratio vs primary one {sa['ratio_empirical_over_primary']['one']:.2f} five {sa['ratio_empirical_over_primary']['five']:.2f} full {sa['ratio_empirical_over_primary']['full']:.2f} | "
              f"vs joint-exch one {sa['ratio_empirical_over_joint_exchangeable']['one']:.2f} full {sa['ratio_empirical_over_joint_exchangeable']['full']:.2f} | vs joint+closed one {sa['ratio_empirical_over_joint_plus_closed']['one']:.2f} full {sa['ratio_empirical_over_joint_plus_closed']['full']:.2f} | MDE factor vs primary one {sa['mde_factor_vs_primary']['one']:.2f} full {sa['mde_factor_vs_primary']['full']:.2f}", flush=True)
    for a2 in ['unet', 'proposed', 'transformer']:
        key = f'{a2} - direct|R_PSNR_of_mean_mse'; dd = B.pair_frame(df, 'direct', a2, 'R_PSNR_of_mean_mse'); dd = with_indicators(FE.add_sibling_columns(dd, fam, members, held_i), members)
        pf = fit_all(dd, 'y', members); sp = structured_paired(dd, pf, pf['joint_ols']['bam_abs']['est'], rng); out['structured_paired'][key] = sp
        print(f"  paired decisions {key}: null one {sp['null']['one']:.3f} full {sp['null']['full']:.3f} | power one {sp['power']['one']:.3f} full {sp['power']['full']:.3f} (dbam {sp['method_by_sibling_effect']:+.3g})", flush=True)
    (R / 'family_joint.json').write_text(json.dumps(out, indent=1, default=float)); print('wrote', R / 'family_joint.json')


if __name__ == '__main__':
    main()
