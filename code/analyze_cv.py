#!/usr/bin/env python3
"""analyze_cv.py — pre-registered analysis of the benchmark-resolution killing experiment (v1).

Pre-registration: PhD/AI/CV/ideas/2026-11-01_benchmark_resolution.md  (H1 share thresholds 0.3/0.7,
death criterion scene ICC < 0.5, H2 interaction, H3 comparison with the implied S^2_gamma).

Model (per architecture, scene-level score y_{u,r,s}; u scene, r repeat, s seed):
    y = mu + gamma_u + delta_{T(u,r)} + (gamma*delta)_{u,r} + eps_{u,r,s}
  gamma  : scene difficulty (test-side)             -> var  s2_scene
  delta  : training-set composition (train-side)    -> var  s2_train   (one per repeat x fold)
  gd     : scene x training-set interaction          -> var  s2_inter   (one per scene x repeat)
  eps    : model seed noise                          -> var  s2_seed
Primary estimator: REML via statsmodels MixedLM with crossed variance components (single group).
Cross-check / bootstrap: method-of-moments on seed-averaged scores.

Finite-population test-side prediction for a random n-of-U scene partition (JOSA A design n=5, U=23):
    V_test = (1 - n/U) * s2_scene / n                        (closed form, scene-equal-weight score)
    V_test_fw = simulated variance of PSNR(frame-weighted mean MSE) over random 5-of-23 draws
Share  s = V_test / sigma2_part(JOSA A observed)   -> H1 decision.
Predicted JOSA A partition variance (K partitions x 6 seeds averaged):
    V_pred = V_test + s2_train + s2_inter/n + s2_seed_run/6

Usage:
    python analyze_cv.py <results_dir> <out_json> [--josaa-numbers numbers_josaa.tex] [--results-t4 dir]
"""
import argparse
import glob
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

U_SCENES = 23
N_TEST_JOSAA = 5
S_SEEDS_JOSAA = 6
Z_MDE = 1.959964 + 0.841621          # two-sided alpha=.05, power=.80
ARCH_TEX = {'direct': 'Direct', 'unet': 'Unet', 'transformer': 'Trans', 'proposed': 'Prop'}
METRICS = {   # scene-level score column -> (name, direction)
    'R_PSNR_of_mean_mse': 'R_PSNR (PSNR of scene-mean MSE, JOSA A convention)',
    'simse_R_gray_mean': 'si-MSE (gray, Grosse 2009)',
    'lmse_R_chen_mean': 'LMSE (Chen & Koltun 2013)',
    'dssim_R_mean': 'DSSIM (Chen & Koltun 2013)',
}


# ----------------------------------------------------------------------------
def load_runs(results_dir):
    rows, parity = [], []
    seen = set()
    for fp in sorted(glob.glob(str(Path(results_dir) / '**' / '*.json'), recursive=True)):
        name = Path(fp).name
        if name.startswith('cv_folds') or name in seen:
            continue
        seen.add(name)                       # dedup across shards (same tag -> identical content)
        d = json.load(open(fp))
        if d.get('kind') == 'parity':
            parity.append(d)
            continue
        if d.get('kind') != 'cv':
            continue
        for sc, ps in d['per_scene'].items():
            row = dict(arch=d['arch'], rep=d['rep'], fold=d['fold'], seed=d['seed'], scene=sc,
                       trainset=f"r{d['rep']}f{d['fold']}", scene_rep=f"{sc}|r{d['rep']}",
                       n_frames=ps['n_frames'], R_MSE_mean=ps['R_MSE_mean'])
            for k in METRICS:
                row[k] = ps[k]
            rows.append(row)
    return pd.DataFrame(rows), parity


def read_josaa_numbers(path):
    out = {}
    if not path or not Path(path).exists():
        return out
    txt = Path(path).read_text()
    for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', txt):
        out[m.group(1)] = m.group(2)
    return out


# ----------------------------------------------------------------------------
def reml_components(df, ycol):
    """Shared multi-start REML (analyze_cv2.reml_components); 2026-11-04 revision, see that docstring."""
    import analyze_cv2 as A2
    return A2.reml_components(df, ycol)


def moment_components(df, ycol):
    """Design-exact method of moments shared with analyze_cv2 (Henderson III on the actual incidence matrices); see that docstring."""
    import analyze_cv2 as A2
    d = df.copy()
    if 'trainset' not in d: d['trainset'] = 'r' + d['rep'].astype(str) + 'f' + d['fold'].astype(str)
    return A2.moment_components(d, ycol)


def run_level_seed_var(df, ycol):
    """Seed variance of the run-level (fold-mean, scene-equal-weight) score."""
    run = df.groupby(['rep', 'fold', 'seed'])[ycol].mean().reset_index()
    return float(run.groupby(['rep', 'fold'])[ycol].var(ddof=1).mean())


def simulate_frame_weighted_test_variance(df, n=N_TEST_JOSAA, n_draw=20000, seed=0):
    """Variance over random n-of-U scene draws of PSNR(frame-weighted mean MSE), using per-scene mean MSE
    averaged over repeats and seeds (test-side only; train-side held at its average)."""
    sc = df.groupby('scene').agg(mse=('R_MSE_mean', 'mean'), n=('n_frames', 'first')).reset_index()
    rng = np.random.RandomState(seed)
    mse, nfr = sc['mse'].values, sc['n'].values.astype(float)
    vals = np.empty(n_draw)
    for i in range(n_draw):
        idx = rng.choice(len(sc), size=n, replace=False)
        vals[i] = -10 * np.log10((mse[idx] * nfr[idx]).sum() / nfr[idx].sum() + 1e-10)
    eq = np.empty(n_draw)
    psnr_u = -10 * np.log10(mse + 1e-10)
    for i in range(n_draw):
        idx = rng.choice(len(sc), size=n, replace=False)
        eq[i] = psnr_u[idx].mean()
    return float(vals.var(ddof=1)), float(eq.var(ddof=1)), sc.assign(psnr=psnr_u).to_dict('records')


def bootstrap_share(df, ycol, sigma2_part_obs, n_boot=300, seed=0):
    """Two-way bootstrap (scenes and repeats with replacement) of the moment estimator -> share CI."""
    rng = np.random.RandomState(seed)
    scenes = sorted(df['scene'].unique()); reps = sorted(df['rep'].unique())
    shares, s2s = [], []
    for b in range(n_boot):
        bs = rng.choice(scenes, size=len(scenes), replace=True)
        br = rng.choice(reps, size=len(reps), replace=True)
        parts = []
        for i, s in enumerate(bs):
            for j, r in enumerate(br):
                sub = df[(df['scene'] == s) & (df['rep'] == r)].copy()
                sub['scene'] = f'{s}#{i}'; sub['rep'] = j
                parts.append(sub)
        bdf = pd.concat(parts, ignore_index=True)
        bdf['trainset'] = bdf['rep'].astype(str) + '_' + bdf['fold'].astype(str)
        try:
            mc = moment_components(bdf, ycol)
        except Exception:
            continue
        vt = (1 - N_TEST_JOSAA / U_SCENES) * mc['s2_scene'] / N_TEST_JOSAA
        s2s.append(mc['s2_scene'])
        shares.append(vt / sigma2_part_obs if sigma2_part_obs else float('nan'))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]
    return {'share_ci95': q(shares), 's2_scene_ci95': q(s2s), 'n_boot_ok': len(shares)}


# ----------------------------------------------------------------------------
def analyze_arch(df, arch, josaa):
    d = df[df['arch'] == arch].copy()
    tex = ARCH_TEX.get(arch)
    sigma2_part_obs = float(josaa.get(f'varPartScene{tex}', 'nan')) if tex else float('nan')
    sigma2_seed_obs = float(josaa.get(f'varSeedScene{tex}', 'nan')) if tex else float('nan')
    out = {'arch': arch, 'n_runs': int(d.groupby(['rep', 'fold', 'seed']).ngroups),
           'n_scene_rows': int(len(d)), 'reps': sorted(map(int, d['rep'].unique())),
           'seeds': sorted(map(int, d['seed'].unique())),
           'josaa_sigma2_part_obs': sigma2_part_obs, 'josaa_sigma2_seed_obs': sigma2_seed_obs,
           'metrics': {}}
    for ycol, label in METRICS.items():
        reml = reml_components(d, ycol)
        mom = moment_components(d, ycol)
        import analyze_cv2 as A2
        comp, used = A2.pick(reml, mom)
        s2_scene = comp['s2_scene']
        v_test = (1 - N_TEST_JOSAA / U_SCENES) * s2_scene / N_TEST_JOSAA
        s2_seed_run = run_level_seed_var(d, ycol); vz, ve = mom['v_zeta'], mom['v_eps']
        v_pred = v_test + comp['s2_train'] + comp['s2_inter'] / N_TEST_JOSAA + A2.seed_var_design(vz, ve, 1, S_SEEDS_JOSAA, False)
        icc_scene = s2_scene / (s2_scene + comp['s2_inter'] + comp['s2_seed']) if (s2_scene + comp['s2_inter'] + comp['s2_seed']) > 0 else float('nan')
        m = {'label': label, 'reml': reml, 'moment': mom, 'used': used, 'reml_consistency_ratio': A2.reml_consistency(reml, mom),
             'V_test_closed_form': v_test, 's2_seed_run_level': s2_seed_run, 'v_zeta': vz, 'v_eps': ve, 'V_pred_josaa_estimator': v_pred,
             'icc_scene_single_seed': icc_scene}
        if ycol == 'R_PSNR_of_mean_mse':
            v_fw, v_eq, scene_table = simulate_frame_weighted_test_variance(d)
            m['V_test_sim_frame_weighted'] = v_fw
            m['V_test_sim_scene_equal'] = v_eq
            m['scene_table'] = scene_table
            share = v_test / sigma2_part_obs if sigma2_part_obs == sigma2_part_obs else float('nan')
            share_fw = v_fw / sigma2_part_obs if sigma2_part_obs == sigma2_part_obs else float('nan')
            m['share_test_side_closed'] = share
            m['share_test_side_sim_fw'] = share_fw
            m['V_pred_over_observed'] = v_pred / sigma2_part_obs if sigma2_part_obs == sigma2_part_obs else float('nan')
            m['bootstrap'] = bootstrap_share(d, ycol, sigma2_part_obs)
            # pre-registered decisions
            dec = 'test-side dominated (s>=0.7)' if share >= 0.7 else (
                  'train-side/interaction dominated (s<=0.3)' if share <= 0.3 else 'mixed (0.3<s<0.7)')
            death = icc_scene < 0.5
            m['decision_H1'] = dec
            m['death_criterion_icc_lt_0.5'] = bool(death)
            # H3: implied S2_gamma under a pure test-side model vs estimated
            implied = sigma2_part_obs / ((1 - N_TEST_JOSAA / U_SCENES) / N_TEST_JOSAA)
            m['H3_implied_s2_gamma_pure_test_side'] = implied
            m['H3_estimated_s2_gamma'] = s2_scene
            m['H3_estimated_between_scene_sd_dB'] = math.sqrt(max(s2_scene, 0))
            # absolute-score MDE ladder (K partitions each side, S=3 seeds)
            lad = {}
            for K in [1, 3, 5, 10, 23]:
                var_part = v_test + comp['s2_train'] + comp['s2_inter'] / N_TEST_JOSAA
                lad[f'K{K}_S3'] = Z_MDE * math.sqrt(2 * (var_part / K + A2.seed_var_design(vz, ve, K, 3, False)))
            # full-coverage design: test-side term removed (n = U), train-side/inter remain per fold (exact coefficients, rounds 2 and 4)
            lad['fullcov_5fold_S3'] = Z_MDE * math.sqrt(2 * (A2.C_T * comp['s2_train'] + comp['s2_inter'] / U_SCENES + A2.seed_var_design(vz, ve, 5, 3, True)))
            m['MDE_absolute_dB'] = lad
        out['metrics'][ycol] = m
    return out


def analyze_pair(df, a, b, ycol='R_PSNR_of_mean_mse'):
    """H2: paired difference (b - a) per scene/rep/seed -> interaction variance and paired MDE."""
    da = df[df['arch'] == a].set_index(['scene', 'rep', 'fold', 'seed'])
    db = df[df['arch'] == b].set_index(['scene', 'rep', 'fold', 'seed'])
    common = da.index.intersection(db.index)
    if len(common) == 0:
        return None
    dd = (db.loc[common, ycol] - da.loc[common, ycol]).reset_index().rename(columns={ycol: 'y'})
    dd['arch'] = f'{b}-{a}'
    dd['trainset'] = 'r' + dd['rep'].astype(str) + 'f' + dd['fold'].astype(str)
    dd['scene_rep'] = dd['scene'] + '|r' + dd['rep'].astype(str)
    dd['n_frames'] = 0; dd['R_MSE_mean'] = 0.0
    reml = reml_components(dd, 'y')
    mom = moment_components(dd, 'y')
    import analyze_cv2 as A2
    comp, used = A2.pick(reml, mom)
    s2_seed_run = run_level_seed_var(dd, 'y'); vz, ve = mom['v_zeta'], mom['v_eps']
    s2_dg = comp['s2_scene']                      # Var_u(gamma_u^b - gamma_u^a): scene x method interaction
    mde = {}
    for K in [1, 3, 5]:
        se2 = ((1 - N_TEST_JOSAA / U_SCENES) * s2_dg / N_TEST_JOSAA + comp['s2_train'] + comp['s2_inter'] / N_TEST_JOSAA) / K + A2.seed_var_design(vz, ve, K, 3, False)
        mde[f'paired_K{K}_S3'] = Z_MDE * math.sqrt(se2)
    mde['paired_fullcov_5fold_S3'] = Z_MDE * math.sqrt(A2.C_T * comp['s2_train'] + comp['s2_inter'] / U_SCENES + A2.seed_var_design(vz, ve, 5, 3, True))
    per_scene_gap = dd.groupby('scene')['y'].mean().to_dict()
    return {'pair': f'{b} - {a}', 'metric': ycol, 'mean_gap': float(dd['y'].mean()), 'n_rows': int(len(dd)),
            'reml': reml, 'moment': mom, 'used': used, 'reml_consistency_ratio': A2.reml_consistency(reml, mom),
            's2_scene_x_method_interaction': s2_dg, 's2_seed_run_level': s2_seed_run, 'v_zeta_diff': vz, 'v_eps_diff': ve,
            'MDE_paired': mde, 'per_scene_gap': per_scene_gap,
            'share_interaction_in_paired_var_K1': ((1 - N_TEST_JOSAA / U_SCENES) * s2_dg / N_TEST_JOSAA) /
                max(1e-12, (1 - N_TEST_JOSAA / U_SCENES) * s2_dg / N_TEST_JOSAA + comp['s2_train'] + comp['s2_inter'] / N_TEST_JOSAA + A2.seed_var_design(vz, ve, 1, 3, False))}


def parity_report(parity, results_t4):
    rep = []
    for p in parity:
        ref = None
        if results_t4:
            for fp in glob.glob(str(Path(results_t4) / '**' / f"protocol_{p['arch']}_scene_p{p['partition']}_s{p['seed']}.json"), recursive=True):
                ref = json.load(open(fp)); break
        rep.append({'arch': p['arch'], 'partition': p['partition'], 'seed': p['seed'],
                    'new_R_PSNR_batchconv': p['agg']['R_PSNR_batchconv'],
                    'new_original_eval_R_PSNR': p['original_eval']['R_PSNR'],
                    'josaa_results_t4_R_PSNR': (ref or {}).get('R_PSNR'),
                    'diff_dB': (p['agg']['R_PSNR_batchconv'] - ref['R_PSNR']) if ref else None,
                    'minutes': p.get('minutes')})
    return rep


def josaa_partition_gap_validation(pairs, results_t4, splits_summary):
    """Out-of-sample check: per-scene UNet-DirectCNN gaps from the CV runs predict the JOSA A per-partition gaps."""
    p0 = next((p for p in pairs if p and p['metric'] == 'R_PSNR_of_mean_mse'), None)
    if not p0 or not Path(splits_summary).exists():
        return None
    gap = p0['per_scene_gap']
    S = json.load(open(splits_summary))
    obs = {}
    seen = set()
    for fp in glob.glob(str(Path(results_t4) / '**' / 'protocol_*_scene_p10*.json'), recursive=True):
        n = Path(fp).name
        if n in seen: continue
        seen.add(n); r = json.load(open(fp))
        obs.setdefault((r['arch'], r['partition']), []).append(r['R_PSNR'])
    rows, pred, act = [], [], []
    for p in [101, 102, 103, 104, 105]:
        ts = S.get(f'scene_p{p}', {}).get('test_scenes')
        if not ts or ('unet', p) not in obs or ('direct', p) not in obs: continue
        pr = float(np.mean([gap[s] for s in ts])); ob = float(np.mean(obs[('unet', p)]) - np.mean(obs[('direct', p)]))
        rows.append({'partition': p, 'test_scenes': ts, 'predicted_gap_dB': pr, 'josaa_observed_gap_dB': ob})
        pred.append(pr); act.append(ob)
    corr = float(np.corrcoef(pred, act)[0, 1]) if len(pred) > 2 else None
    return {'rows': rows, 'pearson_r': corr, 'sign_agreement': int(sum(np.sign(a) == np.sign(b) for a, b in zip(pred, act)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir')
    ap.add_argument('out_json')
    ap.add_argument('--josaa-numbers', default='/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/paper_josaa/numbers_josaa.tex')
    ap.add_argument('--results-t4', default='/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/results_t4')
    ap.add_argument('--splits-summary', default='/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/repo_sintel-scene-partition-variance/splits/SUMMARY.json')
    args = ap.parse_args()

    df, parity = load_runs(args.results_dir)
    josaa = read_josaa_numbers(args.josaa_numbers)
    out = {'version': 'v1', 'n_cv_runs': int(df.groupby(['arch', 'rep', 'fold', 'seed']).ngroups) if len(df) else 0,
           'archs': {}, 'pairs': [], 'parity': parity_report(parity, args.results_t4),
           'josaa_numbers_used': {k: v for k, v in josaa.items() if k.startswith('var')}}
    archs = sorted(df['arch'].unique()) if len(df) else []
    for a in archs:
        out['archs'][a] = analyze_arch(df, a, josaa)
    if 'direct' in archs and 'unet' in archs:
        for ycol in METRICS:
            out['pairs'].append(analyze_pair(df, 'direct', 'unet', ycol))
    # test-side share of the *predicted* partition variance, per metric (JOSA A observed exists for PSNR only)
    for a in archs:
        for ycol, m in out['archs'][a]['metrics'].items():
            c = m['reml'] if m['used'] == 'reml' else m['moment']
            tot = m['V_pred_josaa_estimator']
            m['share_test_side_of_predicted'] = m['V_test_closed_form'] / tot if tot > 0 else float('nan')
    out['josaa_partition_gap_validation'] = josaa_partition_gap_validation(out['pairs'], args.results_t4, args.splits_summary)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))

    # console summary
    print(f"cv runs: {out['n_cv_runs']}  archs: {archs}")
    for a in archs:
        m = out['archs'][a]['metrics']['R_PSNR_of_mean_mse']
        c = m['reml'] if m['used'] == 'reml' else m['moment']
        print(f"\n[{a}] runs={out['archs'][a]['n_runs']}  est={m['used']}")
        print(f"  s2_scene={c['s2_scene']:.4f}  s2_train={c['s2_train']:.4f}  s2_inter={c['s2_inter']:.4f}  s2_seed={c['s2_seed']:.4f}")
        print(f"  V_test closed={m['V_test_closed_form']:.4f}  sim(fw)={m['V_test_sim_frame_weighted']:.4f}  "
              f"JOSA A sigma2_part={out['archs'][a]['josaa_sigma2_part_obs']}")
        print(f"  share closed={m['share_test_side_closed']:.3f}  share sim(fw)={m['share_test_side_sim_fw']:.3f}  "
              f"CI={m['bootstrap']['share_ci95']}  V_pred/obs={m['V_pred_over_observed']:.3f}")
        print(f"  H1: {m['decision_H1']}   death(icc<0.5)={m['death_criterion_icc_lt_0.5']} (icc={m['icc_scene_single_seed']:.3f})")
        print(f"  H3: implied S2_gamma={m['H3_implied_s2_gamma_pure_test_side']:.3f} vs estimated={m['H3_estimated_s2_gamma']:.3f} "
              f"(sd {m['H3_estimated_between_scene_sd_dB']:.2f} dB)")
        print(f"  MDE abs: " + ', '.join(f"{k}={v:.2f}" for k, v in m['MDE_absolute_dB'].items()))
    for a in archs:
        print(f"\n[{a}] test-side share of predicted partition variance, by metric:")
        for ycol, m in out['archs'][a]['metrics'].items():
            c = m['reml'] if m['used'] == 'reml' else m['moment']
            print(f"   {METRICS[ycol][:36]:36s} share={m['share_test_side_of_predicted']:.2f}  icc1={m['icc_scene_single_seed']:.2f}  "
                  f"(scene {c['s2_scene']:.3g} train {c['s2_train']:.3g} inter {c['s2_inter']:.3g} seed {c['s2_seed']:.3g})")
    for p in out['pairs']:
        if p:
            print(f"\n[pair {p['pair']} | {METRICS[p['metric']][:30]}] gap={p['mean_gap']:.4g}  "
                  f"s2_scene_x_method={p['s2_scene_x_method_interaction']:.4g}  interaction share(K1,S3)={p['share_interaction_in_paired_var_K1']:.2f}  "
                  + ', '.join(f"{k}={v:.3g}" for k, v in p['MDE_paired'].items()))
    for r in out['parity']:
        print(f"\n[parity {r['arch']} p{r['partition']} s{r['seed']}] new={r['new_R_PSNR_batchconv']:.3f}  "
              f"josaa={r['josaa_results_t4_R_PSNR']}  diff={r['diff_dB']}")
    v = out.get('josaa_partition_gap_validation')
    if v:
        print(f"\n[JOSA A per-partition gap validation] pearson r={v['pearson_r']:.3f}, sign agreement {v['sign_agreement']}/{len(v['rows'])}")
        for r in v['rows']:
            print(f"   p{r['partition']}: predicted {r['predicted_gap_dB']:+.2f} | observed {r['josaa_observed_gap_dB']:+.2f}")


if __name__ == '__main__':
    main()
