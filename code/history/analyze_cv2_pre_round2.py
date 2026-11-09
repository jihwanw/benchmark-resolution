#!/usr/bin/env python3
"""analyze_cv2.py — pre-registered analysis of the benchmark-resolution MAIN experiment (v2).

Pre-registration: PhD/AI/CV/ideas/2026-11-01_benchmark_resolution.md ("본실험 설계 (사전 등록 갱신, 2026-11-02)").
Primary estimands: scale-invariant metrics (si-MSE gray, LMSE Chen&Koltun, DSSIM); secondary: PSNR.

Per architecture and metric (random design, scene-level score y_{u,r,s}):
    y = mu + gamma_u (scene) + delta_T (training set = rep x fold) + (gamma*delta)_{u,r} + eps (seed)
  REML (statsmodels MixedLM, crossed VCs) with method-of-moments cross-check.
  -> test-side finite-population variance V_test=(1-n/U) s2_scene/n, share of predicted partition variance,
     ICC(scene), absolute MDE ladder.
H2  all 6 architecture pairs: scene x method interaction variance and paired MDE (K=1/3/5, full-coverage).
H3  out-of-sample: per-scene pair gaps predict the JOSA A per-partition gaps (results_t4, 6 pairs x 5 partitions).
H4  gauge: decomposition of PSNR vs scene-gain-aligned PSNR (R_siPSNR_scenegain_rgb); interaction reduction.
H5  TransformerUNet seed variance (scene- and run-level) vs the other architectures.
H6  stratified vs random partitions: simulated V_test from the per-scene table (both designs) and the observed
    fold-mean variance of the stratified repeats (direct/unet) vs random repeats.
Control: GPU non-determinism — v2 vs v1 same-configuration runs (direct/unet, seeds 42/43, random design).
"""
import argparse, glob, json, math, warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
U = 23; N_TEST = 5; S_JOSAA = 6
Z_MDE = 1.959964 + 0.841621
ARCH_TEX = {'direct': 'Direct', 'unet': 'Unet', 'transformer': 'Trans', 'proposed': 'Prop'}
PRIMARY = {'simse_R_gray_mean': 'si-MSE (gray, Grosse 2009)', 'lmse_R_chen_mean': 'LMSE (Chen & Koltun 2013)',
           'dssim_R_mean': 'DSSIM (Chen & Koltun 2013)'}
SECONDARY = {'R_PSNR_of_mean_mse': 'R_PSNR (PSNR of scene-mean MSE)', 'R_siPSNR_scenegain_rgb': 'PSNR after per-scene optimal RGB gain (H4)'}
METRICS = {**PRIMARY, **SECONDARY}


def load_runs(results_dir):
    rows, seen = [], set()
    for fp in sorted(glob.glob(str(Path(results_dir) / '**' / 'cv2_*_v2.json'), recursive=True)):
        name = Path(fp).name
        if name in seen: continue
        seen.add(name); d = json.load(open(fp))
        if d.get('kind') != 'cv': continue
        for sc, ps in d['per_scene'].items():
            row = dict(arch=d['arch'], design=d['design'], rep=d['rep'], fold=d['fold'], seed=d['seed'], scene=sc,
                       trainset=f"{d['design']}r{d['rep']}f{d['fold']}", scene_rep=f"{sc}|{d['design']}r{d['rep']}",
                       n_frames=ps['n_frames'], R_MSE_mean=ps['R_MSE_mean'], minutes=d['minutes'])
            for k in METRICS: row[k] = ps.get(k, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def load_v1(results_dir_v1):
    rows, seen = [], set()
    for fp in sorted(glob.glob(str(Path(results_dir_v1) / '**' / 'cv_*_v1.json'), recursive=True)):
        name = Path(fp).name
        if 'folds' in name or name in seen: continue
        seen.add(name); d = json.load(open(fp))
        if d.get('kind') != 'cv': continue
        for sc, ps in d['per_scene'].items():
            rows.append(dict(arch=d['arch'], rep=d['rep'], fold=d['fold'], seed=d['seed'], scene=sc,
                             **{k: ps.get(k, np.nan) for k in PRIMARY}, R_PSNR_of_mean_mse=ps['R_PSNR_of_mean_mse']))
    return pd.DataFrame(rows)


def read_tex_numbers(path):
    import re
    out = {}
    if path and Path(path).exists():
        for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', Path(path).read_text()): out[m.group(1)] = m.group(2)
    return out


# ----------------------------------------------------------------------------
REML_METHODS = (['lbfgs', 'powell'], ['powell'], ['nm'], ['bfgs'], ['cg'])   # several optimiser paths; best REML llf wins
REML_CONSISTENCY_BAND = (0.80, 1.25)   # accepted ratio of REML-implied to observed variance of scene means (see pick())


def reml_components(df, ycol):
    """Crossed variance components by REML (statsmodels MixedLM, single group, three VCs).
    2026-11-04 revision: the optimiser is run from several method paths and the converged fit with the highest
    restricted log-likelihood is kept. Before this, a single path ('lbfgs' then 'powell') was used and in three fits
    (flow out3/raft_large, flow Fl-all/raft_small, intrinsic si-MSE/unet) it stopped at a solution whose implied variance
    of scene means was 2-7x below the observed one; pick() now also tests this consistency against the moment estimator."""
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return None
    d = df[['scene', 'trainset', 'scene_rep', ycol]].dropna().copy().rename(columns={ycol: 'y'}); d['g'] = 1
    if len(d) < 10: return None
    best, errors = None, []
    for methods in REML_METHODS:
        try:
            md = smf.mixedlm('y ~ 1', d, groups='g', re_formula='0',
                             vc_formula={'scene': '0 + C(scene)', 'trainset': '0 + C(trainset)', 'inter': '0 + C(scene_rep)'})
            fit = md.fit(reml=True, method=methods, maxiter=3000)
            comp = {n: float(v) for n, v in zip(list(md.exog_vc.names), fit.vcomp)}
            cand = dict(mu=float(fit.params['Intercept']), s2_scene=comp.get('scene', np.nan), s2_train=comp.get('trainset', np.nan),
                        s2_inter=comp.get('inter', np.nan), s2_seed=float(fit.scale), converged=bool(fit.converged),
                        llf=float(fit.llf), method='+'.join(methods))
            if not all(np.isfinite([cand['s2_scene'], cand['s2_train'], cand['s2_inter'], cand['s2_seed'], cand['llf']])): continue
            if best is None or (cand['converged'] and not best['converged']) or (cand['converged'] == best['converged'] and cand['llf'] > best['llf']):
                best = cand
        except Exception as e:
            errors.append(f'{"+".join(methods)}: {e}')
    if best is None: return {'error': '; '.join(errors) or 'no fit'}
    best['n_paths_tried'] = len(REML_METHODS)
    return best


def moment_components(df, ycol):
    d = df.dropna(subset=[ycol])
    S = d.groupby(['scene', 'rep', 'fold'])['seed'].nunique().mean()
    g = d.groupby(['scene', 'rep', 'fold'])[ycol]
    s2_seed = float(g.var(ddof=1).mean()) if S > 1 else float('nan')
    m = g.mean().reset_index(); m['trainset'] = m['rep'].astype(str) + '_' + m['fold'].astype(str)
    R = m.groupby('scene')['rep'].nunique().mean(); nf = m.groupby('trainset')['scene'].count().mean()
    m['e'] = m[ycol] - m.groupby('scene')[ycol].transform('mean')
    m['d'] = m.groupby('trainset')['e'].transform('mean'); m['w'] = m['e'] - m['d']
    seed_term = (s2_seed / S) if S > 1 else 0.0
    dof = max(1, len(m) - m['trainset'].nunique() - m['scene'].nunique() + 1)
    s2_inter = max(0.0, float((m['w'] ** 2).sum() / dof) - seed_term)
    m['c'] = m[ycol] - m.groupby('rep')[ycol].transform('mean')
    var_d = float(m.groupby('trainset')['c'].mean().var(ddof=1))
    s2_scene_raw = float(m.groupby('scene')[ycol].mean().var(ddof=1))
    s2_scene = max(0.0, s2_scene_raw - var_d / R)
    s2_train = max(0.0, var_d - (1 - nf / U) * s2_scene / nf - s2_inter / nf - seed_term / nf)
    s2_scene = max(0.0, s2_scene_raw - (s2_train + s2_inter + seed_term) / R)
    return dict(s2_scene=s2_scene, s2_train=s2_train, s2_inter=s2_inter, s2_seed=s2_seed, s2_scene_raw=s2_scene_raw,
                R=float(R), nf=float(nf), S=float(S))


def reml_consistency(reml, mom):
    """Ratio of the REML-implied variance of scene means, s2_scene + (s2_train + s2_inter + s2_seed/S)/R (eq. mom1),
    to the observed variance of scene means (mom['s2_scene_raw']). A fit far from 1 is a spurious optimum."""
    if not reml or 'error' in reml or not mom or not mom.get('s2_scene_raw'): return float('nan')
    implied = reml['s2_scene'] + (reml['s2_train'] + reml['s2_inter'] + reml['s2_seed'] / mom['S']) / mom['R']
    return float(implied / mom['s2_scene_raw']) if mom['s2_scene_raw'] > 0 else float('nan')


def pick(reml, mom):
    """REML if it exists, is finite and is consistent with the observed variance of scene means; otherwise moments.
    The 'used' string records why."""
    ok = reml and 'error' not in reml and all(np.isfinite([reml['s2_scene'], reml['s2_train'], reml['s2_inter']]))
    if not ok: return mom, 'moment (reml failed)'
    ratio = reml_consistency(reml, mom)
    if np.isfinite(ratio) and not (REML_CONSISTENCY_BAND[0] <= ratio <= REML_CONSISTENCY_BAND[1]):
        return mom, f'moment (reml inconsistent, ratio={ratio:.2f})'
    return reml, 'reml'


def run_level_seed_var(df, ycol):
    run = df.dropna(subset=[ycol]).groupby(['rep', 'fold', 'seed'])[ycol].mean().reset_index()
    return float(run.groupby(['rep', 'fold'])[ycol].var(ddof=1).mean())


def scene_table(df, ycol):
    return df.dropna(subset=[ycol]).groupby('scene').agg(score=(ycol, 'mean'), n=('n_frames', 'first'), mse=('R_MSE_mean', 'mean')).reset_index()


def simulate_designs(tab, strata, n=N_TEST, n_draw=20000, seed=0):
    """Test-side variance of the fold mean (scene-equal-weight) for random n-of-U draws vs stratified systematic folds."""
    rng = np.random.RandomState(seed)
    sc = tab['scene'].tolist(); score = tab['score'].values
    vals = np.array([score[rng.choice(len(sc), size=n, replace=False)].mean() for _ in range(n_draw)])
    strat = strata['stratum_of']
    svals = []
    for _ in range(n_draw // 5):
        order = []
        for k in sorted(set(strat.values())):
            mem = [s for s in sc if strat[s] == k]; rng.shuffle(mem); order.extend(mem)
        idx = {s: i for i, s in enumerate(sc)}
        for f in range(5):
            fold = order[f::5]; svals.append(score[[idx[s] for s in fold]].mean())
    svals = np.array(svals)
    return float(vals.var(ddof=1)), float(svals.var(ddof=1))


# ----------------------------------------------------------------------------
def analyze_arch(df, arch, josaa, strata):
    d = df[(df['arch'] == arch) & (df['design'] == 'rand')].copy()
    tex = ARCH_TEX.get(arch)
    out = {'arch': arch, 'n_runs': int(d.groupby(['rep', 'fold', 'seed']).ngroups), 'reps': sorted(map(int, d['rep'].unique())),
           'seeds': sorted(map(int, d['seed'].unique())), 'mean_minutes': float(d.groupby(['rep', 'fold', 'seed'])['minutes'].first().mean()),
           'josaa_sigma2_part_obs_psnr': float(josaa.get(f'varPartScene{tex}', 'nan')), 'josaa_sigma2_seed_obs_psnr': float(josaa.get(f'varSeedScene{tex}', 'nan')),
           'josaa_rho_psnr': float(josaa.get(f'varRatioScene{tex}', 'nan')), 'metrics': {}}
    for ycol, label in METRICS.items():
        if d[ycol].isna().all(): continue
        reml = reml_components(d, ycol); mom = moment_components(d, ycol); comp, used = pick(reml, mom)
        s2_scene, s2_train, s2_inter, s2_seed = comp['s2_scene'], comp['s2_train'], comp['s2_inter'], comp['s2_seed']
        v_test = (1 - N_TEST / U) * s2_scene / N_TEST
        s2_seed_run = run_level_seed_var(d, ycol)
        v_pred = v_test + s2_train + s2_inter / N_TEST + s2_seed_run / S_JOSAA
        denom = s2_scene + s2_inter + s2_seed
        tab = scene_table(d, ycol)
        v_rand_sim, v_strat_sim = simulate_designs(tab, strata)
        m = {'label': label, 'used': used, 'reml': reml, 'moment': mom, 'reml_consistency_ratio': reml_consistency(reml, mom),
             'V_test_closed_form': v_test, 's2_seed_run_level': s2_seed_run, 'V_pred_partition_var': v_pred,
             'share_test_side_of_predicted': v_test / v_pred if v_pred > 0 else np.nan,
             'icc_scene_single_seed': s2_scene / denom if denom > 0 else np.nan,
             'scene_table': tab.to_dict('records'),
             'H6_sim_V_test_random': v_rand_sim, 'H6_sim_V_test_stratified': v_strat_sim,
             'H6_sim_variance_reduction': 1 - v_strat_sim / v_rand_sim if v_rand_sim > 0 else np.nan}
        lad = {}
        for K in [1, 3, 5, 10, 23]:
            lad[f'K{K}_S3'] = Z_MDE * math.sqrt(2 * ((v_test + s2_train + s2_inter / N_TEST) / K + s2_seed_run / (3 * K)))
        lad['fullcov_5fold_S3'] = Z_MDE * math.sqrt(2 * ((s2_train + s2_inter / N_TEST) / 5 + s2_seed_run / 15))
        m['MDE_absolute'] = lad
        if ycol == 'R_PSNR_of_mean_mse' and np.isfinite(out['josaa_sigma2_part_obs_psnr']):
            m['share_test_side_vs_josaa_observed'] = v_test / out['josaa_sigma2_part_obs_psnr']
            m['V_pred_over_josaa_observed'] = v_pred / out['josaa_sigma2_part_obs_psnr']
        out['metrics'][ycol] = m
    # H4: gauge — interaction under PSNR vs scene-gain-aligned PSNR
    a, b = out['metrics'].get('R_PSNR_of_mean_mse'), out['metrics'].get('R_siPSNR_scenegain_rgb')
    if a and b:
        ca = a['reml'] if a['used'] == 'reml' else a['moment']; cb = b['reml'] if b['used'] == 'reml' else b['moment']
        red = 1 - cb['s2_inter'] / ca['s2_inter'] if ca['s2_inter'] > 0 else np.nan
        out['H4_gauge'] = {'s2_inter_psnr': ca['s2_inter'], 's2_inter_scenegain_psnr': cb['s2_inter'],
                           'interaction_reduction': red, 'passes_pre_registered_50pct': bool(red >= 0.5) if np.isfinite(red) else None,
                           'icc_psnr': a['icc_scene_single_seed'], 'icc_scenegain_psnr': b['icc_scene_single_seed']}
    # H6 observed: stratified vs random fold-mean variance (same arch, scene-equal-weight), by metric
    ds = df[(df['arch'] == arch) & (df['design'] == 'strat')]
    if len(ds):
        obs = {}
        for ycol in METRICS:
            if d[ycol].isna().all() or ds[ycol].isna().all(): continue
            fr = d.groupby(['rep', 'fold', 'seed'])[ycol].mean().groupby(level=[0, 1]).mean()     # fold means (seed-avg) random
            fs = ds.groupby(['rep', 'fold', 'seed'])[ycol].mean().groupby(level=[0, 1]).mean()    # stratified
            # within-repeat variance of fold means (removes repeat-level offsets), averaged over repeats
            vr = float(fr.groupby(level=0).var(ddof=1).mean()); vs = float(fs.groupby(level=0).var(ddof=1).mean())
            obs[ycol] = {'var_fold_means_random': vr, 'var_fold_means_stratified': vs, 'n_rand_reps': int(fr.index.get_level_values(0).nunique()),
                         'n_strat_reps': int(fs.index.get_level_values(0).nunique()), 'reduction': 1 - vs / vr if vr > 0 else np.nan}
        out['H6_observed'] = obs
    return out


def analyze_pair(df, a, b, ycol):
    da = df[(df['arch'] == a) & (df['design'] == 'rand')].set_index(['scene', 'rep', 'fold', 'seed'])
    db = df[(df['arch'] == b) & (df['design'] == 'rand')].set_index(['scene', 'rep', 'fold', 'seed'])
    common = da.index.intersection(db.index)
    if len(common) == 0: return None
    dd = (db.loc[common, ycol] - da.loc[common, ycol]).reset_index().rename(columns={ycol: 'y'}).dropna()
    dd['trainset'] = 'r' + dd['rep'].astype(str) + 'f' + dd['fold'].astype(str); dd['scene_rep'] = dd['scene'] + '|r' + dd['rep'].astype(str)
    dd['n_frames'] = 0; dd['R_MSE_mean'] = 0.0
    reml = reml_components(dd, 'y'); mom = moment_components(dd, 'y'); comp, used = pick(reml, mom)
    s2_seed_run = run_level_seed_var(dd, 'y'); s2_dg = comp['s2_scene']
    mde = {}
    for K in [1, 3, 5]:
        se2 = ((1 - N_TEST / U) * s2_dg / N_TEST + comp['s2_train'] + comp['s2_inter'] / N_TEST) / K + s2_seed_run / (3 * K)
        mde[f'paired_K{K}_S3'] = Z_MDE * math.sqrt(se2)
    mde['paired_fullcov_5fold_S3'] = Z_MDE * math.sqrt((comp['s2_train'] + comp['s2_inter'] / N_TEST) / 5 + s2_seed_run / 15)
    tot = (1 - N_TEST / U) * s2_dg / N_TEST + comp['s2_train'] + comp['s2_inter'] / N_TEST + s2_seed_run / 3
    gap = float(dd['y'].mean())
    return {'pair': f'{b} - {a}', 'a': a, 'b': b, 'metric': ycol, 'mean_gap': gap, 'n_rows': int(len(dd)), 'used': used, 'reml': reml, 'moment': mom,
            'reml_consistency_ratio': reml_consistency(reml, mom),
            's2_scene_x_method': s2_dg, 's2_train_diff': comp['s2_train'], 's2_inter_diff': comp['s2_inter'], 's2_seed_run_level': s2_seed_run,
            'share_interaction_K1S3': ((1 - N_TEST / U) * s2_dg / N_TEST) / tot if tot > 0 else np.nan,
            'MDE_paired': mde, 'gap_over_MDE_K1': abs(gap) / mde['paired_K1_S3'] if mde['paired_K1_S3'] > 0 else np.nan,
            'gap_over_MDE_fullcov': abs(gap) / mde['paired_fullcov_5fold_S3'] if mde['paired_fullcov_5fold_S3'] > 0 else np.nan,
            'per_scene_gap': dd.groupby('scene')['y'].mean().to_dict()}


def josaa_validation(pairs, results_t4, splits_summary):
    """H3: per-scene gaps (PSNR) predict JOSA A per-partition gaps for all pairs with results_t4 data."""
    if not Path(splits_summary).exists(): return None
    S = json.load(open(splits_summary)); obs, seen = {}, set()
    for fp in glob.glob(str(Path(results_t4) / '**' / 'protocol_*_scene_p10*.json'), recursive=True):
        n = Path(fp).name
        if n in seen: continue
        seen.add(n); r = json.load(open(fp)); obs.setdefault((r['arch'], r['partition']), []).append(r['R_PSNR'])
    rows = []
    for p in pairs:
        if not p or p['metric'] != 'R_PSNR_of_mean_mse': continue
        g = p['per_scene_gap']
        for part in [101, 102, 103, 104, 105]:
            ts = S.get(f'scene_p{part}', {}).get('test_scenes')
            if not ts or (p['a'], part) not in obs or (p['b'], part) not in obs: continue
            rows.append({'pair': p['pair'], 'partition': part, 'predicted_gap_dB': float(np.mean([g[s] for s in ts])),
                         'observed_gap_dB': float(np.mean(obs[(p['b'], part)]) - np.mean(obs[(p['a'], part)]))})
    if len(rows) < 3: return {'rows': rows}
    pr = np.array([r['predicted_gap_dB'] for r in rows]); ob = np.array([r['observed_gap_dB'] for r in rows])
    return {'rows': rows, 'n': len(rows), 'pearson_r': float(np.corrcoef(pr, ob)[0, 1]),
            'sign_agreement': int((np.sign(pr) == np.sign(ob)).sum()), 'rmse_dB': float(np.sqrt(np.mean((pr - ob) ** 2)))}


def nondeterminism_control(df, df1):
    """Same configuration (arch, rep, fold, seed, scene) run in v1 and v2: sd of the difference = GPU non-determinism floor."""
    if df1 is None or not len(df1): return None
    a = df[(df['design'] == 'rand') & (df['arch'].isin(['direct', 'unet'])) & (df['seed'].isin([42, 43]))].set_index(['arch', 'rep', 'fold', 'seed', 'scene'])
    b = df1.set_index(['arch', 'rep', 'fold', 'seed', 'scene'])
    common = a.index.intersection(b.index)
    out = {'n_common_scene_rows': int(len(common))}
    for ycol in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean']:
        diff = (a.loc[common, ycol] - b.loc[common, ycol]).dropna()
        out[ycol] = {'sd_diff_scene_level': float(diff.std(ddof=1)), 'mean_abs_diff': float(diff.abs().mean()),
                     'nondet_var_single_run': float(diff.var(ddof=1) / 2)}      # var(x1-x2)=2*var_nondet
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir'); ap.add_argument('out_json')
    ap.add_argument('--strata', default=str(Path(__file__).parent / 'strata_v2.json'))
    ap.add_argument('--results-v1', default='/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution/results_cv_v1')
    ap.add_argument('--josaa-numbers', default='/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/paper_josaa/numbers_josaa.tex')
    ap.add_argument('--results-t4', default='/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/results_t4')
    ap.add_argument('--splits-summary', default='/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/repo_sintel-scene-partition-variance/splits/SUMMARY.json')
    args = ap.parse_args()

    df = load_runs(args.results_dir); strata = json.load(open(args.strata)); josaa = read_tex_numbers(args.josaa_numbers)
    df1 = load_v1(args.results_v1) if Path(args.results_v1).exists() else None
    archs = sorted(df['arch'].unique()) if len(df) else []
    out = {'version': 'v2', 'n_runs': int(df.groupby(['arch', 'design', 'rep', 'fold', 'seed']).ngroups) if len(df) else 0,
           'n_runs_by_design': df.groupby('design').apply(lambda g: g.groupby(['arch', 'rep', 'fold', 'seed']).ngroups).to_dict() if len(df) else {},
           'archs': {a: analyze_arch(df, a, josaa, strata) for a in archs}, 'pairs': []}
    for a, b in combinations(archs, 2):
        for ycol in METRICS:
            r = analyze_pair(df, a, b, ycol)
            if r: out['pairs'].append(r)
    out['H3_josaa_validation'] = josaa_validation(out['pairs'], args.results_t4, args.splits_summary)
    out['nondeterminism_control'] = nondeterminism_control(df, df1)
    # H5: seed variance comparison across architectures (PSNR and si-MSE)
    def _picked(a, y):
        mm = out['archs'][a]['metrics'][y]; return mm['reml'] if mm['used'] == 'reml' else mm['moment']
    out['H5_seed_variance'] = {a: {y: {'s2_seed_scene_level': _picked(a, y)['s2_seed'],
                                       's2_seed_run_level': out['archs'][a]['metrics'][y]['s2_seed_run_level'],
                                       's2_scene': _picked(a, y)['s2_scene']}
                                   for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean'] if y in out['archs'][a]['metrics']} for a in archs}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))

    print(f"runs: {out['n_runs']} {out['n_runs_by_design']}  archs: {archs}")
    for a in archs:
        A = out['archs'][a]
        print(f"\n[{a}] runs={A['n_runs']} mean {A['mean_minutes']:.1f} min")
        for y, m in A['metrics'].items():
            c = m['reml'] if m['used'] == 'reml' else m['moment']
            extra = f"  share_vs_JOSAA={m['share_test_side_vs_josaa_observed']:.2f} Vpred/obs={m['V_pred_over_josaa_observed']:.2f}" if 'share_test_side_vs_josaa_observed' in m else ''
            print(f"  {m['label'][:40]:40s} [{m['used']}] scene {c['s2_scene']:.3g} train {c['s2_train']:.3g} inter {c['s2_inter']:.3g} seed {c['s2_seed']:.3g} | "
                  f"test-share {m['share_test_side_of_predicted']:.2f} icc {m['icc_scene_single_seed']:.2f} | H6 sim red {m['H6_sim_variance_reduction']:.2f}{extra}")
        if 'H4_gauge' in A:
            h = A['H4_gauge']; print(f"  H4 gauge: inter PSNR {h['s2_inter_psnr']:.3g} -> scene-gain PSNR {h['s2_inter_scenegain_psnr']:.3g} (reduction {h['interaction_reduction']:.2f}, pass50={h['passes_pre_registered_50pct']}); icc {h['icc_psnr']:.2f}->{h['icc_scenegain_psnr']:.2f}")
        if 'H6_observed' in A:
            for y, o in A['H6_observed'].items():
                print(f"  H6 obs {y[:22]:22s}: var fold-means random {o['var_fold_means_random']:.3g} vs strat {o['var_fold_means_stratified']:.3g} (reduction {o['reduction']:.2f}; reps {o['n_rand_reps']}/{o['n_strat_reps']})")
    print("\n[H2 pairs] metric | pair | gap | s2(scene x method) | inter share K1 | MDE K1 | MDE fullcov | gap/MDE_K1")
    for p in out['pairs']:
        print(f"  {p['metric'][:16]:16s} {p['pair']:24s} {p['mean_gap']:+.4g} | {p['s2_scene_x_method']:.3g} | {p['share_interaction_K1S3']:.2f} | "
              f"{p['MDE_paired']['paired_K1_S3']:.3g} | {p['MDE_paired']['paired_fullcov_5fold_S3']:.3g} | {p['gap_over_MDE_K1']:.2f}")
    v = out['H3_josaa_validation']
    if v and 'pearson_r' in v: print(f"\n[H3 JOSA A validation] n={v['n']} r={v['pearson_r']:.3f} sign agreement {v['sign_agreement']}/{v['n']} rmse {v['rmse_dB']:.2f} dB")
    nd = out['nondeterminism_control']
    if nd: print(f"\n[non-determinism control] n={nd['n_common_scene_rows']} " + ' '.join(f"{k[:10]}: sd_diff={v['sd_diff_scene_level']:.3g}" for k, v in nd.items() if isinstance(v, dict)))


if __name__ == '__main__':
    main()
