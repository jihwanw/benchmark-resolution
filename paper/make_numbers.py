#!/usr/bin/env python3
"""make_numbers.py — generate numbers.tex (LaTeX macros) for the IJCV manuscript from the result files.
Rule: no number in the manuscript is typed by hand; every macro below traces to a JSON key.

Sources
  results/benchmark_resolution/killing_v1.json   (killing experiment, direct+unet, S=2)
  results/benchmark_resolution/main_v2.json      (main experiment, 4 archs, S=3; H4-H6; non-determinism; JOSA A validation)
  results/benchmark_resolution/flow_v1.json      (optical-flow arm, raft_small/large; F-H1..F-H4)
  JEI_IntrinsicUQ/revision_2026-08/paper_josaa/numbers_josaa.tex (JOSA A observed partition/seed variances)
"""
import json, math, re, sys
import numpy as np
from pathlib import Path

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
J = Path('/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/revision_2026-08/paper_josaa/numbers_josaa.tex')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'numbers.tex'

k1 = json.load(open(R / 'killing_v1.json')); m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
josaa = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', J.read_text()))

M = {}
DIG = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
def mac(name, val, fmt='{:.2f}'):
    name = re.sub(r'\d', lambda m: DIG[m.group(0)], name)   # digits -> words (LaTeX macro names must be letters)
    name = re.sub(r'[^A-Za-z]', '', name)
    M[name] = fmt.format(val) if isinstance(val, (int, float)) else str(val)
def pval(p):
    """p-value text for the manuscript (round 2, A5): '<10^{-4}' below that, three decimals below 0.01, two otherwise; without the leading 'p'."""
    if p < 1e-4: return '<10^{-4}'
    if p < 1e-3: return '<0.001'
    return f'={p:.3f}' if p < 0.01 else f'={p:.2f}'

ARCH = {'direct': 'Direct', 'unet': 'Unet', 'transformer': 'Trans', 'proposed': 'Prop', 'raft_small': 'RaftS', 'raft_large': 'RaftL'}
MET = {'simse_R_gray_mean': 'Simse', 'lmse_R_chen_mean': 'Lmse', 'dssim_R_mean': 'Dssim', 'R_PSNR_of_mean_mse': 'Psnr',
       'R_siPSNR_scenegain_rgb': 'GainPsnr', 'epe_mean': 'Epe', 'fl_all_mean': 'Flall', 'out3_mean': 'Outthree', 'epe_occ_mean': 'EpeOcc', 'epe_noc_mean': 'EpeNoc', 'epe_pixel_weighted': 'EpePix'}

# --- design constants
mac('nScenes', 23, '{:d}'); mac('nFramesSintel', 1064, '{:d}'); mac('nPairsFlow', 1041, '{:d}')
mac('nRunsKilling', k1['n_cv_runs'], '{:d}'); mac('nRunsMain', m2['n_runs'], '{:d}'); mac('nRunsMainRand', m2['n_runs_by_design'].get('rand', 0), '{:d}')
mac('nRunsMainStrat', m2['n_runs_by_design'].get('strat', 0), '{:d}'); mac('nRunsFlow', fl['n_runs'], '{:d}')
mac('nRunsTotal', k1['n_cv_runs'] + 2 + m2['n_runs'] + fl['n_runs'] + 2, '{:d}')

# --- JOSA A observed (PSNR) partition/seed variances and ratios
for a, t in [('direct', 'Direct'), ('unet', 'Unet'), ('transformer', 'Trans'), ('proposed', 'Prop')]:
    mac(f'josaaVarPart{t}', float(josaa[f'varPartScene{t}']), '{:.3f}'); mac(f'josaaVarSeed{t}', float(josaa[f'varSeedScene{t}']), '{:.3f}')
    mac(f'josaaRho{t}', float(josaa[f'varRatioScene{t}']), '{:.1f}')
mac('josaaFpSceneMax', max(float(josaa[f'fpScene{t}']) for t in ['Direct', 'Unet', 'Trans', 'Prop']), '{:.1f}')
mac('josaaFpSceneMin', min(float(josaa[f'fpScene{t}']) for t in ['Direct', 'Unet', 'Trans', 'Prop']), '{:.1f}')
mac('josaaArchFlipPairs', josaa['archFlipPairs']); mac('josaaArchPairTotal', josaa['archPairTotal'])

# --- main experiment: per arch x metric decomposition
for a, A in m2['archs'].items():
    mac(f'mainMinutes{ARCH[a]}', A['mean_minutes'], '{:.1f}')
    for y, m in A['metrics'].items():
        c = m['reml'] if m['used'] == 'reml' else m['moment']; t = ARCH[a] + MET[y]
        mac(f'sScene{t}', c['s2_scene'], '{:.3g}'); mac(f'sTrain{t}', c['s2_train'], '{:.3g}'); mac(f'sInter{t}', c['s2_inter'], '{:.3g}'); mac(f'sSeed{t}', c['s2_seed'], '{:.3g}')
        mac(f'share{t}', m['share_test_side_of_predicted']); mac(f'icc{t}', m['icc_scene_single_seed'])
        mac(f'vTest{t}', m['V_test_closed_form'], '{:.3g}'); mac(f'sSeedRun{t}', m['s2_seed_run_level'], '{:.3g}')
        for K, v in m['MDE_absolute'].items(): mac(f'mde{t}{K.replace("_", "")}', v, '{:.3g}')
        mac(f'hsixSim{t}', m['H6_sim_variance_reduction'])
        if 'share_test_side_vs_josaa_observed' in m:
            mac(f'shareJosaa{t}', m['share_test_side_vs_josaa_observed']); mac(f'vPredObs{t}', m['V_pred_over_josaa_observed'])
    if 'H4_gauge' in A:
        h = A['H4_gauge']; t = ARCH[a]
        mac(f'hfourRed{t}', h['interaction_reduction']); mac(f'hfourIccPsnr{t}', h['icc_psnr']); mac(f'hfourIccGain{t}', h['icc_scenegain_psnr'])
        mac(f'hfourInterPsnr{t}', h['s2_inter_psnr'], '{:.3g}'); mac(f'hfourInterGain{t}', h['s2_inter_scenegain_psnr'], '{:.3g}')
    for y, o in A.get('H6_observed', {}).items():
        t = ARCH[a] + MET[y]; mac(f'hsixObsRed{t}', o['reduction']); mac(f'hsixObsRand{t}', o['var_fold_means_random'], '{:.3g}'); mac(f'hsixObsStrat{t}', o['var_fold_means_stratified'], '{:.3g}')
# ranges across archs for the primary metrics

for y in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse', 'R_siPSNR_scenegain_rgb']:
    sh = [m2['archs'][a]['metrics'][y]['share_test_side_of_predicted'] for a in m2['archs']]; ic = [m2['archs'][a]['metrics'][y]['icc_scene_single_seed'] for a in m2['archs']]
    mac(f'shareMin{MET[y]}', min(sh)); mac(f'shareMax{MET[y]}', max(sh)); mac(f'iccMin{MET[y]}', min(ic)); mac(f'iccMax{MET[y]}', max(ic))
red = [m2['archs'][a]['H4_gauge']['interaction_reduction'] for a in m2['archs'] if 'H4_gauge' in m2['archs'][a]]
mac('hfourRedMin', min(red)); mac('hfourRedMax', max(red))

# --- pairs (main): interaction share and gap/MDE
pairs = m2['pairs']
def pair_tag(p): return ARCH[p['b']] + ARCH[p['a']] + MET[p['metric']]
for p in pairs:
    t = pair_tag(p); mac(f'gap{t}', p['mean_gap'], '{:+.3g}'); mac(f'sxm{t}', p['s2_scene_x_method'], '{:.3g}'); mac(f'ishare{t}', p['share_interaction_K1S3'])
    mac(f'gapMdeKone{t}', p['gap_over_MDE_K1']); mac(f'gapMdeFull{t}', p['gap_over_MDE_fullcov'])
    mac(f'mdePairKone{t}', p['MDE_paired']['paired_K1_S3'], '{:.3g}'); mac(f'mdePairFull{t}', p['MDE_paired']['paired_fullcov_5fold_S3'], '{:.3g}')
direct_pairs = [p for p in pairs if 'direct' in (p['a'], p['b'])]; other_pairs = [p for p in pairs if 'direct' not in (p['a'], p['b'])]
mac('ishareDirectMin', min(p['share_interaction_K1S3'] for p in direct_pairs)); mac('ishareDirectMax', max(p['share_interaction_K1S3'] for p in direct_pairs))
mac('ishareOtherMin', min(p['share_interaction_K1S3'] for p in other_pairs)); mac('ishareOtherMax', max(p['share_interaction_K1S3'] for p in other_pairs))
mac('nPairMetricCells', len(pairs), '{:d}'); mac('nResolvableKone', sum(p['gap_over_MDE_K1'] >= 1 for p in pairs), '{:d}')
mac('nResolvableFull', sum(p['gap_over_MDE_fullcov'] >= 1 for p in pairs), '{:d}'); mac('maxGapMdeKone', max(p['gap_over_MDE_K1'] for p in pairs))
# H3 validation
v = m2['H3_josaa_validation']; mac('hthreeN', v['n'], '{:d}'); mac('hthreeR', v['pearson_r'], '{:.3f}'); mac('hthreeSign', v['sign_agreement'], '{:d}'); mac('hthreeRmse', v['rmse_dB'])
mac('hthreeRPooled', v['pearson_r_pooled'], '{:.3f}'); mac('hthreeSignPooled', v['sign_agreement_pooled'], '{:d}'); mac('hthreeRmsePooled', v['rmse_pooled_dB']); mac('hthreePredCorr', v['pearson_r_equal_vs_pooled_prediction'], '{:.3f}')
rows = v['rows']; import numpy as np
sub = [r for r in rows if 'direct' in r['pair']]; pr = np.array([r['predicted_gap_dB'] for r in sub]); ob = np.array([r['observed_gap_dB'] for r in sub])
mac('hthreeDirectN', len(sub), '{:d}'); mac('hthreeDirectR', float(np.corrcoef(pr, ob)[0, 1]), '{:.3f}'); mac('hthreeDirectSign', int((np.sign(pr) == np.sign(ob)).sum()), '{:d}')
sub = [r for r in rows if 'direct' not in r['pair']]; pr = np.array([r['predicted_gap_dB'] for r in sub]); ob = np.array([r['observed_gap_dB'] for r in sub])
mac('hthreeOtherN', len(sub), '{:d}'); mac('hthreeOtherR', float(np.corrcoef(pr, ob)[0, 1]), '{:.3f}'); mac('hthreeOtherSign', int((np.sign(pr) == np.sign(ob)).sum()), '{:d}'); mac('hthreeOtherMeanAbs', float(np.abs(ob).mean()))
# killing-experiment validation (direct/unet pair, 5 partitions)
kv = k1['josaa_partition_gap_validation']; mac('killValR', kv['pearson_r'], '{:.3f}'); mac('killValSign', kv['sign_agreement'], '{:d}'); mac('killValN', len(kv['rows']), '{:d}')
kp = next(p for p in k1['pairs'] if p['metric'] == 'R_PSNR_of_mean_mse')
from scipy.stats import spearmanr
ktab = {r['scene']: r['psnr'] for r in k1['archs']['direct']['metrics']['R_PSNR_of_mean_mse']['scene_table']}
ks = sorted(ktab); rho_k, p_k = spearmanr([ktab[s] for s in ks], [kp['per_scene_gap'][s] for s in ks])
mac('killSpearmanDiffGap', float(rho_k), '{:.2f}'); mac('killSpearmanDiffGapP', pval(float(p_k)))
mac('killGapSd', float(np.std([kp['per_scene_gap'][s] for s in ks], ddof=1)), '{:.2f}'); mac('killGapMean', kp['mean_gap'], '{:+.2f}')
mac('killScenesFavorUnet', int(sum(kp['per_scene_gap'][s] > 0 for s in ks)), '{:d}'); mac('killScenesFavorDirect', int(sum(kp['per_scene_gap'][s] < 0 for s in ks)), '{:d}')
# non-determinism control
nd = m2['nondeterminism_control']; mac('ndN', nd['n_common_scene_rows'], '{:d}')
for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean']:
    mac(f'ndSd{MET[y]}', nd[y]['sd_diff_scene_level'], '{:.3g}'); mac(f'ndVar{MET[y]}', nd[y]['nondet_var_single_run'], '{:.3g}')
    for a in ['direct', 'unet']:   # per-architecture (round 2, B3): own rerun variance, run-bootstrap interval, mean shift
        c = m2['archs'][a]['metrics'][y]; cc = c['reml'] if c['used'] == 'reml' else c['moment']; v = nd['by_arch'][a][y]
        mac(f'ndRatio{ARCH[a]}{MET[y]}', v['nondet_var_single_run'] / cc['s2_seed']); mac(f'ndVar{ARCH[a]}{MET[y]}', v['nondet_var_single_run'], '{:.3g}')
        mac(f'ndRatioLo{ARCH[a]}{MET[y]}', v['nondet_var_ci95_runboot'][0] / cc['s2_seed']); mac(f'ndRatioHi{ARCH[a]}{MET[y]}', v['nondet_var_ci95_runboot'][1] / cc['s2_seed'])
        mac(f'ndSd{ARCH[a]}{MET[y]}', v['sd_diff_scene_level'], '{:.3g}'); mac(f'ndMeanDiff{ARCH[a]}{MET[y]}', v['mean_diff'], '{:+.3g}')
    mac('ndRuns', nd['by_arch']['direct']['R_PSNR_of_mean_mse']['n_runs'], '{:d}')
    for a in ['direct', 'unet']:
        for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean']:
            v = nd['by_arch'][a][y]
            if 'run_level_share' in v: mac(f'ndRunShare{ARCH[a]}{MET[y]}', v['run_level_share']); mac(f'ndPairShiftSd{ARCH[a]}{MET[y]}', v['mean_pair_shift_sd'], '{:.3g}')
            if 'run_level_share_ci95_runboot' in v: mac(f'ndRunShareLo{ARCH[a]}{MET[y]}', v['run_level_share_ci95_runboot'][0]); mac(f'ndRunShareHi{ARCH[a]}{MET[y]}', v['run_level_share_ci95_runboot'][1])
# H5
for a in m2['archs']:
    for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean']:
        h = m2['H5_seed_variance'][a][y]; mac(f'hfiveSeedScene{ARCH[a]}{MET[y]}', h['s2_seed_scene_level'], '{:.3g}'); mac(f'hfiveSeedRun{ARCH[a]}{MET[y]}', h['s2_seed_run_level'], '{:.3g}')

# --- killing experiment headline (PSNR primary) + parity
for a in ['direct', 'unet']:
    m = k1['archs'][a]['metrics']['R_PSNR_of_mean_mse']; t = ARCH[a]
    mac(f'killShare{t}', m['share_test_side_closed']); mac(f'killShareLo{t}', m['bootstrap']['share_ci95'][0]); mac(f'killShareHi{t}', m['bootstrap']['share_ci95'][1])
    mac(f'killVpredObs{t}', m['V_pred_over_observed']); mac(f'killIcc{t}', m['icc_scene_single_seed']); mac(f'killInter{t}', (m['reml'] if m['used'] == 'reml' else m['moment'])['s2_inter']); mac(f'killTrain{t}', (m['reml'] if m['used'] == 'reml' else m['moment'])['s2_train']); mac(f'killImpliedSg{t}', m['H3_implied_s2_gamma_pure_test_side']); mac(f'killEstSg{t}', m['H3_estimated_s2_gamma']); mac(f'killSdScene{t}', m['H3_estimated_between_scene_sd_dB'])
for r in k1['parity']: mac(f'parityDiff{ARCH[r["arch"]]}', r['diff_dB'], '{:+.2f}')

# --- flow arm
for a, A in fl['archs'].items():
    t = ARCH[a]; mac(f'flowMinutes{t}', A['mean_minutes'], '{:.1f}')
    for y, m in A['metrics'].items():
        c = m['reml'] if m['used'] == 'reml' else m['moment']; tt = t + MET[y]
        mac(f'sScene{tt}', c['s2_scene'], '{:.3g}'); mac(f'sTrain{tt}', c['s2_train'], '{:.3g}'); mac(f'sInter{tt}', c['s2_inter'], '{:.3g}'); mac(f'sSeed{tt}', c['s2_seed'], '{:.3g}')
        mac(f'share{tt}', m['share_test_side_of_predicted']); mac(f'icc{tt}', m['icc_scene_single_seed'])
        for K, v2 in m['MDE_absolute'].items(): mac(f'mde{tt}{K.replace("_", "")}', v2, '{:.3g}')
    lad = A['metrics']['epe_mean']['MDE_absolute']; mac(f'mdeRatioKoneFull{t}', lad['K1_S3'] / lad['fullcov_5fold_S3'], '{:.1f}')
    h = A.get('FH4_zeroshot_vs_finetuned_scene_difficulty', {}); mac(f'fhfourRho{t}', h.get('spearman', float('nan')), '{:.3f}')
    mac(f'flowZeroShotEpe{t}', fl['zeroshot'][a]['epe_pair_mean'], '{:.3f}'); mac(f'flowZsSceneMean{t}', h.get('epe_zeroshot_mean', float('nan')), '{:.2f}'); mac(f'flowFtSceneMean{t}', h.get('epe_finetuned_scene_mean', float('nan')), '{:.2f}')
    tab = {r['scene']: r['score'] for r in A['metrics']['epe_mean']['scene_table']}; mac(f'flowSceneMin{t}', min(tab.values())); mac(f'flowSceneMax{t}', max(tab.values()), '{:.1f}')
    hard = sorted(tab, key=tab.get, reverse=True)[:4]
    for i, sc in enumerate(hard): mac(f'flowHard{t}{["One", "Two", "Three", "Four"][i]}', sc.replace('_', r'\_'))
    mac(f'flowSceneMinName{t}', min(tab, key=tab.get).replace('_', r'\_')); mac(f'flowSceneMaxName{t}', max(tab, key=tab.get).replace('_', r'\_'))
for p in fl['pairs']:
    tt = 'RaftSRaftL' + MET[p['metric']]; mac(f'gap{tt}', p['mean_gap'], '{:+.3g}'); mac(f'ishare{tt}', p['share_interaction_K1S3']); mac(f'gapMdeKone{tt}', p['gap_over_MDE_K1']); mac(f'gapMdeFull{tt}', p['gap_over_MDE_fullcov'])
    mac(f'mdePairKone{tt}', p['MDE_paired']['paired_K1_S3'], '{:.3g}'); mac(f'mdePairFull{tt}', p['MDE_paired']['paired_fullcov_5fold_S3'], '{:.3g}')
xt = fl['FH3_cross_task']
for k, v2 in xt.items():
    arch, rest = k.split('|'); lvl = 'Scene' if 'scene_level' in rest else 'Fold'; met = 'Simse' if 'simse' in rest else ('Psnr' if 'PSNR' in rest else 'Lmse')
    mac(f'xt{lvl}{ARCH[arch]}{met}Spearman', v2['spearman'], '{:.2f}')
    if 'pearson' in v2: mac(f'xt{lvl}{ARCH[arch]}{met}Pearson', v2['pearson'], '{:.2f}'); mac(f'xt{lvl}{ARCH[arch]}{met}P', v2['p'], '{:.2f}')

# flow: method gap vs difficulty (Fig 5b)
fpair = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'); ftab = {r['scene']: r['score'] for r in fl['archs']['raft_large']['metrics']['epe_mean']['scene_table']}
fs = sorted(ftab); rho_f, p_f = spearmanr([ftab[s] for s in fs], [fpair['per_scene_gap'][s] for s in fs])
mac('flowSpearmanDiffGap', float(rho_f), '{:.2f}'); mac('flowSpearmanDiffGapP', pval(float(p_f)))
mac('flowGapMin', min(fpair['per_scene_gap'].values()), '{:.2f}'); mac('flowGapMax', max(fpair['per_scene_gap'].values()), '{:.2f}')
# --- REML vs method-of-moments agreement (all arch x metric fits, intrinsic + flow) and the single-path REML audit
fits = [(a, y, m) for a in k1['archs'] for y, m in k1['archs'][a]['metrics'].items()] + [(a, y, m) for a in m2['archs'] for y, m in m2['archs'][a]['metrics'].items()] + [(a, y, m) for a in fl['archs'] for y, m in fl['archs'][a]['metrics'].items()]
rel = {'s2_scene': [], 's2_inter': [], 's2_seed': []}; train_zero = 0; train_share = []
for a, y, m in fits:
    r, mo = m['reml'], m['moment']
    if not r or 'error' in r: continue
    for k in rel:
        if r[k] > 0: rel[k].append(abs(mo[k] - r[k]) / r[k])
    train_zero += int(mo['s2_train'] == 0.0)
    tot = r['s2_scene'] + r['s2_train'] + r['s2_inter'] + r['s2_seed']; train_share.append(r['s2_train'] / tot if tot > 0 else 0.0)
mac('remlMomNfits', len(fits), '{:d}'); mac('remlMomSceneMaxRelPct', 100 * max(rel['s2_scene']), '{:.0f}'); mac('remlMomInterMaxRelPct', 100 * max(rel['s2_inter']), '{:.0f}')
mac('remlMomSeedMaxRelPct', 100 * max(rel['s2_seed']), '{:.0f}'); mac('remlMomTrainZero', train_zero, '{:d}'); mac('remlMomTrainMaxSharePct', 100 * max(train_share), '{:.0f}')
incons = [(a, y) for a, y, m in fits if str(m['used']).startswith('moment (reml inconsistent')] + [(p['pair'], p['metric']) for p in k1['pairs'] + m2['pairs'] + fl['pairs'] if str(p['used']).startswith('moment (reml inconsistent')]
mac('remlInconsistentFits', len(incons), '{:d}'); mac('remlNfitsAll', len(fits) + len(k1['pairs']) + len(m2['pairs']) + len(fl['pairs']), '{:d}')
# audit of the earlier single-path REML (results archived as *_pre20261104.json): fits whose implied scene-mean variance was outside the band
pre = [R / 'killing_v1_pre20261104.json', R / 'main_v2_pre20261104.json', R / 'flow_v1_pre20261104.json']
if all(p.exists() for p in pre):
    spur = []; worst = 1.0
    for p in pre:
        d = json.load(open(p))
        for a in d['archs']:
            for y, m in d['archs'][a]['metrics'].items():
                r, mo = m['reml'], m['moment']
                if not r or 'error' in r or not mo.get('s2_scene_raw'): continue
                ratio = (r['s2_scene'] + (r['s2_train'] + r['s2_inter'] + r['s2_seed'] / mo['S']) / mo['R']) / mo['s2_scene_raw']
                if not (0.8 <= ratio <= 1.25): spur.append((a, y, ratio)); worst = min(worst, ratio)
    mac('remlSinglePathSpurious', len(spur), '{:d}'); mac('remlSinglePathWorstRatio', worst); mac('remlSinglePathWorstFactor', 1 / worst, '{:.0f}')
# exact Henderson-III coefficient of s2_train in the unit-mean variance row (Q1) for the main-experiment design vs the naive 1/R (round 2, B2)
_mo = m2['archs']['direct']['metrics']['R_PSNR_of_mean_mse']['moment']
mac('momCoefTrainQone', _mo['coef_Q1'][1], '{:.3f}'); mac('momCoefTrainNaive', 1 / _mo['R'], '{:.3f}'); mac('momCoefTrainEqualFold', (23 - _mo['nf']) / 22 / _mo['R'], '{:.3f}')
mac('momDofResid', _mo['dof_resid'], '{:d}'); mac('momNcells', _mo['n_cells'], '{:d}')
mac('vZetaDirectPsnr', _mo['v_zeta'], '{:.3f}'); mac('vEpsDirectPsnr', _mo['v_eps'], '{:.3f}')
if 'coef_Q3' in _mo: mac('momCoefTrainQthree', _mo['coef_Q3'][1], '{:.3f}'); mac('momCoefCellQthree', _mo['coef_Q3'][2], '{:.3f}')
# run-level seed shift absorbed by the cell-level train-side component (round 3, B2): Var(zeta_bar_T) ~ (tau2 - s2_seed/n)/S
_vz = _mo['v_zeta'] / _mo['S']   # the correction applied to the cell-level train-side solution (round 4, A1)
mac('zetaShareDirectPsnr', _vz, '{:.3f}'); mac('zetaPctDirectPsnr', 100 * _vz / _mo['s2_train_cell'] if _mo['s2_train_cell'] > 0 else float('nan'), '{:.0f}')
# conditional (unit-centred Q3) vs unconditional third moment equation: real-data values and fixed-difficulty simulation (compare_q3_forms.py)
QC = R / 'estimator_q3_comparison.json'
if QC.exists():
    qc = json.load(open(QC)); st = next(x for x in qc['settings'] if x['arch'] == 'direct' and x['metric'] == 'R_PSNR_of_mean_mse')
    mac('momTrainUncondDirectPsnr', st['real_data']['unconditional']['s2_train'], '{:.3f}'); mac('momTrainCondDirectPsnr', st['real_data']['conditional']['s2_train'], '{:.3f}')
    mac('remlTrainDirectPsnr', st['real_data']['reml']['s2_train'], '{:.3f}'); mac('valTruthTrainDirectPsnr', st['truth']['s2_train'], '{:.3f}')
    mac('valUncondTrainDirectPsnr', st['sim_unconditional']['s2_train']['mean'], '{:.3f}'); mac('valCondTrainDirectPsnr', st['sim_conditional']['s2_train']['mean'], '{:.3f}')
    mac('valQthreeNdata', qc['n_data'], '{:d}')
# PSNR interaction relative to the scene component, main experiment (abstract / contribution iii wording)
_r = []
for a in m2['archs']:
    mm = m2['archs'][a]['metrics']['R_PSNR_of_mean_mse']; c = mm['reml'] if mm['used'] == 'reml' else mm['moment']; _r.append(c['s2_inter'] / c['s2_scene'])
mac('psnrInterOverSceneMin', min(_r), '{:.1f}'); mac('psnrInterOverSceneMax', max(_r), '{:.1f}'); mac('psnrInterGeSceneN', sum(x >= 1 for x in _r), '{:d}')
import design_mde as D
# headline counts under the REML components (round 3, B7): same plug-in rule, REML components in place of the moment ones
def _counts(pairs, use):
    k1 = full = n = 0
    for p in pairs:
        if p['metric'] not in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse', 'R_siPSNR_scenegain_rgb']: continue
        c = p[use]
        if not c or 'error' in c: continue
        q = dict(p); q['s2_scene_x_method'] = c['s2_scene']; q['s2_train_diff'] = c['s2_train']; q['s2_inter_diff'] = c['s2_inter']; g = abs(p['mean_gap']); n += 1
        k1 += g / D.mde_paired(q, 1, 3, False) >= 1; full += g / D.mde_paired(q, 5, 3, True) >= 1
    return n, k1, full
_n, _k1, _full = _counts(m2['pairs'], 'reml'); mac('nResolvableKoneReml', _k1, '{:d}'); mac('nResolvableFullReml', _full, '{:d}'); mac('nPairMetricCellsReml', _n, '{:d}')
_fp = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'); _q = dict(_fp); _c = _fp['reml']; _q['s2_scene_x_method'] = _c['s2_scene']; _q['s2_train_diff'] = _c['s2_train']; _q['s2_inter_diff'] = _c['s2_inter']
mac('gapMdeFullRaftSRaftLEpeReml', abs(_fp['mean_gap']) / D.mde_paired(_q, 5, 3, True)); mac('gapMdeKoneRaftSRaftLEpeReml', abs(_fp['mean_gap']) / D.mde_paired(_q, 1, 3, False))
# F-test for the observed stratified-vs-random fold-mean variance (round 3, C9): within-repeat variances, dof (reps x (folds-1))
from scipy.stats import f as _fdist
for a in ['direct', 'unet']:
    for y in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse']:
        o = m2['archs'][a].get('H6_observed', {}).get(y)
        if not o: continue
        F = o['var_fold_means_stratified'] / o['var_fold_means_random']; d1 = o['n_strat_reps'] * 4; d2 = o['n_rand_reps'] * 4
        cdf = _fdist.cdf(F, d1, d2); p2 = 2 * min(cdf, 1 - cdf)
        mac(f'hsixObsF{ARCH[a]}{MET[y]}', F); mac(f'hsixObsP{ARCH[a]}{MET[y]}', pval(p2)); mac(f'hsixObsDofStrat', d1, '{:d}'); mac(f'hsixObsDofRand', d2, '{:d}')
# --- review round 1 additions ---------------------------------------------------------------------------
# training-run count (excludes the two zero-shot evaluations)
mac('nRunsTrain', k1['n_cv_runs'] + 2 + m2['n_runs'] + fl['n_runs'], '{:d}')
# frame-weighted (ratio-estimator) vs unit-weighted V_test in the killing experiment (A3)
for a in ['direct', 'unet']:
    m = k1['archs'][a]['metrics']['R_PSNR_of_mean_mse']
    mac(f'killVtestFwRatio{ARCH[a]}', m['V_test_sim_frame_weighted'] / m['V_test_closed_form']); mac(f'killVtestEqRatio{ARCH[a]}', m['V_test_sim_scene_equal'] / m['V_test_closed_form'])
# design cost (C1): MDE under alternative designs from the shared helper
import sys; sys.path.insert(0, str(Path(__file__).parent)); mD = m2['archs']['direct']['metrics']; mS = fl['archs']['raft_small']['metrics']
pUD = next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'simse_R_gray_mean'); pFL = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean')
k13, f51, f52, f53, f103 = (1, 3, False), (5, 1, True), (5, 2, True), (5, 3, True), (10, 3, True)
mac('dcRatioFullOneSimse', D.mde_abs(mD['simse_R_gray_mean'], *k13) / D.mde_abs(mD['simse_R_gray_mean'], *f51), '{:.1f}')
mac('dcRatioFullOnePsnr', D.mde_abs(mD['R_PSNR_of_mean_mse'], *k13) / D.mde_abs(mD['R_PSNR_of_mean_mse'], *f51), '{:.1f}')
mac('dcRatioFullOneEpe', D.mde_abs(mS['epe_mean'], *k13) / D.mde_abs(mS['epe_mean'], *f51), '{:.1f}')
mac('dcRatioFullThreeEpe', D.mde_abs(mS['epe_mean'], *k13) / D.mde_abs(mS['epe_mean'], *f53), '{:.1f}')
mac('dcRatioRandFiveSimse', D.mde_abs(mD['simse_R_gray_mean'], *k13) / D.mde_abs(mD['simse_R_gray_mean'], 5, 3, False), '{:.1f}')
mac('dcGapMdeFullOneRaft', abs(pFL['mean_gap']) / D.mde_paired(pFL, *f51)); mac('dcGapMdeFullOneUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, *f51))
mac('dcGapMdeFullTwoRepUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, *f103)); mac('dcGapMdeFullThreeUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, *f53))
mac('dcGapMdeKoneSoneRaft', abs(pFL['mean_gap']) / D.mde_paired(pFL, 1, 1, False))
# per-channel gain on the stored bamboo_2 predictions (A2)
GC = R / 'qual' / 'gauge_channel_gain.json'
if GC.exists():
    gc = json.load(open(GC)); sm = gc['summary']
    mac('qChanGapRaw', sm['gap_raw_means'], '{:.1f}'); mac('qChanGapScalar', sm['gap_scalar_means'], '{:.1f}'); mac('qChanGapChannel', sm['gap_channel_means'], '{:.1f}')
    mac('qChanOutScalarMin', sm['out_scalar'][0], '{:.1f}'); mac('qChanOutScalarMax', sm['out_scalar'][1], '{:.1f}'); mac('qChanOutChannelMin', sm['out_channel'][0], '{:.1f}'); mac('qChanOutChannelMax', sm['out_channel'][1], '{:.1f}')
    mac('qChanInChannelMin', sm['in_channel'][0], '{:.1f}'); mac('qChanInChannelMax', sm['in_channel'][1], '{:.1f}')
    outc = [p for p in gc['per_composition'] if not p['bamboo_1_in_train']]
    for i, ch in enumerate(['R', 'G', 'B']):
        mac(f'qChanOutGain{ch}Min', min(p['gain_rgb'][i] for p in outc)); mac(f'qChanOutGain{ch}Max', max(p['gain_rgb'][i] for p in outc))
    mac('qChanRemovedScalarPct', 100 * (1 - sm['gap_scalar_means'] / sm['gap_raw_means']), '{:.0f}'); mac('qChanRemovedChannelPct', 100 * (1 - sm['gap_channel_means'] / sm['gap_raw_means']), '{:.0f}')
    mac('qChanNframes', len(gc['frames']), '{:d}')
# parametric-bootstrap intervals (B1)
BSF = R / 'bootstrap_ci.json'
if BSF.exists():
    bs = json.load(open(BSF)); mac('bootN', bs['n_boot'], '{:d}')
    for exp, key in [('intrinsic', 'intrinsic'), ('flow', 'flow'), ('killing', 'killing')]:
        for a, AA in bs[exp].items():
            for y, r in AA.items():
                if y not in MET: continue
                t = ('kill' if exp == 'killing' else '') + ARCH[a] + MET[y]
                mac(f'shareLo{t}', r['share']['ci95'][0]); mac(f'shareHi{t}', r['share']['ci95'][1]); mac(f'iccLo{t}', r['icc']['ci95'][0]); mac(f'iccHi{t}', r['icc']['ci95'][1])
    # killing experiment: parametric interval of the share vs the earlier grid's observed partition variance (only V_test varies)
    for a in ['direct', 'unet']:
        obs = k1['archs'][a]['josaa_sigma2_part_obs']; cis = bs['killing'][a]['R_PSNR_of_mean_mse']['s2_scene']['ci95']; f = (1 - 5 / 23) / 5
        mac(f'killShareLoP{ARCH[a]}', f * cis[0] / obs); mac(f'killShareHiP{ARCH[a]}', f * cis[1] / obs)
    lo = [bs['intrinsic'][a]['simse_R_gray_mean']['share']['ci95'][0] for a in bs['intrinsic']]; hi = [bs['intrinsic'][a]['simse_R_gray_mean']['share']['ci95'][1] for a in bs['intrinsic']]
    mac('shareSimseCiMin', min(lo)); mac('shareSimseCiMax', max(hi))
    lo = [bs['intrinsic'][a]['R_PSNR_of_mean_mse']['share']['ci95'][0] for a in bs['intrinsic']]; hi = [bs['intrinsic'][a]['R_PSNR_of_mean_mse']['share']['ci95'][1] for a in bs['intrinsic']]
    mac('sharePsnrCiMin', min(lo)); mac('sharePsnrCiMax', max(hi))
# --- review round 2 additions ---------------------------------------------------------------------------
from scipy.stats import chi2, spearmanr as _sp
fc_i = [m['fullcov_check']['ratio_obs_pred'] for A in m2['archs'].values() for m in A['metrics'].values()]
fc_f = [m['fullcov_check']['ratio_obs_pred'] for A in fl['archs'].values() for m in A['metrics'].values()]
mac('fcRatioMinIntr', min(fc_i)); mac('fcRatioMaxIntr', max(fc_i)); mac('fcRatioMinFlow', min(fc_f)); mac('fcRatioMaxFlow', max(fc_f))
mac('fcChiLo', chi2.ppf(0.025, 4) / 4); mac('fcChiHi', chi2.ppf(0.975, 4) / 4); mac('fcNfits', len(fc_i) + len(fc_f), '{:d}')
mac('fcWithin', sum(chi2.ppf(0.025, 4) / 4 <= r <= chi2.ppf(0.975, 4) / 4 for r in fc_i + fc_f), '{:d}')
# sensitivity of the full-coverage resolution to within-assignment correlations (round 2, C2): factor on the MDE at rho = 0.25 and 0.5
for tag, m in [('DirectPsnr', mD['R_PSNR_of_mean_mse']), ('DirectSimse', mD['simse_R_gray_mean']), ('RaftSEpe', fl['archs']['raft_small']['metrics']['epe_mean'])]:
    c = D.picked(m)
    for rho, rt in [(0.25, 'Quarter'), (0.5, 'Half')]:
        mac(f'sensFull{tag}{rt}', D.fullcov_mde_factor(c['s2_train'], c['s2_inter'], m['v_zeta'], m['v_eps'], 3, rho), '{:.2f}')
for tag, p in [('UnetDirectSimse', pUD), ('RaftSRaftLEpe', pFL)]:
    for rho, rt in [(0.25, 'Quarter'), (0.5, 'Half')]:
        f = D.fullcov_mde_factor(p['s2_train_diff'], p['s2_inter_diff'], p['v_zeta_diff'], p['v_eps_diff'], 3, rho)
        mac(f'sensFull{tag}{rt}', f, '{:.2f}'); mac(f'sensGapMdeFull{tag}{rt}', abs(p['mean_gap']) / (D.mde_paired(p, 5, 3, True) * f))
# integer design search (round 4, C1): cheapest full-coverage design (R repeats x S seeds, 5RS runs per method) whose plug-in gap/MDE
# reaches 1 (and 1.25 as a margin) for the headline intrinsic comparison; the prevailing recipe costs 3 runs per method
for tgt, tt in [(1.0, 'One'), (1.25, 'OneQuarter')]:
    b = D.cheapest_design(pUD, tgt)
    if b: mac(f'dsMinRuns{tt}UnetDirectSimse', b[0], '{:d}'); mac(f'dsMinR{tt}UnetDirectSimse', b[1], '{:d}'); mac(f'dsMinS{tt}UnetDirectSimse', b[2], '{:d}'); mac(f'dsMinRatio{tt}UnetDirectSimse', b[3])
mac('dcGapMdeFullTwoSeedsUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, 5, 2, True)); mac('dcGapMdeFullTwoSeedsRaft', abs(pFL['mean_gap']) / D.mde_paired(pFL, 5, 2, True))
mac('dcThisStudyRuns', 75, '{:d}'); mac('dcPrevailingRuns', 3, '{:d}'); mac('dcThisStudyOverPrevailing', 75 / 3, '{:.0f}')
mac('dcGapMdeFullFiveRepUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, 25, 3, True)); mac('dcGapMdeFullFourRepThreeUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, 20, 3, True))
mac('dcGapMdeFullSixRepOneUnetDirectSimse', abs(pUD['mean_gap']) / D.mde_paired(pUD, 30, 1, True))
# training-set size diagnostic (round 4, C4): trainsize_check.json
TS = R / 'trainsize_check.json'
if TS.exists():
    ts = json.load(open(TS)); _ps = [ts['intrinsic'][a]['R_PSNR_of_mean_mse'] for a in ts['intrinsic']]
    mac('tsPsnrDiffMin', min(x['mean_diff_19_minus_18'] for x in _ps), '{:+.2f}'); mac('tsPsnrDiffMax', max(x['mean_diff_19_minus_18'] for x in _ps), '{:+.2f}'); mac('tsPsnrSeMax', max(x['se'] for x in _ps))
    _oth = [ts['intrinsic'][a][y]['diff_over_between_scene_sd'] for a in ts['intrinsic'] for y in ts['intrinsic'][a] if y != 'R_PSNR_of_mean_mse'] + [ts['flow'][a]['epe_mean']['diff_over_between_scene_sd'] for a in ts['flow']]
    mac('tsOtherOverSdMaxAbs', max(abs(x) for x in _oth)); mac('tsPsnrOverSdMax', max(x['diff_over_between_scene_sd'] for x in _ps)); mac('tsNscenesBoth', _ps[0]['n_scenes_both'], '{:d}')
# gap vs difficulty with an independent reference (round 2, C4); coupled versions kept for the explanation
_tab = {a: {r['scene']: r['score'] for r in m2['archs'][a]['metrics']['R_PSNR_of_mean_mse']['scene_table']} for a in m2['archs']}
_pud = next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'R_PSNR_of_mean_mse'); _sc = sorted(_pud['per_scene_gap']); _g = [_pud['per_scene_gap'][x] for x in _sc]
for ref, tag in [('proposed', 'Prop'), ('transformer', 'Trans'), ('direct', 'Direct'), ('unet', 'Unet')]:
    rho, pv = _sp([_tab[ref][x] for x in _sc], _g); mac(f'gapRho{tag}', float(rho), '{:+.2f}'); mac(f'gapRhoP{tag}', pval(float(pv)))
_fp = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean'); _fsc = sorted(_fp['per_scene_gap']); _fg = [_fp['per_scene_gap'][x] for x in _fsc]
for ref, tag in [('raft_large', 'RaftL'), ('raft_small', 'RaftS')]:
    rho, pv = _sp([fl['zeroshot'][ref]['per_scene_epe'][x] for x in _fsc], _fg); mac(f'flowGapRhoZs{tag}', float(rho), '{:.2f}'); mac(f'flowGapRhoZsP{tag}', pval(float(pv)))
mac('gapSdMain', float(np.std(_g, ddof=1))); mac('gapMeanMain', float(np.mean(_g)), '{:+.2f}')
mac('gapScenesFavorUnetMain', int(sum(x > 0 for x in _g)), '{:d}'); mac('gapScenesFavorDirectMain', int(sum(x < 0 for x in _g)), '{:d}')
# REML cross-check: fits whose consistency ratio lies outside [0.80, 1.25] under the multi-start fit
_ratios = [m['reml_consistency_ratio'] for a, y, m in fits if m.get('reml_consistency_ratio') is not None]
mac('remlCrossOutside', sum(not (0.8 <= r <= 1.25) for r in _ratios if np.isfinite(r)), '{:d}'); mac('remlCrossMin', min(_ratios)); mac('remlCrossMax', max(_ratios))
# bootstrap intervals on MDEs and paired gap/MDE (round 2, C1)
if BSF.exists():
    bs = json.load(open(BSF))
    for exp, a, y, t in [('intrinsic', 'direct', 'R_PSNR_of_mean_mse', 'DirectPsnr'), ('intrinsic', 'direct', 'simse_R_gray_mean', 'DirectSimse'), ('flow', 'raft_small', 'epe_mean', 'RaftSEpe'), ('flow', 'raft_large', 'epe_mean', 'RaftLEpe')]:
        r = bs[exp][a][y]
        for k, nm in [('mde_K1_S3', 'KoneSthree'), ('mde_full_S3', 'fullcovfivefoldSthree')]:
            mac(f'mde{t}{nm}Lo', r[k]['ci95'][0], '{:.3g}'); mac(f'mde{t}{nm}Hi', r[k]['ci95'][1], '{:.3g}')
    for key, t in [('unet - direct|simse_R_gray_mean', 'UnetDirectSimse'), ('unet - direct|lmse_R_chen_mean', 'UnetDirectLmse'), ('unet - direct|R_PSNR_of_mean_mse', 'UnetDirectPsnr'), ('raft_small - raft_large|epe_mean', 'RaftSRaftLEpe')]:
        if key in bs.get('pairs', {}):
            r = bs['pairs'][key]
            mac(f'gapMdeKone{t}Lo', r['gap_over_mde_K1']['ci95'][0]); mac(f'gapMdeKone{t}Hi', r['gap_over_mde_K1']['ci95'][1]); mac(f'gapMdeFull{t}Lo', r['gap_over_mde_full']['ci95'][0]); mac(f'gapMdeFull{t}Hi', r['gap_over_mde_full']['ci95'][1])
            if 'gap' in r: mac(f'gapCiLo{t}', r['gap']['ci95'][0], '{:+.3g}'); mac(f'gapCiHi{t}', r['gap']['ci95'][1], '{:+.3g}')
    # how many of the 30 intrinsic pair-metric cells have an interval for gap/MDE_full that excludes 1 (above or below)
    cells = [r for k, r in bs['pairs'].items() if not k.startswith('raft')]
    mac('nPairCellsBoot', len(cells), '{:d}'); mac('nPairCellsFullAboveOne', sum(r['gap_over_mde_full']['ci95'][0] >= 1 for r in cells), '{:d}'); mac('nPairCellsFullBelowOne', sum(r['gap_over_mde_full']['ci95'][1] < 1 for r in cells), '{:d}')
# estimator validation study (round 2, B1/B2)
VF = R / 'estimator_validation.json'
if VF.exists():
    val = json.load(open(VF)); mac('valNdata', val['n_data'], '{:d}'); mac('valNboot', val['n_boot'], '{:d}')
    cov_s = [st['coverage95']['share'] for st in val['settings'] if st.get('coverage95')]; cov_i = [st['coverage95']['icc'] for st in val['settings'] if st.get('coverage95')]
    if cov_s: mac('valCovShareMin', min(cov_s)); mac('valCovShareMax', max(cov_s)); mac('valCovIccMin', min(cov_i)); mac('valCovIccMax', max(cov_i))
    bias_s = [st['moment']['share']['mean'] - st['truth']['share'] for st in val['settings']]; bias_i = [st['moment']['icc']['mean'] - st['truth']['icc'] for st in val['settings']]
    mac('valBiasShareMaxAbs', max(abs(b) for b in bias_s), '{:.3f}'); mac('valBiasIccMaxAbs', max(abs(b) for b in bias_i), '{:.3f}'); mac('valNsettings', len(val['settings']), '{:d}')
# display-run typicality (round 3, C5): qual_display_check.py -> qual/display_check.json
DC = R / 'qual' / 'display_check.json'
if DC.exists():
    dc = json.load(open(DC))
    for s, g in dc['intrinsic']['scenes'].items():
        t = s.replace('_', '').replace('5', 'five').replace('2', 'two')
        mac(f'qDispGap{t}', g['gap_unet_minus_direct_display'], '{:+.1f}'); mac(f'qCellGap{t}', g['gap_unet_minus_direct_main_cell_mean'], '{:+.1f}'); mac(f'qAllGap{t}', g['gap_unet_minus_direct_main_all_cells'], '{:+.1f}')
        for a in ['direct', 'unet']:
            mac(f'qSeedMin{ARCH[a]}{t}', g[a]['main_min'], '{:.1f}'); mac(f'qSeedMax{ARCH[a]}{t}', g[a]['main_max'], '{:.1f}'); mac(f'qSeedMean{ARCH[a]}{t}', g[a]['main_mean'], '{:.1f}')
    mac('qDispOutsideUnet', sum(not g['unet']['display_within_seed_range'] for g in dc['intrinsic']['scenes'].values()), '{:d}')
    if 'gauge' in dc:
        gi = [c for c in dc['gauge']['compositions'] if c['bamboo_1_in_train']]; go = [c for c in dc['gauge']['compositions'] if not c['bamboo_1_in_train']]
        mac('qGaugeMainInPsnrMin', min(min(c['main_psnr']) for c in gi), '{:.1f}'); mac('qGaugeMainInPsnrMax', max(max(c['main_psnr']) for c in gi), '{:.1f}')
        mac('qGaugeMainOutPsnrMin', min(min(c['main_psnr']) for c in go), '{:.1f}'); mac('qGaugeMainOutPsnrMax', max(max(c['main_psnr']) for c in go), '{:.1f}')
        mac('qGaugeMainInGainMin', min(min(c['main_gain']) for c in gi)); mac('qGaugeMainInGainMax', max(max(c['main_gain']) for c in gi))
        mac('qGaugeMainOutGainMin', min(min(c['main_gain']) for c in go)); mac('qGaugeMainOutGainMax', max(max(c['main_gain']) for c in go))
        mac('qGaugeMainNseeds', len(dc['gauge']['compositions'][0]['main_seeds']), '{:d}')
    mac('qDispOutsideDirect', sum(not g['direct']['display_within_seed_range'] for g in dc['intrinsic']['scenes'].values()), '{:d}')
    for s, g in dc['flow']['scenes'].items():
        t = s.replace('_', '').replace('4', 'four').replace('2', 'two')
        for a in ['raft_small', 'raft_large']:
            mac(f'qFlowSeedMin{ARCH[a]}{t}', g[a]['main_min'], '{:.2f}'); mac(f'qFlowSeedMax{ARCH[a]}{t}', g[a]['main_max'], '{:.2f}')
# misspecification study (round 3, B1): validate_misspec.py -> estimator_misspec.json
MS = R / 'estimator_misspec.json'
if MS.exists():
    ms = json.load(open(MS)); mac('msNdata', ms['n_data'], '{:d}'); mac('msNboot', ms['n_boot'], '{:d}'); mac('msNsettings', len(ms['settings']), '{:d}')
    SC = {'hetero': 'Hetero', 'corr': 'Corr', 'heavy': 'Heavy', 'assign': 'Assign'}
    for sc, tag in SC.items():
        bs_ = [st['scenarios'][sc]['bias']['share'] for st in ms['settings'] if sc in st['scenarios']]; bi_ = [st['scenarios'][sc]['bias']['icc'] for st in ms['settings'] if sc in st['scenarios']]
        cs_ = [st['scenarios'][sc]['coverage95']['share'] for st in ms['settings'] if sc in st['scenarios']]; ci_ = [st['scenarios'][sc]['coverage95']['icc'] for st in ms['settings'] if sc in st['scenarios']]
        if not bs_: continue
        mac(f'msBiasShareMaxAbs{tag}', max(abs(b) for b in bs_), '{:.3f}'); mac(f'msBiasIccMaxAbs{tag}', max(abs(b) for b in bi_), '{:.3f}')
        mac(f'msCovShareMin{tag}', min(cs_)); mac(f'msCovShareMax{tag}', max(cs_)); mac(f'msCovIccMin{tag}', min(ci_)); mac(f'msCovIccMax{tag}', max(ci_))
        fr_ = [st['scenarios'][sc].get('fullcov_var_true_over_predicted') for st in ms['settings'] if sc in st['scenarios']]; fr_ = [x for x in fr_ if x is not None]
        if fr_: mac(f'msFcRatioMin{tag}', min(fr_)); mac(f'msFcRatioMax{tag}', max(fr_)); mac(f'msFcMdeFactorMin{tag}', math.sqrt(min(fr_))); mac(f'msFcMdeFactorMax{tag}', math.sqrt(max(fr_)))
        if sc == 'assign':
            st0 = ms['settings'][0]; mac('msFcMdeFactorAssignDirectPsnr', math.sqrt(st0['scenarios']['assign']['fullcov_var_true_over_predicted']))
            stf = next((x for x in ms['settings'] if x['experiment'] == 'flow'), None)
            if stf and 'assign' in stf['scenarios']: mac('msFcMdeFactorAssignRaftS', math.sqrt(stf['scenarios']['assign']['fullcov_var_true_over_predicted']))
        tr_ = [st['scenarios'][sc]['moment']['s2_train']['mean'] for st in ms['settings'] if sc in st['scenarios']]
        if sc in ('corr', 'assign'):
            st0 = ms['settings'][0]
            if sc in st0['scenarios']: mac(f'ms{tag}TrainEstDirectPsnr', st0['scenarios'][sc]['moment']['s2_train']['mean'], '{:.3f}'); mac(f'ms{tag}InterEstDirectPsnr', st0['scenarios'][sc]['moment']['s2_inter']['mean'], '{:.3f}')
    allc = [st['scenarios'][sc]['coverage95'][k] for st in ms['settings'] for sc in st['scenarios'] for k in ['share', 'icc']]
    if allc: mac('msCovAllMin', min(allc)); mac('msCovAllMax', max(allc))
    allb = [abs(st['scenarios'][sc]['bias'][k]) for st in ms['settings'] for sc in st['scenarios'] for k in ['share', 'icc']]
    if allb: mac('msBiasAllMax', max(allb), '{:.3f}')
    htr = [st.get('hetero_total_ratio') for st in ms['settings'] if st.get('hetero_total_ratio')]
    if htr: mac('msHeteroTotalRatioMin', min(htr), '{:.1f}'); mac('msHeteroTotalRatioMax', max(htr), '{:.1f}')
    # heavy tails: ICC coverage is the weak point; report its minimum across settings and the share's
    hv = [st['scenarios']['heavy'] for st in ms['settings'] if 'heavy' in st['scenarios']]
    if hv: mac('msHeavyIccCovMin', min(x['coverage95']['icc'] for x in hv)); mac('msHeavyShareCovMin', min(x['coverage95']['share'] for x in hv))
    # the DirectCNN PSNR share under the correlated-interaction scenario (the case the sensitivity analysis uses)
    st0 = ms['settings'][0]
    if 'corr' in st0['scenarios']: mac('msCorrShareBiasDirectPsnr', st0['scenarios']['corr']['bias']['share'], '{:+.3f}'); mac('msCorrCovShareDirectPsnr', st0['scenarios']['corr']['coverage95']['share'])
# calibrated diagnostics (round 4, B1): calibrate_diagnostics.py -> calibration_diagnostics.json
CAL = R / 'calibration_diagnostics.json'
if CAL.exists():
    cal = json.load(open(CAL)); fc_ = cal['fullcov']; mac('calNsim', cal['n_sim'], '{:d}')
    mac('calFcWithin', sum(r['inside_empirical_band'] for r in fc_), '{:d}'); mac('calFcN', len(fc_), '{:d}'); mac('calFcPmin', min(r['empirical_p_two_sided'] for r in fc_), '{:.3f}')
    mac('calFcLoMin', min(r['empirical_ci95'][0] for r in fc_)); mac('calFcLoMax', max(r['empirical_ci95'][0] for r in fc_)); mac('calFcHiMin', min(r['empirical_ci95'][1] for r in fc_)); mac('calFcHiMax', max(r['empirical_ci95'][1] for r in fc_))
    outs = [r for r in fc_ if not r['inside_empirical_band']]
    if outs: mac('calFcOutsideName', {'raft_large': 'RAFT-large', 'raft_small': 'RAFT-small', 'direct': 'DirectCNN', 'unet': 'UNet', 'proposed': 'Proposed', 'transformer': 'TransUNet'}[outs[0]['arch']] + ' ' + {'epe_occ_mean': 'occluded-pixel EPE', 'epe_noc_mean': 'non-occluded EPE', 'epe_mean': 'EPE'}.get(outs[0]['metric'], outs[0]['metric']).replace('_', '\\_')); mac('calFcOutsideP', outs[0]['empirical_p_two_sided'], '{:.3f}'); mac('calFcOutsideRatio', outs[0]['observed_ratio'])
    for r in cal['stratification']:
        t = ARCH[r['arch']] + MET[r['metric']]
        mac(f'calStratMed{t}', r['sim_median']); mac(f'calStratP{t}', pval(r['empirical_p_two_sided'])); mac(f'calStratLo{t}', r['empirical_ci95'][0]); mac(f'calStratHi{t}', r['empirical_ci95'][1])
# operating characteristics of the MDE rule (round 4, B2): validate_power.py -> power_validation.json
PW = R / 'power_validation.json'
if PW.exists():
    pw = json.load(open(PW)); rs = pw['results']; mac('pwNsim', pw['n_sim'], '{:d}')
    mac('pwNullMin', min(r['null_rejection_model_se'] for r in rs), '{:.3f}'); mac('pwNullMax', max(r['null_rejection_model_se'] for r in rs), '{:.3f}')
    mac('pwPowerMin', min(r['power_at_mde_model_se'] for r in rs)); mac('pwPowerMax', max(r['power_at_mde_model_se'] for r in rs))
    fk = next(r for r in rs if r['experiment'] == 'flow' and r['design'] == 'K1_S3'); mac('pwNullFlowKone', fk['null_rejection_model_se'], '{:.3f}'); mac('pwPowerFlowKone', fk['power_at_mde_model_se'])
    emp = [r for r in rs if r['null_rejection_empirical_se'] is not None]
    mac('pwNullEmpMin', min(r['null_rejection_empirical_se'] for r in emp), '{:.3f}'); mac('pwNullEmpMax', max(r['null_rejection_empirical_se'] for r in emp), '{:.3f}')
    mac('pwPowerEmpMin', min(r['power_at_mde_empirical_se'] for r in emp)); mac('pwPowerEmpMax', max(r['power_at_mde_empirical_se'] for r in emp))
    e5 = [r for r in emp if r['design'] == 'K5_S3']; mac('pwPowerEmpKfiveMin', min(r['power_at_mde_empirical_se'] for r in e5)); mac('pwPowerEmpKfiveMax', max(r['power_at_mde_empirical_se'] for r in e5))
    e2 = [r for r in emp if r['design'] == 'full_R2_S3']; mac('pwPowerEmpRtwoMin', min(r['power_at_mde_empirical_se'] for r in e2)); mac('pwPowerEmpRtwoMax', max(r['power_at_mde_empirical_se'] for r in e2))
    mac('pwNdesigns', len({r['design'] for r in rs}), '{:d}'); mac('pwNpairs', len({(r['pair'], r['metric']) for r in rs}), '{:d}')
# finite-pilot operating characteristics (round 6, B2): validate_pilot_power.py -> pilot_power_validation.json
PP = R / 'pilot_power_validation.json'
if PP.exists():
    pp = json.load(open(PP)); rs = pp['results']; mac('ppNpilot', pp['n_pilot'], '{:d}'); mac('ppNeval', pp['n_eval'], '{:d}')
    mac('ppSeRatioMedMin', min(r['se_hat_over_true_median'] for r in rs)); mac('ppSeRatioMedMax', max(r['se_hat_over_true_median'] for r in rs))
    mac('ppSeRatioQfiveMin', min(r['se_hat_over_true_q05'] for r in rs)); mac('ppSeRatioQninefiveMax', max(r['se_hat_over_true_q95'] for r in rs))
    mac('ppNullMin', min(r['null_rejection_mean'] for r in rs), '{:.3f}'); mac('ppNullMax', max(r['null_rejection_mean'] for r in rs), '{:.3f}'); mac('ppNullQninefiveMax', max(r['null_rejection_q95'] for r in rs), '{:.3f}')
    mac('ppPowerMin', min(r['power_at_true_mde_mean'] for r in rs)); mac('ppPowerMax', max(r['power_at_true_mde_mean'] for r in rs))
    mac('ppPowerQfiveMin', min(r['power_at_true_mde_q05'] for r in rs)); mac('ppPowerQninefiveMax', max(r['power_at_true_mde_q95'] for r in rs))
    mac('ppNdesigns', len({r['design'] for r in rs}), '{:d}'); mac('ppNpairs', len({(r['pair'], r['metric']) for r in rs}), '{:d}')
    mac('ppMcSdNull', math.sqrt(0.05 * 0.95 / pp['n_eval']), '{:.3f}'); mac('ppMcSdPower', math.sqrt(0.8 * 0.2 / pp['n_eval']), '{:.3f}')
# equal-cost ordering under the assignment-level sensitivity (round 6, C2): 15 runs per method, full coverage x3 seeds vs five random partitions x3 seeds
for tag, m in [('DirectPsnr', mD['R_PSNR_of_mean_mse']), ('DirectSimse', mD['simse_R_gray_mean']), ('RaftSEpe', fl['archs']['raft_small']['metrics']['epe_mean'])]:
    full = D.mde_abs(m, 5, 3, True); rand5 = D.mde_abs(m, 5, 3, False); c = D.picked(m)
    f_cf = D.fullcov_mde_factor(c['s2_train'], c['s2_inter'], m['v_zeta'], m['v_eps'], 3, 0.5)
    mac(f'eqcFull{tag}', full, '{:.3g}'); mac(f'eqcRandFive{tag}', rand5, '{:.3g}'); mac(f'eqcFullInflCf{tag}', full * f_cf, '{:.3g}')
    mac(f'eqcFlipCf{tag}', 'yes' if full * f_cf > rand5 else 'no')
if MS.exists() and 'msFcMdeFactorAssignDirectPsnr' in M:
    mac('eqcFullInflSimDirectPsnr', D.mde_abs(mD['R_PSNR_of_mean_mse'], 5, 3, True) * float(M['msFcMdeFactorAssignDirectPsnr']), '{:.3g}')
# matched-baseline stratification simulation, weighted estimator variant (round 6, B1)
for a in ['direct', 'unet']:
    for y in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse']:
        mm = m2['archs'][a]['metrics'][y]
        if 'H6_sim_variance_reduction_weighted' in mm: mac(f'hsixSimW{ARCH[a]}{MET[y]}', mm['H6_sim_variance_reduction_weighted'], '{:+.2f}')
# precision of the 20-run two-assignment design (round 7, C1): validate_r2_precision.py -> r2_precision.json
RP = R / 'r2_precision.json'
if RP.exists():
    rp = json.load(open(RP)); rs = rp['results']; mac('rpNsim', rp['n_sim'], '{:d}')
    for k, tag in [('r2s2', 'Rtwo'), ('full', 'Full')]:
        mac(f'rpSceneRelSdMin{tag}', min(r['designs'][k]['s2_scene']['rel_sd_vs_truth'] for r in rs)); mac(f'rpSceneRelSdMax{tag}', max(r['designs'][k]['s2_scene']['rel_sd_vs_truth'] for r in rs))
        mac(f'rpInterRelSdMin{tag}', min(r['designs'][k]['s2_inter']['rel_sd_vs_truth'] for r in rs)); mac(f'rpInterRelSdMax{tag}', max(r['designs'][k]['s2_inter']['rel_sd_vs_truth'] for r in rs))
        mac(f'rpMdeRelSdMin{tag}', min(r['designs'][k]['mde_full']['rel_sd_vs_truth'] for r in rs)); mac(f'rpMdeRelSdMax{tag}', max(r['designs'][k]['mde_full']['rel_sd_vs_truth'] for r in rs))
        mac(f'rpInterZeroMax{tag}', max(r['designs'][k]['s2_inter']['frac_zero'] for r in rs))
# searched-design ratio intervals (round 7, B2) from the bootstrap
if BSF.exists():
    bs = json.load(open(BSF)); r = bs['pairs'].get('unet - direct|simse_R_gray_mean', {})
    for key, t in [('gap_over_mde_R6S1', 'RsixSone'), ('gap_over_mde_R9S1', 'RnineSone'), ('gap_over_mde_R2S2', 'RtwoStwo')]:
        if key in r: mac(f'gapMde{t}UnetDirectSimseLo', r[key]['ci95'][0]); mac(f'gapMde{t}UnetDirectSimseHi', r[key]['ci95'][1]); mac(f'gapMde{t}UnetDirectSimseMed', r[key]['median'])
# training-size fixed-effect sensitivity (round 8, A1): trainsize_adjust.py -> trainsize_adjust.json
TA = R / 'trainsize_adjust.json'
if TA.exists():
    ta = json.load(open(TA)); ps = [ta['archs'][a]['R_PSNR_of_mean_mse'] for a in ta['archs']]
    red = [1 - x['size_adjusted']['s2_train'] / x['primary']['s2_train'] for x in ps if x['primary']['s2_train'] > 0]
    mac('taTrainRedPsnrMin', 100 * min(red), '{:.0f}'); mac('taTrainRedPsnrMax', 100 * max(red), '{:.0f}')
    fr = [1 - x['ladder_size_adjusted']['fullcov_S3'] / x['ladder_primary']['fullcov_S3'] for x in ps]; rr = [1 - x['ladder_size_adjusted']['K5_S3'] / x['ladder_primary']['K5_S3'] for x in ps]
    mac('taFullMdeRedPsnrMin', 100 * min(fr), '{:.0f}'); mac('taFullMdeRedPsnrMax', 100 * max(fr), '{:.0f}'); mac('taRandMdeRedPsnrMin', 100 * min(rr), '{:.0f}'); mac('taRandMdeRedPsnrMax', 100 * max(rr), '{:.0f}')
    sh = [x['fullcov_target_shift_dB_mixture_minus_18'] for x in ps]; mac('taShiftPsnrMin', min(sh)); mac('taShiftPsnrMax', max(sh))
    d_ = ta['archs']['direct']['R_PSNR_of_mean_mse']; mac('taDirectFullPrimary', d_['ladder_primary']['fullcov_S3']); mac('taDirectFullAdj', d_['ladder_size_adjusted']['fullcov_S3']); mac('taDirectRandPrimary', d_['ladder_primary']['K5_S3']); mac('taDirectRandAdj', d_['ladder_size_adjusted']['K5_S3'])
    mac('taDirectTrainPrimary', d_['primary']['s2_train'], '{:.3f}'); mac('taDirectTrainAdj', d_['size_adjusted']['s2_train'], '{:.3f}')
    flips = sum((x['ladder_size_adjusted']['fullcov_S3'] < x['ladder_size_adjusted']['K5_S3']) != (x['ladder_primary']['fullcov_S3'] < x['ladder_primary']['K5_S3']) for x in ps); mac('taOrderFlips', flips, '{:d}')
    pr = ta['pairs']; chg = [abs(v['size_adjusted']['gap_over_mde_full'] - v['primary']['gap_over_mde_full']) for v in pr.values()]
    mac('taPairRatioMaxChange', max(chg)); mac('taNpairs', len(pr), '{:d}')
    ud = pr['unet - direct|R_PSNR_of_mean_mse']; mac('taUnetDirectSizeDiffPsnr', ud['method_by_size_effect'], '{:+.2f}'); mac('taUnetDirectSizeDiffSePsnr', ud['se'])
    mx = max(abs(v['method_by_size_effect']) / v['se'] for v in pr.values() if v['se'] > 0); mac('taPairSizeMaxZ', mx, '{:.1f}')
# --- qualitative runs (representative configurations retrained for display; numbers for captions/text)
Qd = R / 'qual'
if (Qd / 'qual_intrinsic_meta.json').exists():
    mi = json.load(open(Qd / 'qual_intrinsic_meta.json')); mf = json.load(open(Qd / 'qual_flow_meta.json'))
    comps = sorted(mi['jobs']['gauge']['compositions'], key=lambda c: c['rep'])
    inn = [c for c in comps if c['bamboo_1_in_train']]; outn = [c for c in comps if not c['bamboo_1_in_train']]
    mac('qGaugeScene', mi['jobs']['gauge']['scene'].replace('_', r'\_')); mac('qGaugeInPsnrMin', min(c['scene_psnr'] for c in inn), '{:.1f}'); mac('qGaugeInPsnrMax', max(c['scene_psnr'] for c in inn), '{:.1f}')
    mac('qGaugeOutPsnrMin', min(c['scene_psnr'] for c in outn), '{:.1f}'); mac('qGaugeOutPsnrMax', max(c['scene_psnr'] for c in outn), '{:.1f}')
    mac('qGaugeInGainPsnrMin', min(c['scene_gain_psnr'] for c in inn), '{:.1f}'); mac('qGaugeInGainPsnrMax', max(c['scene_gain_psnr'] for c in inn), '{:.1f}')
    mac('qGaugeOutGainPsnrMin', min(c['scene_gain_psnr'] for c in outn), '{:.1f}'); mac('qGaugeOutGainPsnrMax', max(c['scene_gain_psnr'] for c in outn), '{:.1f}')
    mac('qGaugeInGainMin', min(c['optimal_gain'] for c in inn)); mac('qGaugeInGainMax', max(c['optimal_gain'] for c in inn)); mac('qGaugeOutGainMin', min(c['optimal_gain'] for c in outn)); mac('qGaugeOutGainMax', max(c['optimal_gain'] for c in outn))
    mac('qGaugeNin', len(inn), '{:d}'); mac('qGaugeNout', len(outn), '{:d}')
    ps = mi['jobs']['inter']['per_scene_psnr']
    for k, v in ps.items(): mac('qInter' + k.replace('/', '').replace('_', ''), v, '{:.1f}')
    for a, MM in mf['models'].items():
        for sc, v in MM['scene_epe_finetuned'].items(): mac('qFlow' + ARCH[a] + sc.replace('_', ''), v, '{:.2f}')
# difficulty orderings of the two tasks (Fig. scenes): five hardest scenes per task and their overlap
_si = {r['scene']: r['score'] for r in m2['archs']['direct']['metrics']['simse_R_gray_mean']['scene_table']}
_ep = {r['scene']: r['score'] for r in fl['archs']['raft_small']['metrics']['epe_mean']['scene_table']}
_hi = sorted(_si, key=_si.get, reverse=True)[:5]; _hf = sorted(_ep, key=_ep.get, reverse=True)[:5]
for i, sc in enumerate(_hi): mac(f'intrHard{["One", "Two", "Three", "Four", "Five"][i]}', sc.replace('_', r'\_'))
for i, sc in enumerate(_hf): mac(f'flowHardFive{["One", "Two", "Three", "Four", "Five"][i]}', sc.replace('_', r'\_'))
mac('hardOverlap', len(set(_hi) & set(_hf)), '{:d}')
_rank_f = {sc: k + 1 for k, sc in enumerate(sorted(_ep, key=_ep.get, reverse=True))}; _rank_i = {sc: k + 1 for k, sc in enumerate(sorted(_si, key=_si.get, reverse=True))}
def _ord(n): return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
mac('rankFlowMountainOne', _ord(_rank_f['mountain_1'])); mac('rankIntrAmbushFour', _ord(_rank_i['ambush_4'])); mac('rankFlowBambooTwo', _ord(_rank_f['bamboo_2'])); mac('rankIntrBambooTwo', _ord(_rank_i['bamboo_2']))
# zoom-window statistics written by make_qual_figs.py (automatic patch selection; see its docstring)
Qs = Path(__file__).parent / 'figs' / 'qual_fig_stats.json'
if Qs.exists():
    Z = json.load(open(Qs)).get('zoom_windows', {})
    for sc, z in Z.get('fig8', {}).items():
        mac('qPatchRmseDirect' + sc.replace('_', ''), z['patch_rmse_direct'], '{:.2f}'); mac('qPatchRmseUnet' + sc.replace('_', ''), z['patch_rmse_unet'], '{:.2f}')
    for sc, z in Z.get('fig9', {}).items():
        mac('qPatchEpeRaftS' + sc.replace('_', ''), z['patch_epe_small'], '{:.2f}'); mac('qPatchEpeRaftL' + sc.replace('_', ''), z['patch_epe_large'], '{:.2f}')
# sibling-family structure of the interaction (round 9, A1/C1): family_effect.py -> family_effect.json
FE = R / 'family_effect.json'
if FE.exists():
    fe = json.load(open(FE)); ARCHS4 = ['direct', 'unet', 'transformer', 'proposed']
    def _nz(v): return '0' if abs(v) < 0.5 else f'{v:.0f}'   # percentages: no '-0'
    mac('famNcells', 23 * 5, '{:d}'); _no = fe['intrinsic']['direct']['R_PSNR_of_mean_mse']['other_two_member_scenes']['n']
    mac('famNtwoMemberScenesOtherWord', {6: 'six', 8: 'eight', 10: 'ten', 12: 'twelve'}.get(_no, str(_no)))
    _never = sorted(f for f, k in fe['realised_coholds'].items() if k == 0); mac('famNeverCoheldFamilies', ' and '.join(_never)); mac('famNneverCoheld', len(_never), '{:d}')
    _W = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
    mac('famNtwoMemberWord', _W.get(len(fe['two_member_families']), str(len(fe['two_member_families']))))
    _rb = fe['realised_coholds']['bamboo']; mac('famRealisedBambooWord', {1: 'once', 2: 'twice'}.get(_rb, f'{_rb} times'))
    _bp = [fe['intrinsic'][a]['R_PSNR_of_mean_mse']['beta_focal_present_minus_absent'] for a in ARCHS4]; mac('famBetaMinPsnrOne', min(_bp), '{:.1f}'); mac('famBetaMaxPsnrOne', max(_bp), '{:.1f}')
    mac('famNtwoMember', len(fe['two_member_families']), '{:d}'); mac('famPcohold', fe['p_cohold_per_assignment']); mac('famExpectedCoholds', fe['expected_coholds_in_5'])
    mac('famRealisedBamboo', fe['realised_coholds']['bamboo'], '{:d}'); mac('famRealisedTotal', fe['realised_coholds_total'], '{:d}'); mac('famRealisedRate', fe['realised_cohold_rate'])
    mac('famPbambooGeRealised', fe['p_focal_coholds_ge_realised']); mac('famNfamilies', len(fe['families']), '{:d}')
    for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_siPSNR_scenegain_rgb']:
        rs = [fe['intrinsic'][a][y] for a in ARCHS4]; t = MET[y]
        b = [r['beta_focal_present_minus_absent'] for r in rs]; mac(f'famBetaMin{t}', min(b), '{:.2f}' if 'PSNR' in y else '{:.3g}'); mac(f'famBetaMax{t}', max(b), '{:.2f}' if 'PSNR' in y else '{:.3g}')
        o = [r['other_two_member_scenes']['max_abs'] for r in rs]; mac(f'famOtherMaxAbs{t}', max(o), '{:.2f}' if 'PSNR' in y else '{:.3g}')
        rd = [100 * r['interaction_reduction_fixed_effect'] for r in rs]; mac(f'famInterRedMin{t}', _nz(min(rd))); mac(f'famInterRedMax{t}', _nz(max(rd)))
        rd2 = [100 * r['interaction_reduction_dropped'] for r in rs]; mac(f'famInterRedDropMin{t}', _nz(min(rd2))); mac(f'famInterRedDropMax{t}', _nz(max(rd2)))
        for a, r in zip(ARCHS4, rs):
            ta = ARCH[a] + t
            mac(f'famBeta{ta}', r['beta_focal_present_minus_absent'], '{:+.2f}' if 'PSNR' in y else '{:+.3g}'); mac(f'famInterPrimary{ta}', r['primary']['s2_inter'], '{:.3g}'); mac(f'famInterFixed{ta}', r['focal_fixed_effect_removed']['s2_inter'], '{:.3g}')
            mac(f'famInterDrop{ta}', r['focal_family_dropped_U21']['s2_inter'], '{:.3g}'); mac(f'famTrainFixed{ta}', r['focal_fixed_effect_removed']['s2_train'], '{:.3g}')
            mac(f'famSharePrimary{ta}', r['primary']['test_share']); mac(f'famShareFixed{ta}', r['focal_fixed_effect_removed']['test_share']); mac(f'famIccPrimary{ta}', r['primary']['icc']); mac(f'famIccFixed{ta}', r['focal_fixed_effect_removed']['icc'])
            mac(f'famMdeFullPrimary{ta}', r['primary']['mde_abs_full_S3'], '{:.3g}'); mac(f'famMdeFullFixed{ta}', r['focal_fixed_effect_removed']['mde_abs_full_S3'], '{:.3g}')
        mac(f'famShareFixedMin{t}', min(r['focal_fixed_effect_removed']['test_share'] for r in rs)); mac(f'famShareFixedMax{t}', max(r['focal_fixed_effect_removed']['test_share'] for r in rs))
        mac(f'famIccFixedMin{t}', min(r['focal_fixed_effect_removed']['icc'] for r in rs)); mac(f'famIccFixedMax{t}', max(r['focal_fixed_effect_removed']['icc'] for r in rs))
        mac(f'famTrainFixedMax{t}', max(r['focal_fixed_effect_removed']['s2_train'] for r in rs), '{:.3g}')
    for a in ARCHS4:
        for y in ['R_PSNR_of_mean_mse', 'dssim_R_mean']:
            dc = fe['draw_conditionality'][f'{a}|{y}']; ta = ARCH[a] + MET[y]
            mac(f'famSimInterMed{ta}', dc['s2_inter']['median'], '{:.3g}'); mac(f'famSimInterLo{ta}', dc['s2_inter']['p5'], '{:.3g}'); mac(f'famSimInterHi{ta}', dc['s2_inter']['p95'], '{:.3g}')
            mac(f'famSimInterPct{ta}', 100 * dc['realised_percentile'], '{:.0f}'); mac(f'famSimShareMed{ta}', dc['test_share']['median']); mac(f'famSimShareLo{ta}', dc['test_share']['p5']); mac(f'famSimShareHi{ta}', dc['test_share']['p95'])
            mac(f'famSimCoholdZero{ta}', 100 * dc['cohold_count']['dist'].get('0', 0) / dc['n_sim'], '{:.0f}'); mac(f'famSimCoholdGeTwo{ta}', 100 * sum(v for k, v in dc['cohold_count']['dist'].items() if int(k) >= 2) / dc['n_sim'], '{:.0f}')
            mac(f'famSimNsim{ta}', dc['n_sim'], '{:d}')
    pcts = [100 * fe['draw_conditionality'][f'{a}|R_PSNR_of_mean_mse']['realised_percentile'] for a in ARCHS4]; mac('famSimInterPctPsnrMin', min(pcts), '{:.0f}'); mac('famSimInterPctPsnrMax', max(pcts), '{:.0f}')
    meds = [fe['draw_conditionality'][f'{a}|R_PSNR_of_mean_mse']['test_share']['median'] for a in ARCHS4]; mac('famSimShareMedPsnrMin', min(meds)); mac('famSimShareMedPsnrMax', max(meds))
    ratio = [fe['intrinsic'][a]['R_PSNR_of_mean_mse']['primary']['s2_inter'] / fe['draw_conditionality'][f'{a}|R_PSNR_of_mean_mse']['s2_inter']['median'] for a in ARCHS4]; mac('famRealisedOverSimMedPsnrMin', min(ratio), '{:.1f}'); mac('famRealisedOverSimMedPsnrMax', max(ratio), '{:.1f}')
    pr = fe['pairs']; dpairs = {k: v for k, v in pr.items() if k.split('|')[0].endswith('- direct')}
    mac('famPairBetaPsnrMin', min(v['method_by_sibling_effect_focal'] for k, v in dpairs.items() if 'PSNR' in k), '{:+.2f}'); mac('famPairBetaPsnrMax', max(v['method_by_sibling_effect_focal'] for k, v in dpairs.items() if 'PSNR' in k), '{:+.2f}')
    chg = {k: abs(v['focal_fixed_effect_removed']['gap_over_mde_full'] - v['primary']['gap_over_mde_full']) for k, v in pr.items()}; kmax = max(chg, key=chg.get)
    mac('famPairRatioMaxChange', chg[kmax]); mac('famNpairs', len(pr), '{:d}')
    pd_ = pr['proposed - direct|lmse_R_chen_mean']; mac('famPropDirectLmseRatioPrimary', pd_['primary']['gap_over_mde_full']); mac('famPropDirectLmseRatioFixed', pd_['focal_fixed_effect_removed']['gap_over_mde_full'])
    ud_ = pr['unet - direct|R_PSNR_of_mean_mse']; mac('famUnetDirectPsnrRatioPrimary', ud_['primary']['gap_over_mde_full']); mac('famUnetDirectPsnrRatioFixed', ud_['focal_fixed_effect_removed']['gap_over_mde_full']); mac('famUnetDirectPsnrBeta', ud_['method_by_sibling_effect_focal'], '{:+.2f}')
    for a in ['raft_small', 'raft_large']:
        f_ = fe['flow'][a]['epe_mean']; mac(f'famFlowBeta{ARCH[a]}', f_['beta_focal_present_minus_absent'], '{:+.3f}'); mac(f'famFlowMaxAbs{ARCH[a]}', f_['all_two_member_scenes_max_abs']); mac(f'famFlowMaxAbsSd{ARCH[a]}', f_['max_abs_over_between_scene_sd'])
    mac('famFlowMaxAbsSdMax', max(fe['flow'][a]['epe_mean']['max_abs_over_between_scene_sd'] for a in ['raft_small', 'raft_large']))
    mac('famFlowInterRedMax', 100 * max(abs(fe['flow'][a]['epe_mean']['interaction_reduction_fixed_effect']) for a in ['raft_small', 'raft_large']), '{:.1f}')
# joint size + family fit, uncertainty, structured design variance (round 10, A1/B1/C1/C2): family_joint.py -> family_joint.json
FJ = R / 'family_joint.json'
if FJ.exists():
    fj = json.load(open(FJ)); ARCHS4 = ['direct', 'unet', 'transformer', 'proposed']; Y = 'R_PSNR_of_mean_mse'
    ps = [fj['intrinsic'][a][Y] for a in ARCHS4]
    sj = [x['joint_ols']['size4']['est'] for x in ps]; mac('fjSizeJointPsnrMin', min(sj), '{:+.2f}'); mac('fjSizeJointPsnrMax', max(sj), '{:+.2f}')
    se = [x['joint_ols']['size4']['se_cluster'] for x in ps]; mac('fjSizeJointSePsnrMin', min(se)); mac('fjSizeJointSePsnrMax', max(se))
    sa = [x['size_only_est'] for x in ps]; mac('fjSizeAlonePsnrMin', min(sa), '{:+.2f}'); mac('fjSizeAlonePsnrMax', max(sa), '{:+.2f}')
    bj = [x['joint_ols']['bam_abs']['est'] for x in ps]; mac('fjBamJointPsnrMinAbs', min(abs(v) for v in bj), '{:.1f}'); mac('fjBamJointPsnrMaxAbs', max(abs(v) for v in bj), '{:.1f}')
    mac('fjBamJointPsnrMinAbsTwo', min(abs(v) for v in bj), '{:.2f}'); mac('fjBamJointPsnrMaxAbsTwo', max(abs(v) for v in bj), '{:.2f}')
    bse = [x['joint_ols']['bam_abs']['se_cluster'] for x in ps]; mac('fjBamSePsnrMin', min(bse)); mac('fjBamSePsnrMax', max(bse))
    cf = [7 * abs(v) / 120 for v in bj]; mac('fjConfoundPsnrMin', min(cf)); mac('fjConfoundPsnrMax', max(cf))
    sh = [x['target_shift_mixture_minus_18_joint'] for x in ps]; mac('fjShiftJointPsnrMaxAbs', max(abs(v) for v in sh)); mac('fjShiftJointPsnrMin', min(sh), '{:+.2f}'); mac('fjShiftJointPsnrMax', max(sh), '{:+.2f}')
    rd = [100 * x['inter_reduction_joint'] for x in ps]; mac('fjRedJointPsnrMin', min(rd), '{:.0f}'); mac('fjRedJointPsnrMax', max(rd), '{:.0f}')
    gl = [abs(x['joint_gls']['size4']['est'] - x['joint_ols']['size4']['est']) for x in ps]; mac('fjGlsOlsSizeMaxDiff', max(gl))
    for a, x in zip(ARCHS4, ps):
        t = ARCH[a]; mac(f'fjSizeJoint{t}Psnr', x['joint_ols']['size4']['est'], '{:+.2f}'); mac(f'fjSizeJointSe{t}Psnr', x['joint_ols']['size4']['se_cluster'])
        mac(f'fjBamJoint{t}Psnr', x['joint_ols']['bam_abs']['est'], '{:+.2f}'); mac(f'fjBamSe{t}Psnr', x['joint_ols']['bam_abs']['se_cluster'])
        mac(f'fjInterJoint{t}Psnr', x['joint_adjusted']['s2_inter'], '{:.3g}'); mac(f'fjShareJoint{t}Psnr', x['joint_adjusted']['test_share']); mac(f'fjIccJoint{t}Psnr', x['joint_adjusted']['icc'])
        mac(f'fjFullJoint{t}Psnr', x['joint_adjusted']['ladder']['fullcov_S3'], '{:.3g}'); mac(f'fjRandJoint{t}Psnr', x['joint_adjusted']['ladder']['K5_S3'], '{:.3g}')
        b = fj['bootstrap'][f'{a}|{Y}']
        mac(f'fjBamCiLo{t}Psnr', b['b_bam']['ci95'][0], '{:.2f}'); mac(f'fjBamCiHi{t}Psnr', b['b_bam']['ci95'][1], '{:.2f}')
        mac(f'fjSizeCiLo{t}Psnr', b['b_size']['ci95'][0], '{:+.2f}'); mac(f'fjSizeCiHi{t}Psnr', b['b_size']['ci95'][1], '{:+.2f}')
        mac(f'fjRedCiLo{t}Psnr', 100 * b['reduction']['ci95'][0], '{:.0f}'); mac(f'fjRedCiHi{t}Psnr', 100 * b['reduction']['ci95'][1], '{:.0f}')
        mac(f'fjShareCiLo{t}Psnr', b['share_adj']['ci95'][0]); mac(f'fjShareCiHi{t}Psnr', b['share_adj']['ci95'][1])
        mac(f'fjIccCiLo{t}Psnr', b['icc_adj']['ci95'][0]); mac(f'fjIccCiHi{t}Psnr', b['icc_adj']['ci95'][1])
        lo = [v for v in fj['loo'][f'{a}|{Y}'].values() if v.get('reduction') is not None]
        mac(f'fjLooRedMin{t}Psnr', 100 * min(v['reduction'] for v in lo), '{:.0f}'); mac(f'fjLooRedMax{t}Psnr', 100 * max(v['reduction'] for v in lo), '{:.0f}')
        mac(f'fjLooBamMinAbs{t}Psnr', min(abs(v['b_bam']) for v in lo), '{:.2f}'); mac(f'fjLooBamMaxAbs{t}Psnr', max(abs(v['b_bam']) for v in lo), '{:.2f}')
        s = fj['structured_absolute'][a]
        for k, kk in [('one', 'One'), ('five', 'Five'), ('full', 'Full')]:
            mac(f'fjStructRatio{kk}{t}Psnr', s['ratio_empirical_over_primary'][k]); mac(f'fjStructEmp{kk}{t}Psnr', s['empirical_var'][k], '{:.3g}'); mac(f'fjStructPred{kk}{t}Psnr', s['predicted_var_primary'][k], '{:.3g}')
            mac(f'fjStructMdeFac{kk}{t}Psnr', s['mde_factor_vs_primary'][k]); mac(f'fjStructClosed{kk}{t}Psnr', s['ratio_empirical_over_joint_plus_closed'][k])
    mac('fjBootN', fj['bootstrap'][f'direct|{Y}']['n_boot'], '{:d}'); mac('fjStructN', fj['structured_absolute']['direct']['n_sim'], '{:d}'); mac('fjDecN', fj['structured_paired']['unet - direct|R_PSNR_of_mean_mse']['n_sim'], '{:d}')
    mac('fjRedCiLoMinPsnr', 100 * min(fj['bootstrap'][f'{a}|{Y}']['reduction']['ci95'][0] for a in ARCHS4), '{:.0f}'); mac('fjRedCiHiMaxPsnr', 100 * max(fj['bootstrap'][f'{a}|{Y}']['reduction']['ci95'][1] for a in ARCHS4), '{:.0f}')
    lo_all = [v['reduction'] for a in ARCHS4 for v in fj['loo'][f'{a}|{Y}'].values() if v.get('reduction') is not None]; mac('fjLooRedMinPsnr', 100 * min(lo_all), '{:.0f}'); mac('fjLooRedMaxPsnr', 100 * max(lo_all), '{:.0f}')
    for k, kk in [('one', 'One'), ('five', 'Five'), ('full', 'Full')]:
        r_ = [fj['structured_absolute'][a]['ratio_empirical_over_primary'][k] for a in ARCHS4]; mac(f'fjStructRatio{kk}PsnrMin', min(r_)); mac(f'fjStructRatio{kk}PsnrMax', max(r_))
        f_ = [fj['structured_absolute'][a]['mde_factor_vs_primary'][k] for a in ARCHS4]; mac(f'fjStructMdeFac{kk}PsnrMin', min(f_)); mac(f'fjStructMdeFac{kk}PsnrMax', max(f_))
        c_ = [fj['structured_absolute'][a]['ratio_empirical_over_joint_plus_closed'][k] for a in ARCHS4]; mac(f'fjStructClosed{kk}PsnrMin', min(c_)); mac(f'fjStructClosed{kk}PsnrMax', max(c_))
        e_ = [fj['structured_absolute'][a]['ratio_empirical_over_joint_exchangeable'][k] for a in ARCHS4]; mac(f'fjStructExch{kk}PsnrMin', min(e_), '{:.1f}'); mac(f'fjStructExch{kk}PsnrMax', max(e_), '{:.1f}')
    mac('fjPshareFold', fj['structured_absolute']['direct']['p_share_fold'], '{:.3f}'); mac('fjQbothInFive', fj['structured_absolute']['direct']['q_both_in_five'], '{:.3f}')
    sp = fj['structured_paired']; mac('fjDecNullMin', min(min(v['null'].values()) for v in sp.values()), '{:.3f}'); mac('fjDecNullMax', max(max(v['null'].values()) for v in sp.values()), '{:.3f}')
    mac('fjDecPowerMin', min(min(v['power'].values()) for v in sp.values())); mac('fjDecPowerMax', max(max(v['power'].values()) for v in sp.values()))
    mac('fjDecDbamMinAbs', min(abs(v['method_by_sibling_effect']) for v in sp.values())); mac('fjDecDbamMaxAbs', max(abs(v['method_by_sibling_effect']) for v in sp.values()))
    pr = fj['pairs']; dp = {k: v for k, v in pr.items() if k.split('|')[0].endswith('- direct') and 'PSNR' in k}
    mac('fjPairDbamPsnrMinAbs', min(abs(v['joint_ols']['bam_abs']['est']) for v in dp.values())); mac('fjPairDbamPsnrMaxAbs', max(abs(v['joint_ols']['bam_abs']['est']) for v in dp.values()))
    chf = {k: abs(v['joint_adjusted']['gap_over_mde_full_fixed_numerator'] - v['primary']['gap_over_mde_full']) for k, v in pr.items()}; cha = {k: abs(v['joint_adjusted']['gap_over_mde_full_adjusted_numerator'] - v['primary']['gap_over_mde_full']) for k, v in pr.items()}
    mac('fjPairFixedNumMaxChange', max(chf.values())); mac('fjPairAdjNumMaxChange', max(cha.values())); mac('fjNpairs', len(pr), '{:d}')
    ud = pr['unet - direct|R_PSNR_of_mean_mse']; mac('fjUDPsnrRatioPrimary', ud['primary']['gap_over_mde_full']); mac('fjUDPsnrRatioFixed', ud['joint_adjusted']['gap_over_mde_full_fixed_numerator']); mac('fjUDPsnrRatioAdj', ud['joint_adjusted']['gap_over_mde_full_adjusted_numerator'])
    mac('fjUDPsnrGapAdj', ud['gap_adjusted'], '{:+.3f}'); mac('fjUDPsnrDbam', ud['joint_ols']['bam_abs']['est'], '{:+.2f}'); mac('fjUDPsnrDsize', ud['joint_ols']['size4']['est'], '{:+.2f}'); mac('fjUDPsnrDsizeSe', ud['joint_ols']['size4']['se_cluster'])
    pdl = pr['proposed - direct|lmse_R_chen_mean']; mac('fjPDLmseRatioPrimary', pdl['primary']['gap_over_mde_full']); mac('fjPDLmseRatioFixed', pdl['joint_adjusted']['gap_over_mde_full_fixed_numerator']); mac('fjPDLmseRatioAdj', pdl['joint_adjusted']['gap_over_mde_full_adjusted_numerator'])
    udl = pr['unet - direct|lmse_R_chen_mean']; mac('fjUDLmseRatioPrimary', udl['primary']['gap_over_mde_full']); mac('fjUDLmseRatioFixed', udl['joint_adjusted']['gap_over_mde_full_fixed_numerator']); mac('fjUDLmseRatioAdj', udl['joint_adjusted']['gap_over_mde_full_adjusted_numerator'])
    cnt_fix = sum(v['joint_adjusted']['gap_over_mde_full_fixed_numerator'] >= 1 for v in pr.values()); cnt_adj = sum(v['joint_adjusted']['gap_over_mde_full_adjusted_numerator'] >= 1 for v in pr.values()); cnt0 = sum(v['primary']['gap_over_mde_full'] >= 1 for v in pr.values())
    mac('fjPairCountFullPrimary', cnt0, '{:d}'); mac('fjPairCountFullFixed', cnt_fix, '{:d}'); mac('fjPairCountFullAdj', cnt_adj, '{:d}')
    dsz = [abs(v['joint_ols']['size4']['est']) / v['joint_ols']['size4']['se_cluster'] for v in pr.values() if v['joint_ols']['size4']['se_cluster'] > 0]; mac('fjPairDsizeMaxZ', max(dsz), '{:.1f}')
    mac('fjPairDsizeWithinTwoSe', sum(z <= 2 for z in dsz), '{:d}'); zk = max(pr, key=lambda k: abs(pr[k]['joint_ols']['size4']['est']) / pr[k]['joint_ols']['size4']['se_cluster'])
    mac('fjPairDsizeMaxPairDsize', pr[zk]['joint_ols']['size4']['est'], '{:+.2g}'); mac('fjPairDsizeMaxPairSe', pr[zk]['joint_ols']['size4']['se_cluster'], '{:.2g}'); mac('fjPairDsizeMaxPairGap', pr[zk]['gap_observed'], '{:+.2g}')
    _pn = {'proposed': 'Proposed', 'direct': 'DirectCNN', 'unet': 'UNet', 'transformer': 'TransUNet'}; _pm = {'simse_R_gray_mean': 'si-MSE', 'lmse_R_chen_mean': 'LMSE', 'R_PSNR_of_mean_mse': 'PSNR'}
    _b, _a = zk.split('|')[0].split(' - '); mac('fjPairDsizeMaxPairName', f"{_pn[_b]}--{_pn[_a]} under {_pm[zk.split('|')[1]]}")
lines = ['% numbers.tex — GENERATED by make_numbers.py; do not edit by hand.'] + [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(M.items())]
out.write_text('\n'.join(lines) + '\n'); print(out, len(M), 'macros')
