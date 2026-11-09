#!/usr/bin/env python3
"""make_tables.py — generate tables/*.tex for the IJCV manuscript from the result JSONs (no hand-typed numbers).
T1 design & compute | T2 intrinsic decomposition | T3 flow decomposition | T4 resolution (MDE ladders + pair cells) | T5 controls (non-determinism, H5, H6)."""
import json
from pathlib import Path
import numpy as np

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution'); T = Path(__file__).parent / 'tables'; T.mkdir(exist_ok=True)
k1 = json.load(open(R / 'killing_v1.json')); m2 = json.load(open(R / 'main_v2.json')); fl = json.load(open(R / 'flow_v1.json'))
AN = {'direct': 'DirectCNN', 'unet': 'UNet', 'transformer': 'TransUNet', 'proposed': 'Proposed', 'raft_small': 'RAFT-small', 'raft_large': 'RAFT-large'}
# series-2 round 7 (C1): LMSE / SSIM spatial support at the 256x256 working resolution, from the window constants of experiment_cv2.py
# (_lmse_chen: k = round(0.1*min(h,w)) = 26, stride k//2 = 13, torch unfold without padding; _ssim: 11x11 window, conv2d without padding)
_HW = 256; _K = max(2, int(round(0.10 * _HW))); _STEP = max(1, _K // 2); _NPOS = (_HW - _K) // _STEP + 1; _LAST = (_NPOS - 1) * _STEP + _K - 1
_NEVER = _HW - 1 - _LAST; _FRAC_NEVER = 100 * (1 - ((_LAST + 1) / _HW) ** 2)
_LMSE_BOUNDARY = (f'{_NPOS} positions per axis at ${_HW}\\times{_HW}$; the last window ends at row and column {_LAST}, so the final {_NEVER} rows and '
                  f'columns, {_FRAC_NEVER:.1f}\\% of the pixels, enter no window')
_SSIM_MAP = f'{_HW - 11 + 1}\\times{_HW - 11 + 1}'
assert (_K, _STEP, _NPOS, _LAST, _NEVER) == (26, 13, 18, 246, 9) and abs(_FRAC_NEVER - 6.9) < 0.05, (_K, _STEP, _NPOS, _LAST, _NEVER, _FRAC_NEVER)  # checked against Fn.unfold/conv2d
MN = {'simse_R_gray_mean': 'si-MSE', 'lmse_R_chen_mean': 'LMSE', 'dssim_R_mean': 'DSSIM', 'R_PSNR_of_mean_mse': 'PSNR', 'R_siPSNR_scenegain_rgb': 'PSNR, scene gain',
      'epe_mean': 'EPE', 'epe_pixel_weighted': 'EPE (pixel-w.)', 'fl_all_mean': 'Fl-all', 'out3_mean': '$>$3\\,px rate', 'epe_occ_mean': 'EPE occ.', 'epe_noc_mean': 'EPE non-occ.'}
def g(x): return f'{x:.3g}'
def gs(x, sig=2):
    """signed, LaTeX scientific notation for small magnitudes (|x|<1e-2), plain otherwise."""
    if x == 0: return '0'
    if abs(x) < 1e-2:
        e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
        return f'${m:+.{sig-1}f}\\times10^{{{e}}}$'
    return f'{x:+.{sig}g}'
def comp(m): return m['reml'] if m['used'] == 'reml' else m['moment']
BS = json.load(open(R / 'bootstrap_ci.json')) if (R / 'bootstrap_ci.json').exists() else None
def ci(exp, a, y, key, nd=2):
    """'[lo, hi]' string of the parametric-bootstrap interval for share/icc, or '' when the file is absent. nd decimals (3 for the flow
    table, whose shares lie within 0.02 of one; round 4, B5)."""
    try: lo, hi = BS[exp][a][y][key]['ci95']; return f' [{lo:.{nd}f}, {hi:.{nd}f}]'
    except Exception: return ''
import design_mde as D

# ---- T1 design
gpu_main = sum(A['mean_minutes'] * A['n_runs'] for A in m2['archs'].values()) / 60 + sum(14.7 * 20 + 23.9 * 20 for _ in [0]) / 60 * 0  # strat runs included in n_runs? no: n_runs counts rand only
strat_runs = m2['n_runs_by_design'].get('strat', 0)
gpu_main = (sum(A['mean_minutes'] * A['n_runs'] for A in m2['archs'].values()) + strat_runs / 2 * (m2['archs']['direct']['mean_minutes'] + m2['archs']['unet']['mean_minutes'])) / 60
gpu_kill = sum(k1['archs'][a]['n_runs'] * (14.6 if a == 'direct' else 24.1) for a in k1['archs']) / 60   # per-run minutes recorded in killing runs
kill_min = {}
import glob
for fp in glob.glob(str(R / 'results_cv_v1' / '*' / 'cv_*_v1.json')):
    if 'folds' in fp: continue
    d = json.load(open(fp)); kill_min.setdefault(d['arch'], []).append(d['minutes'])
gpu_kill = sum(sum(v) for v in kill_min.values()) / 60
gpu_flow = sum(A['mean_minutes'] * A['n_runs'] for A in fl['archs'].values()) / 60
rows = [
 ('Killing experiment', 'intrinsic', 'DirectCNN, UNet', '5 random repeats $\\times$ 5 folds', '2 (42, 43)', f"{k1['n_cv_runs']} + 2 parity", f'{gpu_kill:.1f}'),
 ('Main experiment', 'intrinsic', 'DirectCNN, UNet, TransUNet, Proposed', '5 random repeats $\\times$ 5 folds; 2 stratified repeats (DirectCNN, UNet)', '3 (42--44); 2 for stratified', f"{m2['n_runs']}", f'{gpu_main:.1f}'),
 ('Flow arm', 'optical flow', 'RAFT-small, RAFT-large (C+T init.)', '5 random repeats $\\times$ 5 folds (same folds)', '3 (42--44)', f"{fl['n_runs']} + 2 zero-shot", f'{gpu_flow:.1f}'),
]
L = ['\\begin{table*}[!t]\\centering\\caption{The experimental programme. All runs use MPI Sintel (23 scenes); folds are shared across experiments (fold-assignment seeds 201--205). GPU hours on a single NVIDIA T4 per run; see Sec.~\\ref{sec:programme} for recipes.}\\label{tab:design}', '\\small\\resizebox{\\textwidth}{!}{\\begin{tabular}{p{2.3cm}p{1.5cm}p{3.6cm}p{4.6cm}p{2.2cm}p{1.6cm}r}\\toprule', 'Experiment & Task & Models & Design & Seeds & Runs & GPU-h \\\\\\midrule']
L += [' & '.join(r) + ' \\\\' for r in rows] + ['\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_design.tex').write_text('\n'.join(L) + '\n')

# ---- T2 intrinsic decomposition (shares of total scene-level variance + ICC + test-side share of predicted partition variance)
L = ['\\begin{table*}[!t]\\centering\\caption{Variance components of the scene-level score (random design; design-exact method of moments, Sec.~\\ref{sec:estimation}, with the REML cross-check in Supplementary Table~S1). Shares are fractions of $\\sigma^2_{\\mathrm{scene}}+\\sigma^2_{\\mathrm{train}}+\\sigma^2_{\\mathrm{inter}}+\\sigma^2_{\\mathrm{seed}}$ (``0.00\'\' denotes a share below 0.005, not a zero component; Supplementary Table~S1 gives the values). ``Test share\'\' is $V_{\\mathrm{test}}$ divided by the predicted variance of a random 5-of-23 partition mean averaged over six seeds (Eq.~\\ref{eq:partvar}); ICC is $\\sigma^2_{\\mathrm{scene}}/(\\sigma^2_{\\mathrm{scene}}+\\sigma^2_{\\mathrm{inter}}+\\sigma^2_{\\mathrm{seed}})$. Brackets: 95\\% parametric-bootstrap intervals of the moment estimator on the actual design, conditional on the 23 scenes (Sec.~\\ref{sec:estimation}). For PSNR the last two columns compare with the partition variance observed in the earlier single-partition grid, whose score is the frame-pooled run-level PSNR averaged over six seeds and not the unit-equal scene-mean PSNR modelled here (Sec.~\\ref{sec:partvar}): an approximate check across the two aggregations, not a same-statistic validation of Eq.~\\eqref{eq:partvar} \\citep{woo2026partition}.}\\label{tab:decomp}',
     '\\footnotesize\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrrrr}\\toprule', 'Model & Metric & scene & train & inter & seed & Test share & ICC & share vs.\\ obs. & $V_{\\mathrm{pred}}/V_{\\mathrm{obs}}$ \\\\\\midrule']
for a, A in m2['archs'].items():
    first = True
    for y in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse', 'R_siPSNR_scenegain_rgb']:
        m = A['metrics'][y]; c = comp(m); tot = c['s2_scene'] + c['s2_train'] + c['s2_inter'] + c['s2_seed']
        so = f"{m['share_test_side_vs_josaa_observed']:.2f}" if 'share_test_side_vs_josaa_observed' in m else '--'
        vp = f"{m['V_pred_over_josaa_observed']:.2f}" if 'V_pred_over_josaa_observed' in m else '--'
        L.append(f"{AN[a] if first else ''} & {MN[y]} & {c['s2_scene']/tot:.2f} & {c['s2_train']/tot:.2f} & {c['s2_inter']/tot:.2f} & {c['s2_seed']/tot:.2f} & {m['share_test_side_of_predicted']:.2f}{ci('intrinsic', a, y, 'share')} & {m['icc_scene_single_seed']:.2f}{ci('intrinsic', a, y, 'icc')} & {so} & {vp} \\\\")
        first = False
    L.append('\\addlinespace')
L += ['\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_decomp_intrinsic.tex').write_text('\n'.join(L) + '\n')

# ---- T3 flow decomposition
L = ['\\begin{table*}[!t]\\centering\\caption{Variance components for the optical-flow arm (random design; design-exact method of moments), as in Table~\\ref{tab:decomp}, with three decimals because the test-side shares lie within 0.02 of one; brackets are 95\\% parametric-bootstrap intervals. Zero-shot rows give the C+T model without Sintel fine-tuning (pair-mean EPE over all 1,041 pairs).}\\label{tab:decomp_flow}',
     '\\footnotesize\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrr}\\toprule', 'Model & Metric & scene & train & inter & seed & Test share & ICC \\\\\\midrule']
for a, A in fl['archs'].items():
    first = True
    for y in ['epe_mean', 'epe_pixel_weighted', 'fl_all_mean', 'out3_mean', 'epe_occ_mean', 'epe_noc_mean']:
        m = A['metrics'][y]; c = comp(m); tot = c['s2_scene'] + c['s2_train'] + c['s2_inter'] + c['s2_seed']
        L.append(f"{AN[a] if first else ''} & {MN[y]} & {c['s2_scene']/tot:.2f} & {c['s2_train']/tot:.2f} & {c['s2_inter']/tot:.2f} & {c['s2_seed']/tot:.2f} & {m['share_test_side_of_predicted']:.3f}{ci('flow', a, y, 'share', 3)} & {m['icc_scene_single_seed']:.3f}{ci('flow', a, y, 'icc', 3)} \\\\")
        first = False
    L.append(f"\\multicolumn{{8}}{{l}}{{\\quad zero-shot EPE {fl['zeroshot'][a]['epe_pair_mean']:.3f}; fine-tuned scene-mean EPE {A['FH4_zeroshot_vs_finetuned_scene_difficulty']['epe_finetuned_scene_mean']:.2f} vs.\\ zero-shot {A['FH4_zeroshot_vs_finetuned_scene_difficulty']['epe_zeroshot_mean']:.2f}; rank correlation {A['FH4_zeroshot_vs_finetuned_scene_difficulty']['spearman']:.3f}}} \\\\\\addlinespace")
L += ['\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_decomp_flow.tex').write_text('\n'.join(L) + '\n')

# ---- T4 resolution: absolute MDE ladders (si-MSE, PSNR for intrinsic; EPE for flow) + pair cells
L = ['\\begin{table*}[!t]\\centering\\caption{Resolution. Top: minimum detectable effect (two-sided $\\alpha=0.05$, power $0.8$) of a difference between two independently evaluated absolute scores, each under $K$ partitions with three seeds, or each under the full-coverage five-fold design (all 23 scenes scored, three seeds); the factor of two of Eq.~\\eqref{eq:seabs} is included, and the standard error of one score alone is the MDE divided by $z\\sqrt2\\approx3.96$. Full-coverage scores target the mixture of 18- and 19-scene training sets (Sec.~\\ref{sec:designs}; for one si-MSE pair the mixture and fixed-18-scene targets differ measurably, Sec.~\\ref{sec:results3}). Bottom: paired comparisons (the six intrinsic architecture pairs under si-MSE, LMSE, and PSNR, and the flow pair under EPE), with the actual mean gap, the share of the paired variance due to the scene$\\times$method interaction under the prevailing recipe, and the gap divided by the MDE. The ratio is a plug-in power assessment: a value $\\ge 1$ means that a true gap of the observed size would be detected with power $\\ge0.8$ under that design; it is not a verdict on the observed gap, whose own uncertainty Sec.~\\ref{sec:results2} reports.}\\label{tab:resolution}',
     '\\footnotesize\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrr}\\toprule', '\\multicolumn{8}{l}{\\textit{Absolute scores}}\\\\', 'Model & Metric & $K{=}1$ & $K{=}3$ & $K{=}5$ & $K{=}10$ & $K{=}23$ & full cov.\\ \\\\\\midrule']
for a, A in m2['archs'].items():
    for y in ['simse_R_gray_mean', 'R_PSNR_of_mean_mse']:
        lad = A['metrics'][y]['MDE_absolute']
        L.append(f"{AN[a]} & {MN[y]} & " + ' & '.join(g(lad[k]) for k in ['K1_S3', 'K3_S3', 'K5_S3', 'K10_S3', 'K23_S3', 'fullcov_5fold_S3']) + ' \\\\')
for a, A in fl['archs'].items():
    lad = A['metrics']['epe_mean']['MDE_absolute']
    L.append(f"{AN[a]} & EPE & " + ' & '.join(g(lad[k]) for k in ['K1_S3', 'K3_S3', 'K5_S3', 'K10_S3', 'K23_S3', 'fullcov_5fold_S3']) + ' \\\\')
L += ['\\midrule', '\\multicolumn{8}{l}{\\textit{Paired comparisons}}\\\\', 'Pair & Metric & gap & inter.\\ share & MDE $K{=}1$ & MDE full cov.\\ & gap/MDE$_{K=1}$ & gap/MDE$_{\\mathrm{full}}$ \\\\\\midrule']
for p in sorted(m2['pairs'], key=lambda p: ({'simse_R_gray_mean': 0, 'lmse_R_chen_mean': 1, 'R_PSNR_of_mean_mse': 2}.get(p['metric'], 9), p['pair'])):
    if p['metric'] not in ('simse_R_gray_mean', 'lmse_R_chen_mean', 'R_PSNR_of_mean_mse'): continue     # series-2 round 1 (B4): LMSE rows added (the decisive comparison)
    L.append(f"{AN[p['b']]}$-${AN[p['a']]} & {MN[p['metric']]} & {p['mean_gap']:+.3g} & {p['share_interaction_K1S3']:.2f} & {g(p['MDE_paired']['paired_K1_S3'])} & {g(p['MDE_paired']['paired_fullcov_5fold_S3'])} & {p['gap_over_MDE_K1']:.2f} & {p['gap_over_MDE_fullcov']:.2f} \\\\")
for p in fl['pairs']:
    if p['metric'] != 'epe_mean': continue
    L.append(f"RAFT-small$-$RAFT-large & EPE & {p['mean_gap']:+.3g} & {p['share_interaction_K1S3']:.2f} & {g(p['MDE_paired']['paired_K1_S3'])} & {g(p['MDE_paired']['paired_fullcov_5fold_S3'])} & {p['gap_over_MDE_K1']:.2f} & {p['gap_over_MDE_fullcov']:.2f} \\\\")
L += ['\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_resolution.tex').write_text('\n'.join(L) + '\n')

# ---- T5 controls: non-determinism, H5 seed variance, H6 stratification
nd = m2['nondeterminism_control']
_nr = nd['by_arch']['direct']['R_PSNR_of_mean_mse']['n_runs'] + nd['by_arch']['unet']['R_PSNR_of_mean_mse']['n_runs']
L = ['\\begin{table}[!t]\\centering\\caption{Controls. Top: same-configuration reruns (v1 vs.\\ v2; identical seed, training code, container image, and GPU type; metric functions identical up to formatting) at the scene level, per architecture: the mean rerun shift, and the implied single-run reproducibility variance, Eq.~\\eqref{eq:nondet}, pooled over that architecture\'s rerun pairs and scenes, as a fraction of its seed component, with a 95\\% interval for the ratio from a bootstrap over that architecture\'s ' + str(nd['by_arch']['direct']['R_PSNR_of_mean_mse']['n_compositions']) + ' training compositions drawn with replacement, on which both the rerun variance (numerator, ' + str(nd['by_arch']['direct']['R_PSNR_of_mean_mse']['n_runs']) + ' rerun pairs per architecture) and the pooled within-cell seed variance (denominator) are recomputed, so that their shared runs are preserved. Middle: scene-level seed variance by architecture (PSNR). Bottom: stratified versus random five-fold assignments of matched fold sizes (5, 5, 5, 4, 4), both reporting unweighted fold means (the systematic one has expected stratum weights 0.34, 0.36, 0.30 against the population\'s 0.348, 0.348, 0.304, an offset of at most 0.012 between-scene standard deviations; Sec.~\\ref{sec:results3}): the simulated fold-mean variance reduction from each model\'s own per-scene table and the observed reduction in the realised runs (two stratified repeats built on DirectCNN si-MSE strata against five random repeats).}\\label{tab:controls}',
     '\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lrrr}\\toprule', '\\multicolumn{4}{l}{\\textit{Single-run reproducibility component} (' + str(nd['n_common_scene_rows']) + ' scene rows, ' + str(_nr) + ' rerun pairs)}\\\\', 'Metric & mean shift D / U & ratio DirectCNN [95\\%] & ratio UNet [95\\%] \\\\\\midrule']
for y in ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean']:
    vd, vu = nd['by_arch']['direct'][y], nd['by_arch']['unet'][y]
    sd_, su_ = comp(m2['archs']['direct']['metrics'][y])['s2_seed'], comp(m2['archs']['unet']['metrics'][y])['s2_seed']
    rd = vd['nondet_var_single_run'] / sd_; ru = vu['nondet_var_single_run'] / su_
    cd = vd['ratio_ci95_jointboot_compositions']; cu = vu['ratio_ci95_jointboot_compositions']       # series-2 round 3 (B1): joint ratio bootstrap
    L.append(f"{MN[y]} & {gs(vd['mean_diff'])} / {gs(vu['mean_diff'])} & {rd:.2f} [{cd[0]:.2f}, {cd[1]:.2f}] & {ru:.2f} [{cu[0]:.2f}, {cu[1]:.2f}] \\\\")
L += ['\\midrule', '\\multicolumn{4}{l}{\\textit{Seed variance by architecture (PSNR)}}\\\\', 'Model & scene-level & run-level & $\\sigma^2_{\\mathrm{scene}}$ \\\\\\midrule']
for a, H in m2['H5_seed_variance'].items():
    v = H['R_PSNR_of_mean_mse']; L.append(f"{AN[a]} & {g(v['s2_seed_scene_level'])} & {g(v['s2_seed_run_level'])} & {g(v['s2_scene'])} \\\\")
L += ['\\midrule', '\\multicolumn{4}{l}{\\textit{Stratified vs.\\ random partitions: variance reduction}}\\\\', 'Model / metric & simulated & observed & reps (rand/strat) \\\\\\midrule']
for a in ['direct', 'unet']:
    A = m2['archs'][a]
    for y in ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse']:
        o = A.get('H6_observed', {}).get(y)
        L.append(f"{AN[a]} / {MN[y]} & {A['metrics'][y]['H6_sim_variance_reduction']:+.2f} & " + (f"{o['reduction']:+.2f} & {o['n_rand_reps']}/{o['n_strat_reps']}" if o else '-- & --') + ' \\\\')
L += ['\\bottomrule\\end{tabular}\\end{table}']
(T / 'tab_controls.tex').write_text('\n'.join(L) + '\n')

# ---- T6 design cost: runs per method -> MDE (absolute and paired) for the headline quantities
mD = m2['archs']['direct']['metrics']; mS = fl['archs']['raft_small']['metrics']
pUD = next(p for p in m2['pairs'] if p['pair'] == 'unet - direct' and p['metric'] == 'simse_R_gray_mean')
pFL = next(p for p in fl['pairs'] if p['metric'] == 'epe_mean')
L = ['\\begin{table*}[!t]\\centering\\caption{What a design costs and what it resolves. Minimum detectable effect (two-sided $\\alpha=0.05$, power 0.8) of a difference between two independently evaluated absolute scores (the listed runs apply to each arm; Eq.~\\eqref{eq:seabs}) and of the two headline paired comparisons, under ten designs (the eight of the ladder and the two that the search of Sec.~\\ref{sec:results3} finds for the intrinsic comparison), ordered by cost, with the number of training runs each requires per method, computed from the estimated components with Eqs.~\\eqref{eq:seabs}, \\eqref{eq:mdepaired} and \\eqref{eq:fullcov}. Paired columns give the MDE and, in parentheses, the actual gap divided by it (a value $\\ge 1$: the design has power $\\ge0.8$ for a true gap of the observed size). The runs column is the cost of evaluating one method once the components are known; the components themselves were estimated from the 75-run design of the last row (Sec.~\\ref{sec:results3}).}\\label{tab:designcost}',
     '\\footnotesize\\setlength{\\tabcolsep}{4pt}\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lrrrrrr@{}}\\toprule',
     ' & & \\multicolumn{3}{c}{MDE of an absolute score} & \\multicolumn{2}{c}{paired MDE (gap/MDE)} \\\\',
     '\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}',
     'Design & runs & \\shortstack[r]{si-MSE\\\\DirectCNN} & \\shortstack[r]{PSNR (dB)\\\\DirectCNN} & \\shortstack[r]{EPE (px)\\\\RAFT-small} & \\shortstack[r]{si-MSE\\\\UNet$-$DirectCNN} & \\shortstack[r]{EPE (px)\\\\RAFT-s$-$RAFT-l} \\\\\\midrule']
SHORT = {'1 partition, 1 seed': '1 partition, 1 seed', '1 partition, 3 seeds (prevailing recipe)': '1 partition, 3 seeds (prevailing)', '5 random partitions, 3 seeds': '5 random partitions, 3 seeds',
         'full coverage, 5 folds, 1 seed': 'full coverage, 1 seed', 'full coverage, 5 folds, 2 seeds': 'full coverage, 2 seeds', 'full coverage, 5 folds, 3 seeds': 'full coverage, 3 seeds',
         'full coverage, 2 repeats, 3 seeds': 'full coverage $\\times$2, 3 seeds', 'full coverage, 5 repeats, 3 seeds (this study)': 'full coverage $\\times$5, 3 seeds (this study)'}
rows = list(D.DESIGNS)
for tgt, lab in [(1.0, 'cheapest with ratio $\\ge1$ (search)'), (1.25, 'cheapest with ratio $\\ge1.25$ (search)')]:
    b = D.cheapest_design(pUD, tgt)
    if b: rows.append((f'full coverage $\\times${b[1]}, {b[2]} seed{"s" if b[2] > 1 else ""}: {lab}', 5 * b[1], b[2], True, b[0]))
rows.sort(key=lambda r: (r[4], not r[3]))
for label, K, S, full, runs in rows:
    a1 = D.mde_abs(mD['simse_R_gray_mean'], K, S, full); a2 = D.mde_abs(mD['R_PSNR_of_mean_mse'], K, S, full); a3 = D.mde_abs(mS['epe_mean'], K, S, full)
    p1 = D.mde_paired(pUD, K, S, full); p2 = D.mde_paired(pFL, K, S, full)
    L.append(f"{SHORT.get(label, label)} & {runs} & {g(a1)} & {g(a2)} & {g(a3)} & {g(p1)} ({abs(pUD['mean_gap'])/p1:.2f}) & {g(p2)} ({abs(pFL['mean_gap'])/p2:.2f}) \\\\")
L += ['\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_designcost.tex').write_text('\n'.join(L) + '\n')

# ---- T7 metric definitions (static: parameters read from experiment_cv2.py / experiment_flow.py, no result numbers)
L = ['\\begin{table*}[!t]\\centering\\caption{Scores used in the paper: definition, colour handling, and scene-level aggregation. $x_f,\\hat x_f$: ground-truth and predicted reflectance of frame $f$ ($256\\times256$); ``gray\'\' is the channel mean; gains $\\alpha$ are least-squares optimal (Sec.~\\ref{sec:gauge}). Flow scores are computed at full resolution over the evaluated pixels (the dataset\'s valid pixels with ground-truth magnitude below 400\\,px).}\\label{tab:metrics}',
     '\\footnotesize\\resizebox{\\textwidth}{!}{\\begin{tabular}{llp{5.2cm}p{4.6cm}}\\toprule', 'Score & Colour & Definition (per frame or pair) & Scene-level score \\\\\\midrule',
     'MSE, PSNR & RGB & $\\mathrm{MSE}_f=\\|x_f-\\hat x_f\\|^2/N$; no gain (absolute gauge) & PSNR of the scene-mean MSE, Eq.~\\eqref{eq:psnr} (the earlier grid\'s convention) \\\\',
     'si-MSE & gray & scalar gain per frame, Eq.~\\eqref{eq:simse} \\citep{grosse2009ground}; invariant to a global scale of $\\hat x_f$ & mean over frames \\\\',
     'LMSE & gray & per-window gain, windows of side $\\mathrm{round}(0.1\\min(h,w))=26$ at stride 13, Eq.~\\eqref{eq:lmse} \\citep{chen2013simple}; one gain fitted inside each window (overlapping windows: not an invariance to an arbitrary spatially varying gain); windows placed without padding from the top-left corner (' + _LMSE_BOUNDARY + ') & mean over frames \\\\',
     'DSSIM & gray & $(1-\\mathrm{SSIM})/2$ on the unaligned prediction, Gaussian window $11\\times11$, $\\sigma=1.5$, $k_1=0.01$, $k_2=0.03$ \\citep{wang2004ssim}; no gain: the structure term of SSIM is scale invariant up to its stabilising constant, the luminance and contrast terms are not; valid convolution without padding, the $' + _SSIM_MAP + '$ map of window scores averaged & mean over frames \\\\',
     'PSNR, scene gain & RGB & one scalar gain pooled over the scene, Eq.~\\eqref{eq:poolgain}; used for the gauge hypothesis & $-10\\log_{10}\\mathrm{MSE}^{(\\alpha)}_u$ \\\\\\addlinespace',
     'EPE & -- & mean end-point error over the evaluated pixels (dataset-valid and ground-truth magnitude $<400$\\,px; Sec.~\\ref{sec:programme}) & mean over the scene\'s pairs (pair-weighted) or over its evaluated pixels (pixel-weighted) \\\\',
     'Fl-all, $>$3\\,px rate & -- & fraction of valid pixels with EPE $>3$\\,px and $>5\\%$ of the flow magnitude (Fl-all) or with EPE $>3$\\,px & mean over pairs \\\\',
     'EPE occ.\\ / non-occ.\\ & -- & EPE restricted to occluded / non-occluded valid pixels & mean over pairs \\\\',
     '\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_metrics.tex').write_text('\n'.join(L) + '\n')

# ---- T8 appendix: REML vs method-of-moments components for every unit-level fit, with the consistency ratio
L = ['\\begin{table*}[!p]\\centering\\caption{Method-of-moments (reported, Sec.~\\ref{sec:estimation}) and multi-start REML (cross-check) estimates of the four variance components for every unit-level fit (killing experiment, main experiment, flow arm). ``Ratio\'\' is the variance of unit means implied by the REML components through the exact design coefficients of \\eqref{eq:momq} divided by the observed one; ``band\'\' marks whether it lies in $[0.80,1.25]$. The moment columns carry the run-level seed correction of Sec.~\\ref{sec:estimation} ($-v_\\zeta/S$ on the train-side component and $-v_e/S$ on the interaction, relative to the cell-level solution of \\eqref{eq:momq}); REML has no run-level term and partitions $v_\\zeta$ between its training-set and residual components. Rows marked $\\dagger$ are the fits at which a single-start REML had stopped at a spurious optimum in the first analysis; all REML columns use the multi-start fit.}\\label{tab:remlmom}',
     '\\scriptsize\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllrrrrrrrrrl}\\toprule', 'Exp. & Model & Metric & \\multicolumn{4}{c}{REML: scene, train, inter, seed} & \\multicolumn{4}{c}{Moments: scene, train, inter, seed} & ratio & band \\\\\\midrule']
PRE = {}
for nm, fn in [('killing', 'killing_v1_pre20261104.json'), ('main', 'main_v2_pre20261104.json'), ('flow', 'flow_v1_pre20261104.json')]:
    if (R / fn).exists():
        d = json.load(open(R / fn))
        for a in d['archs']:
            for y, m in d['archs'][a]['metrics'].items():
                r, mo = m['reml'], m['moment']
                if not r or 'error' in r or not mo.get('s2_scene_raw'): continue
                ratio = (r['s2_scene'] + (r['s2_train'] + r['s2_inter'] + r['s2_seed'] / mo['S']) / mo['R']) / mo['s2_scene_raw']
                PRE[(nm, a, y)] = not (0.8 <= ratio <= 1.25)
for nm, d, keys in [('killing', k1, ['R_PSNR_of_mean_mse', 'simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean']),
                    ('main', m2, ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse', 'R_siPSNR_scenegain_rgb']),
                    ('flow', fl, ['epe_mean', 'epe_pixel_weighted', 'fl_all_mean', 'out3_mean', 'epe_occ_mean', 'epe_noc_mean'])]:
    for a in d['archs']:
        for y in keys:
            m = d['archs'][a]['metrics'].get(y)
            if not m: continue
            r, mo = m['reml'], m['moment']; dag = '$^\\dagger$' if PRE.get((nm, a, y)) else ''
            rr = ' & '.join(g(r[k]) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed']) if r and 'error' not in r else '-- & -- & -- & --'
            mm = ' & '.join(g(mo[k]) for k in ['s2_scene', 's2_train', 's2_inter', 's2_seed'])
            ratio = m.get('reml_consistency_ratio'); used = 'yes' if (ratio is not None and 0.8 <= ratio <= 1.25) else 'no'
            L.append(f"{nm} & {AN[a]} & {MN[y]}{dag} & {rr} & {mm} & {ratio:.2f} & {used} \\\\")
        L.append('\\addlinespace')
L += ['\\bottomrule\\end{tabular}}\\end{table*}']
(T / 'tab_reml_moments.tex').write_text('\n'.join(L) + '\n')
# supplement copy: references to main-text labels prefixed for xr (supplementary.tex), table numbered S1 there
import re as _re
(T / 'tab_reml_moments_supp.tex').write_text(_re.sub(r'\\(ref|eqref)\{(?!tab:remlmom)', lambda m: '\\' + m.group(1) + '{M-', '\n'.join(L)).replace('[!p]', '[!htb]') + '\n')

# ---- T9 family structure of the PSNR interaction (round 9 A1/C1, round 10 B1/C1/C3): family_effect.json + family_joint.json
FEJ, FJJ = R / 'family_effect.json', R / 'family_joint.json'
if FEJ.exists() and FJJ.exists():
    fe = json.load(open(FEJ)); fj = json.load(open(FJJ)); Y = 'R_PSNR_of_mean_mse'
    L = ['\\begin{table*}[!t]\\centering\\caption{The PSNR scene-by-training-set interaction and the bamboo pair (Sec.~\\ref{sec:results1}), per architecture. Three quantities that are not interchangeable are kept apart. '
         '\\emph{Observed design}: the primary moment fit of Table~\\ref{tab:decomp} on the five realised assignments, which held the pair out together twice. '
         '\\emph{Family-adjusted}: the same cells after the jointly fitted training-size and sibling effects are removed (Sec.~\\ref{sec:results3}); a descriptive adjustment, with the fitted sibling coefficient $\\hat b$ (PSNR of a bamboo scene when its sibling is held out with it minus when it is in training; cluster-robust standard error by training set) and 95\\% parametric-bootstrap intervals from the joint model (' + str(fj['bootstrap'][f'direct|{Y}']['n_boot']) + ' replicates). '
         '\\emph{Random designs, structured model}: the primary fit refitted to ' + str(fe['draw_conditionality'][f'direct|{Y}']['n_sim']) + ' simulated five-assignment designs generated from the adjusted exchangeable components plus the deterministic sibling shift (median and 5--95\\% range), and the ratio of the empirical variance of a design\\textquoteright s score under that generator (' + str(fj['structured_absolute']['direct']['n_sim']) + ' designs) to the variance Eqs.~\\eqref{eq:partvar} and \\eqref{eq:fullcov} predict from the primary components, for one partition of five with three seeds and for one full-coverage assignment with three seeds.}\\label{tab:family}',
         '\\resizebox{\\textwidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{l rr r rr rr rr rr}\\toprule',
         ' & \\multicolumn{2}{c}{observed design} & \\multicolumn{3}{c}{family-adjusted (descriptive)} & \\multicolumn{2}{c}{random designs, structured model} & \\multicolumn{2}{c}{Var ratio (empirical/primary)} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}',
         'Model & inter.\\ (dB$^2$) & test share & $\\hat b$ (dB) & inter.\\ (dB$^2$), reduction [95\\%] & test share [95\\%] & inter.\\ median [5--95\\%] & test share median & 1 partition & full cov. \\\\\\midrule']
    for a in ['direct', 'unet', 'transformer', 'proposed']:
        x = fj['intrinsic'][a][Y]; b = fj['bootstrap'][f'{a}|{Y}']; dc = fe['draw_conditionality'][f'{a}|{Y}']; s = fj['structured_absolute'][a]
        L.append(f"{AN[a]} & {g(x['primary']['s2_inter'])} & {x['primary']['test_share']:.2f} & {x['joint_ols']['bam_abs']['est']:+.2f} ({x['joint_ols']['bam_abs']['se_cluster']:.2f}) & "
                 f"{g(x['joint_adjusted']['s2_inter'])}, {100*x['inter_reduction_joint']:.0f}\\% [{100*b['reduction']['ci95'][0]:.0f}, {100*b['reduction']['ci95'][1]:.0f}] & {x['joint_adjusted']['test_share']:.2f} [{b['share_adj']['ci95'][0]:.2f}, {b['share_adj']['ci95'][1]:.2f}] & "
                 f"{g(dc['s2_inter']['median'])} [{g(dc['s2_inter']['p5'])}, {g(dc['s2_inter']['p95'])}] & {dc['test_share']['median']:.2f} & {s['ratio_empirical_over_primary']['one']:.2f} & {s['ratio_empirical_over_primary']['full']:.2f} \\\\")
    L += ['\\bottomrule\\end{tabular}}\\end{table*}']
    (T / 'tab_family.tex').write_text('\n'.join(L) + '\n')

print('tables written:', sorted(p.name for p in T.glob('*.tex')), f'| GPU-h kill {gpu_kill:.1f} main {gpu_main:.1f} flow {gpu_flow:.1f}')
