#!/usr/bin/env python3
"""stratification_offset.py — expectation of the unweighted systematic fold mean versus the population mean (series-2 review, round 5, B1).

The stratified design (analyze_cv2.simulate_designs, eq. systematic) concatenates the three strata (sizes 8, 8, 7, hard to easy) after
shuffling within each and takes fold k = positions k, k+5, ... Each fold therefore has a fixed stratum composition (fold 1: 2/2/1 of 5,
folds 2-3: 2/1/2, folds 4-5: 1/2/1 of 4), and the uniform average of the five unweighted fold means has expected stratum weights
w = (0.34, 0.36, 0.30) instead of the population weights W = (8, 8, 7)/23. Its expectation is sum_h w_h mu_h, not mu_U, so the
fold-mean variance reductions of Table 10 compare dispersions about slightly different centres. This script computes, from the per-scene
tables of main_v2.json (the same tables simulate_designs uses) and the strata of strata_v2.json: the design weights (derived from the
fold positions, checked against a direct enumeration), the expected offset bias = sum_h w_h mu_h - mu_U per model and metric, the offset
relative to the between-scene SD and to the systematic fold-mean SD, and the mean-squared error about mu_U (variance + bias^2) of the
unweighted systematic fold mean beside the variance of the random one (unbiased) and of the weighted systematic estimator (unbiased:
each fold spans every stratum, so its expectation is sum_h W_h mu_h = mu_U). Variances come from analyze_cv2.simulate_designs itself.
Output: results/benchmark_resolution/stratification_offset.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_cv2 as A

R = Path('/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution')
STRATA = Path(__file__).parent / 'strata_v2.json'
METRICS = ['simse_R_gray_mean', 'lmse_R_chen_mean', 'dssim_R_mean', 'R_PSNR_of_mean_mse']


def design_weights(sizes, K=5):
    """Expected stratum weight of the uniform average of the K unweighted systematic fold means, from the fold positions alone."""
    strat_of_pos = np.concatenate([[h] * n for h, n in enumerate(sizes)]); U = len(strat_of_pos)
    w = np.zeros(len(sizes))
    for k in range(K):
        pos = np.arange(k, U, K); counts = np.bincount(strat_of_pos[pos], minlength=len(sizes))
        w += counts / len(pos) / K
    return w


def main():
    m2 = json.load(open(R / 'main_v2.json')); strata = json.load(open(STRATA))
    strat_of = strata['stratum_of']; sizes = strata['strata_sizes_hard_to_easy']; hs = sorted(set(strat_of.values()))
    assert [sum(1 for s in strat_of if strat_of[s] == h) for h in hs] == sizes
    w = design_weights(sizes); W = np.array(sizes) / sum(sizes)
    # enumeration check of the design weights: simulate the within-stratum shuffles and average the stratum fractions of fold members
    rng = np.random.RandomState(0); acc = np.zeros(len(sizes)); n_it = 2000
    for _ in range(n_it):
        order = []
        for h in hs:
            mem = [s for s in strat_of if strat_of[s] == h]; rng.shuffle(mem); order.extend(mem)
        for f in range(5):
            fold = order[f::5]; acc += np.bincount([strat_of[s] for s in fold], minlength=len(sizes)) / len(fold) / 5
    assert np.allclose(acc / n_it, w, atol=1e-12), (acc / n_it, w)
    out = {'strata_sizes': sizes, 'population_weights': W.tolist(), 'design_weights_unweighted_systematic': w.tolist(), 'models': {}}
    print('population weights', np.round(W, 3), 'design weights', np.round(w, 3))
    for a in ['direct', 'unet']:
        out['models'][a] = {}
        for y in METRICS:
            tab = pd.DataFrame(m2['archs'][a]['metrics'][y]['scene_table']); score = tab.set_index('scene')['score']
            mu_U = float(score.mean()); mu_h = np.array([score[[s for s in score.index if strat_of[s] == h]].mean() for h in hs])
            bias = float(w @ mu_h - mu_U)
            v_rand, v_strat, v_stratw, v_rand5 = A.simulate_designs(tab, strata)
            sd_scene = float(score.std(ddof=1))
            rec = {'mu_U': mu_U, 'stratum_means_hard_to_easy': mu_h.tolist(), 'bias_unweighted_systematic': bias, 'bias_over_between_scene_sd': bias / sd_scene,
                   'bias_over_systematic_fold_sd': bias / np.sqrt(v_strat), 'var_random': v_rand, 'var_systematic_unweighted': v_strat, 'var_systematic_weighted': v_stratw,
                   'mse_systematic_unweighted': v_strat + bias ** 2, 'reduction_variance': 1 - v_strat / v_rand, 'reduction_mse': 1 - (v_strat + bias ** 2) / v_rand,
                   'reduction_weighted': 1 - v_stratw / v_rand, 'bias_sq_over_var_systematic': bias ** 2 / v_strat}
            out['models'][a][y] = rec
            print(f"{a:7s} {y[:10]:10s} bias {bias:+.3g} = {rec['bias_over_between_scene_sd']:+.3f} scene SD = {rec['bias_over_systematic_fold_sd']:+.3f} fold SD | reduction var {rec['reduction_variance']:+.2f} mse {rec['reduction_mse']:+.2f} weighted {rec['reduction_weighted']:+.2f}", flush=True)
    (R / 'stratification_offset.json').write_text(json.dumps(out, indent=1)); print('wrote', R / 'stratification_offset.json')


if __name__ == '__main__':
    main()
