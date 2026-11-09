"""design_mde.py — minimum detectable effect of a design from estimated components (shared by make_tables.py and make_numbers.py).
Formulas are those of analyze_cv2.py / Sec. 3.3-3.4 (Eqs. seabs, mdepaired, fullcov):
  absolute, random partitions : MDE = z sqrt(2 [ (V_test + s2_train + s2_inter/n)/K + (v_zeta + v_eps/n)/(K S) ])
  absolute, full coverage     : MDE = z sqrt(2 [ (c_T s2_train + s2_inter/U)/R + (c_T v_zeta + v_eps/U)/(R S) ]),  K = 5 R folds scored
  paired,   random partitions : MDE = z sqrt(   [ (1-n/U) S2_dg/n + s2_train_d + s2_inter_d/n ]/K + (v_zeta_d + v_eps_d/n)/(K S) )
  paired,   full coverage     : MDE = z sqrt(   [ c_T s2_train_d + s2_inter_d/U ]/R + (c_T v_zeta_d + v_eps_d/U)/(R S) )
  (round 4, A1/A3: seed noise = run-level shift v_zeta + independent part v_eps, fitted by analyze_cv2.moment_components)
with U = 23, n = 5, z = z_{0.975} + z_{0.8}. 'runs' is the number of training runs per method.
Full coverage uses the exact coefficients of the unit-weighted mean over the folds of 5, 5, 5, 4, 4 units (review round 2, C2):
  Var(ybar_A) = [ c_T s2_train + s2_inter / U + c_T s2_run / S ] / R,  c_T = sum_k (n_k/U)^2 = 107/529,  R = number of repeats."""
import math
U, N, Z = 23, 5, 1.959964 + 0.841621
FOLDS = [5, 5, 5, 4, 4]; C_T = sum((n / U) ** 2 for n in FOLDS)
DESIGNS = [  # (label, K, S, fullcov, runs per method)
    ('1 partition, 1 seed', 1, 1, False, 1),
    ('1 partition, 3 seeds (prevailing recipe)', 1, 3, False, 3),
    ('5 random partitions, 3 seeds', 5, 3, False, 15),
    ('full coverage, 5 folds, 1 seed', 5, 1, True, 5),
    ('full coverage, 5 folds, 2 seeds', 5, 2, True, 10),
    ('full coverage, 5 folds, 3 seeds', 5, 3, True, 15),
    ('full coverage, 2 repeats, 3 seeds', 10, 3, True, 30),
    ('full coverage, 5 repeats, 3 seeds (this study)', 25, 3, True, 75),
]

def picked(m): return m['reml'] if m['used'] == 'reml' else m['moment']

def seed_var(vz, ve, K, S, fullcov):
    """Seed term (round 4, A1/A3): seed noise = run-level shift (v_zeta, shared by the units of a run) + independent part (v_eps).
    Random K partitions of N units: (v_zeta + v_eps/N)/(K S). Full coverage, K = 5R folds: (C_T v_zeta + v_eps/U)/(R S), exact for unequal folds."""
    if fullcov: return (C_T * vz + ve / U) / ((K / len(FOLDS)) * S)
    return (vz + ve / N) / (K * S)

def var_design(s2_scene, s2_train, s2_inter, vz, ve, K, S, fullcov):
    """Variance of the design's mean score. Random: K independent n-of-U partitions. Full coverage: K = 5R folds (R repeats)."""
    if fullcov:
        R = K / len(FOLDS)
        return (C_T * s2_train + s2_inter / U) / R + seed_var(vz, ve, K, S, True)
    v_test = (1 - N / U) * s2_scene / N
    return (v_test + s2_train + s2_inter / N) / K + seed_var(vz, ve, K, S, False)

def mde_abs(m, K, S, fullcov):
    c = picked(m)
    return Z * math.sqrt(2 * var_design(c['s2_scene'], c['s2_train'], c['s2_inter'], m['v_zeta'], m['v_eps'], K, S, fullcov))

def mde_paired(p, K, S, fullcov):
    return Z * math.sqrt(var_design(p['s2_scene_x_method'], p['s2_train_diff'], p['s2_inter_diff'], p['v_zeta_diff'], p['v_eps_diff'], K, S, fullcov))


def var_fullcov_corr(s2_train, s2_inter, vz, ve, S, rho_delta, rho_eta, R=1):
    """Full-coverage variance under within-assignment correlations (Sec. 3.4, eq. fullcovgen): a correlation rho_delta between the
    training-set effects of different folds adds (1 - C_T) rho_delta s2_train; a correlation rho_eta between the interaction cells of
    the units of one fold adds (C_T - 1/U) rho_eta s2_inter. rho = 0 recovers var_design(..., fullcov=True)."""
    return (C_T * s2_train + (1 - C_T) * rho_delta * s2_train + s2_inter / U + (C_T - 1 / U) * rho_eta * s2_inter) / R + seed_var(vz, ve, 5 * R, S, True)

def fullcov_mde_factor(c_train, c_inter, vz, ve, S, rho):
    """Factor by which the full-coverage MDE grows when the training-set effects of different folds are correlated at rho_delta = rho.
    A within-fold correlation of the interaction cells (rho_eta) is not varied here: at the cell level it is indistinguishable from a
    training-set effect and is absorbed by the fitted s2_train, so eq. fullcov applied to the fitted components already includes it
    (validate_misspec.py, scenario 'corr'). The MDE scales with the square root of the variance."""
    return math.sqrt(var_fullcov_corr(c_train, c_inter, vz, ve, S, rho, 0.0) / var_fullcov_corr(c_train, c_inter, vz, ve, S, 0.0, 0.0))


def cheapest_design(p, target, Rmax=12, Smax=6):
    """Cheapest full-coverage design (R repeats x S seeds, 5RS runs per method) whose plug-in gap/MDE reaches target (round 4, C1 / round 5, C2).
    Returns (runs, R, S, ratio) or None."""
    best = None
    for Rr in range(1, Rmax + 1):
        for S in range(1, Smax + 1):
            ratio = abs(p['mean_gap']) / mde_paired(p, 5 * Rr, S, True); cost = 5 * Rr * S
            if ratio >= target and (best is None or cost < best[0] or (cost == best[0] and ratio > best[3])): best = (cost, Rr, S, ratio)
    return best
