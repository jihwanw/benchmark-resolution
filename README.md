# Benchmark Resolution: What a Few-Scene Benchmark Can and Cannot Decide — code, records, and generators

Author: Jihwan Woo (Amazon Web Services, Seoul). Companion repository to the manuscript "Benchmark Resolution: What a Few-Scene
Benchmark Can and Cannot Decide" (International Journal of Computer Vision, in preparation / under review). Repository: https://github.com/jihwanw/benchmark-resolution

Every number, table, and figure in the manuscript is generated from the files in this repository by the scripts in `paper/`.

## Contents
- `code/` — analysis (`analyze_cv.py` killing experiment, `analyze_cv2.py` main experiment, `analyze_flow.py` flow arm;
  `bootstrap_components.py` intervals; `validate_*.py`, `calibrate_diagnostics.py`, `compare_q3_forms.py` estimator and power
  validation; `trainsize_check.py`, `trainsize_adjust.py`, `family_effect.py`, `family_joint.py` the training-size and sibling-family
  analyses; `target_consistency_audit.py` the reconstruction-loss/target audit and gain decomposition, `rgb_simse_control.py` the
  RGB single-gain si-MSE control, `flow_mask_audit.py` the 400-px cutoff audit, `stratification_offset.py` the systematic-fold-mean
  offset, `composition_distinctness_check.py` the distinct-composition check of Proposition 1; `INTERFACE_CARD_sintel_loader_and_losses.md`
  records, with file-and-line sources, the input/target encodings, losses, and flow mask conventions these analyses rely on), training and evaluation (`v7_base.py`, `experiment_v8.py`, `experiment_cv_v1_killing.py`, `experiment_cv2.py`,
  `experiment_flow.py`), qualitative renders (`qual_intrinsic.py`, `qual_flow.py`, `qual_display_check.py`), strata (`make_strata.py`,
  `strata_v2.json`), summaries (`make_summary_*.py`), and this assembly script (`make_release.py`).
  `code/history/` holds the analysis versions replaced during review (single-start REML; equal-fold moment forms; unconditional Q3).
  `code/launch/` holds the EC2 launch scripts and instance records (bucket name redacted).
- `results/` — aggregates (`killing_v1.json`, `main_v2.json`, `flow_v1.json`), validation outputs (`estimator_validation.json`,
  `estimator_misspec.json`, `estimator_q3_comparison.json`, `calibration_diagnostics.json`, `power_validation.json`,
  `pilot_power_validation.json`, `r2_precision.json`, `bootstrap_ci.json`, `trainsize_check.json`, `trainsize_adjust.json`,
  `family_effect.json`, `family_joint.json`, `target_consistency_audit.json`, `rgb_simse_control.json`, `flow_mask_audit.json`,
  `stratification_offset.json`, `composition_distinctness.json`), the pre-2026-11-04 single-start-REML results and the review-round archives, and the
  per-run records: `results_cv_v1/` (killing experiment, 100 runs + parity), `results_cv_v2/` (main experiment, 340 runs),
  `results_flow_v1/` (flow arm, 150 runs + 2 zero-shot). Each per-run JSON carries the architecture, design, repeat, fold, fold seed,
  model seed, the held-out and training scene lists, and per-scene and per-frame scores; `cv_folds_v2.json` / the flow fold file are the
  shared fold assignments (seeds 201–205). `results/qual/*.json` are the metadata of the display runs.
- `paper/` — `make_numbers.py` (→ `numbers.tex`, every macro the manuscript cites), `make_tables.py` (→ `tables/*.tex`;
  `tab_resolution.tex` is the paired-comparison block of the main text and `tab_resolution_abs.tex` the absolute-score ladder of the
  supplement), `make_figs.py`, `make_fig1.py`, `make_qual_figs.py` (→ `figs/`; `fig7_gauge_top` and `fig10_interaction_ctx` are the
  compact main-text versions of the full-page `fig7_gauge`, `fig8_interaction` and `fig9_flow_interaction`, which are in the
  supplement), `design_mde.py` (MDE and design-search functions);
  `external/numbers_josaa.tex` carries the observed partition and seed variances of the earlier study that the killing experiment
  tests against (its repository: https://github.com/jihwanw/sintel-scene-partition-variance).
- `prereg/` — the pre-registration text of each stage, verbatim (`PREREGISTRATION_ko.md`) and in English translation
  (`PREREGISTRATION_en.md`), and `TIMELINE.md`, which dates each stage against the launch records.

## Not included, and why
The qualitative render arrays (`results/qual/qual_intrinsic.npz`, `qual_flow.npz`, 190 MB) contain frames of MPI-Sintel (Butler et al.,
ECCV 2012; renders of the Blender Foundation's open film *Sintel*, CC BY 3.0) and are not ours to redistribute; `code/qual_intrinsic.py`
and `code/qual_flow.py` regenerate them deterministically (seed 42, compositions listed in `results/qual/*_meta.json`). Run logs and the
manuscript's LaTeX source are not included.

## Reproducing the numbers
The scripts read and write under two root paths set as constants at the top of each file
(`/Users/jihwanw/PhD/AI/CV/results/benchmark_resolution` and `/Users/jihwanw/PhD/AI/CV/paper_ijcv`); point them at your checkout's
`results/` and `paper/`. Then
```
pip install -r requirements.txt
python3 code/analyze_cv.py        # killing experiment  -> results/killing_v1.json
python3 code/analyze_cv2.py       # main experiment     -> results/main_v2.json
python3 code/analyze_flow.py      # flow arm            -> results/flow_v1.json
python3 code/bootstrap_components.py; python3 code/validate_estimators.py; python3 code/validate_misspec.py
python3 code/calibrate_diagnostics.py; python3 code/validate_power.py; python3 code/validate_pilot_power.py --n-pilot 200 --n-eval 1000
python3 code/validate_r2_precision.py; python3 code/trainsize_check.py; python3 code/trainsize_adjust.py
python3 code/family_effect.py; python3 code/family_joint.py
python3 code/target_consistency_audit.py; python3 code/rgb_simse_control.py; python3 code/stratification_offset.py
python3 code/composition_distinctness_check.py   # flow_mask_audit.py needs the MPI-Sintel flow extras (invalid/occlusion masks)
python3 paper/make_numbers.py; python3 paper/make_tables.py; python3 paper/make_figs.py; python3 paper/make_fig1.py
```
Training runs (`experiment_cv2.py`, `experiment_flow.py`) were executed on NVIDIA T4 GPUs (g4dn.xlarge) from the launch scripts in
`code/launch/`; MPI-Sintel training data (`training_images.zip`, `MPI-Sintel-training_extras.zip`) must be downloaded from the dataset's
site.

## License
Code: MIT (see `LICENSE`). Result records are released under the same terms; they contain scores only, no dataset images.

## Citation
J. Woo, "Benchmark Resolution: What a Few-Scene Benchmark Can and Cannot Decide," code and result records, https://github.com/jihwanw/benchmark-resolution (2026).
A DOI for the archived release will be added on acceptance.
