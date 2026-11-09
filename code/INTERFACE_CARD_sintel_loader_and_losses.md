# Interface card — Sintel loader, targets, training losses, and evaluator fields (benchmark_resolution)

Written 2026-11-09 (series-2 review, round 1, item A1) before `target_consistency_audit.py` was run. Every entry has a source; entries
without one are marked [UNVERIFIED] and no reported number depends on them.

| Item | Value | Source |
|---|---|---|
| Input files | 8-bit RGB PNG, `albedo/`, `clean/`, `final/` per scene; `Image.open(...).convert('RGB')` | `v7_base.py:100–102`; PNG chunks checked with PIL: `info` is empty (no gAMA, no iCCP) for `alley_1/frame_0001.png` in albedo and final → transfer function undocumented |
| Transform | `transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])`, size = 256; Resize on a PIL image = PIL bilinear (torchvision default `InterpolationMode.BILINEAR`); ToTensor = uint8/255 → float32 in [0,1], CHW; no linearisation | `v7_base.py:80–83`; `experiment_cv2.py` passes 256 (docstring line 10 "256 px") |
| Targets | `R = A`; `S = clamp(C/(A+1e-4), 0, 2)/2`; `H = clamp(F−C, 0, 1)`; input `I = F` | `v7_base.py:104–106`; loader yields `(F, R, S, H)` (`cache['F']` is the input: `v7_base.py:113–116`) |
| DirectCNN loss | `mse(Rp,R) + mse(Sp,S) + 0.5·mse(Rp·Sp, I) + 0.02·smooth(Sp)` | `v7_base.py:383–389` (`train_2head`, `smooth_lambda=0.02`) |
| UNet / TransUNet loss | `mse(Rp,R) + mse(Sp,S) + 0.5·mse(Hp,H) + mse(Rp·Sp+Hp, I) + 0.02·smooth(Sp)` | `v7_base.py:410–417` (`train_3head`); TransformerUNet trained with the same function: `experiment_cv2.py:209` |
| Proposed loss | same four terms + 0.02·smooth + heteroscedastic term `nw·0.5·(tlv + res·exp(−tlv))` after warm-up, `nw` ramping to 0.03; warm-up = int(0.7·epochs) | `v7_base.py:446–459`; `experiment_cv2.py:211` (`warmup=int(epochs*0.7)`), epochs = 60 (`experiment_cv2.py:265`) |
| Algebraic consequence | substituting the targets into the reconstruction term gives `R·S + H = A·clamp(C/(A+1e-4),0,2)/2 + clamp(F−C,0,1) ≈ C/2 + (F−C)⁺ ≠ F`: the term is minimised by a decomposition whose product is twice the supervised one | derived from the two rows above; quantified by `target_consistency_audit.py` |
| Evaluator: Recon_MSE | per frame `mean((recon − I)²)` over 3×256×256 values; `recon = Rp·Sp` (direct) or `Rp·Sp+Hp` (others) | `experiment_cv2.py:145–151, 164` |
| Evaluator: gain statistics | per frame `gain_R_rgb_xy = Σ R·Rp`, `gain_R_rgb_xx = Σ Rp²`, `gain_R_rgb_tt = Σ R²`, `n`; per-scene sums `gain_R_rgb_xy_sum`, `gain_R_rgb_xx_sum` in `per_scene`; pooled optimal gain `α_u = xy_sum/xx_sum` (Eq. 24) | `experiment_cv2.py:98–101, 166–168`; per-scene aggregation fields seen in `results_cv_v2/shard0/cv2_direct_randr0_f3_s43_v2.json` (`gain_R_rgb_xy_sum`, `gain_R_rgb_xx_sum`) |
| Evaluator: si-MSE variants | `simse_R_gray = _si_mse(gray(R), gray(Rp))` with `gray = mean over channels`; `simse_R_rgb = _si_mse(R, Rp)` on the RGB tensor; `_si_mse` = per-frame scalar gain `a = Σx·x̂/Σx̂²`, `mean((x − a·x̂)²)` | `experiment_cv2.py:89, 92–95, 154` |
| Per-scene record fields used | `per_scene[scene]['Recon_MSE_mean']`, `['simse_R_rgb_mean']`, `['simse_R_gray_mean']`, `['gain_R_rgb_xy_sum']`, `['gain_R_rgb_xx_sum']`, `['n_frames']` | field names read from an actual record (above) and asserted in the audit script |
| Local data copy | `/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/data/sintel_real/training/{albedo,clean,final}` — 23 scene directories; PNG 1024×436 RGB | `ls` and PIL check 2026-11-09; frame count asserted = 1064 in the audit script |
| Python environment for the audit | `/Users/jihwanw/PhD/AI/JEI_IntrinsicUQ/env/bin/python` — torch 2.11.0, torchvision 0.26.0 (the training runs used torch 2.11.0+cu130: per-run record field `torch`) | checked 2026-11-09 |

Controls run before any number is reported (in `target_consistency_audit.py`):
1. Known-answer control: synthetic `A`, `C = 2·A·s` with `s ∈ [0.1, 0.9]`, `F = C` → `S ≈ s`, `H = 0`; expected `MSE(R·S+H, F) = mean((C/2)²)` and `MSE(2·R·S+H, F) ≈ 0` (both within 1e-6 relative to the stabiliser).
2. Reproduction control: the audit rebuilds the dataset tensors through `v7_base.SintelDataset._get_global_cache(root, 256)` itself (the same code path as training), so no re-implementation of the transform is involved; frame count asserted.
3. Field-name control: every per-scene field is accessed by name with an assertion that it exists in the record.

## Flow arm (added 2026-11-10, series-2 round 3, A1)

| Item | Value | Source |
|---|---|---|
| Ground-truth flow files | Middlebury `.flo`: float32 tag 202021.25 ("PIEH"), int32 W, int32 H, then H·W·2 float32 (u, v interleaved); tag asserted | `experiment_flow.py:36–43` (`read_flo`) |
| Invalid mask | `invalid/<scene>/<frame>.png`, `convert('L') > 0` = invalid; occlusion mask likewise from `occlusions/` | `experiment_flow.py:66–68` |
| Pair list | consecutive clean frames (a, b) with `flow/<scene>/<a stem>.flo` present; 1,041 pairs (asserted) | `experiment_flow.py:46–54`; `flow_mask_audit.py` |
| Evaluated pixels | `valid = ~invalid & (‖gt‖ < MAX_FLOW)`, `MAX_FLOW = 400.0`, applied in `__getitem__` for training and evaluation alike | `experiment_flow.py:31, 87`; `evaluate` uses `valid` for EPE, outlier rates, Fl-all, occ/non-occ (`experiment_flow.py:139–`) |
| Origin of the cutoff | torchvision `references/optical_flow/utils.py`: `sequence_loss(..., max_flow=400)` masks `flow_norm < max_flow` in the **training loss**; `compute_metrics(flow_pred, flow_gt, valid_flow_mask)` uses the dataset mask only | fetched 2026-11-10 from raw.githubusercontent.com/pytorch/vision/main/references/optical_flow/utils.py |
| What the cutoff removes here | 3,664 of 464,424,715 dataset-valid pixels (0.0008%), 5 pairs, 3 scenes (ambush_2 1,589 px = 0.018%; cave_2 2,056 px = 0.009%; ambush_4 19 px); max ‖gt‖ 455.4 px; dataset mask removes 0.074% | `results/benchmark_resolution/flow_mask_audit.json` |
| Local copies used | clean frames: `JEI_IntrinsicUQ/data/sintel_real/training/clean`; flow/invalid/occlusions: extracted from `s3://<bucket>/JEI_IntrinsicUQ/data/sintel_flow_extras.tar` (3.73 GB) into `AI/CV/data/sintel_flow_extras/`; `flow/` deleted after the audit (disk), `invalid/` and `occlusions/` kept | this session |
