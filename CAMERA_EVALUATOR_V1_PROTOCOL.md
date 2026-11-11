# DynaMesh4D Camera Evaluator v1 — Frozen Protocol

**Status:** `FROZEN_BEFORE_TEST`  
**Freeze date:** 2026-10-02  
**Leakage verdict:** `NO_TEST_LEAKAGE`  
**Root mapping gate:** `ROOT_MAPPING_RELIABLE`

This protocol was fixed without inspecting formal Camera test results. It must not be changed in response to test performance. It authorizes evaluation only; it does not authorize retraining, checkpoint modification, or test-time calibration.

## 1. Frozen models and permitted ablations

| Variant | Frame-wise adapter | Temporal adapter |
|---|---:|---:|
| Raw | OFF | OFF |
| Frame-wise | ON | OFF |
| Full | ON | ON |

`Temporal-only` is forbidden because the temporal checkpoint was trained on frame-wise-refined features. Across the three variants, Body prediction, frames, GT, identities, body-to-joint mapping, extrinsics, and evaluator must be byte-identical; only the Camera Branch switches may differ.

Frozen checkpoints:

- Frame-wise: `/root/sam-body4d-0.2.0/outputs/dynamesh_v2_camera_phase2/framewise/best.pt`, SHA256 `fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2`.
- Temporal: `/root/sam-body4d-0.2.0/outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt`, SHA256 `e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865`.

## 2. Leakage boundary and formal evaluation universe

Both checkpoints use 3DPW train for optimization and 3DPW validation for model selection. Neither optimization nor checkpoint selection contains an official 3DPW test sequence. The exact sequence lists, manifest hashes, cache counts, and source-code evidence are recorded in `CAMERA_DATA_LEAKAGE_AUDIT.json`.

The held-out formal Camera benchmark is therefore the official-test intersection:

1. `downtown_stairs_00`
2. `flat_guitar_01`
3. `flat_packBags_00`
4. `outdoors_fencing_01`

The pre-existing 22-sequence 3DPW-OC set includes sequences used to train or select the Camera Branch. It remains valid for the already-frozen Body-Branch study, but **must not be presented as a held-out Camera-Branch test benchmark**. If run for diagnosis, it must be labeled non-held-out analysis and excluded from the formal Camera main table.

The formal evaluator requires exact coverage of the frozen official-test person/frame manifest. Missing, unmatched, or duplicate identities are fatal errors; no available-case filtering is allowed.

## 3. Frozen common geometry

### 3.1 Body and joint mapping

1. MHR local mesh: 18,439 vertices.
2. Official barycentric MHR→SMPL mapping: `/root/data1/MHR-main/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz`, SHA256 `4bfe2fd6abb598f38ba3a3aa5506ac45b4efb3cdaed7fe4757cfbbf20bbe6ac7`.
3. SMPL mesh: 6,890 vertices.
4. H36M regressor: `/root/DuoMo-main/data/body_models/J_regressor_h36m.npy`, SHA256 `c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0`.
5. Every nonzero H36M regressor row is normalized to sum to 1 before application.
6. Pelvis/root is normalized H36M joint 0. Spatial joint metrics use J14 indices `[6, 5, 4, 1, 2, 3, 13, 12, 11, 14, 15, 16, 0, 7]`.

No empirical root offset is permitted.

### 3.2 Transform chain

For local point `x_local`, predicted camera translation `t_pred`, and 3DPW world→camera extrinsics `(R_cw, t_cw)`:

```text
x_cam_pred   = x_local + t_pred
x_world_pred = R_cw^T (x_cam_pred - t_cw)
```

The inverse is:

```text
x_cam = R_cw x_world + t_cw
```

All geometry is computed internally in metres. Rotations are proper 3×3 matrices acting on column vectors. Camera rotation is taken from 3DPW GT extrinsics and is identical across ablations. Camera rotation error is not reported because the Camera Branch does not predict rotation.

### 3.3 Identity and time

Identity is the exact ordered tuple `(sequence, person_id, local_frame_index, frame_id)`. Each person track must have strictly ordered, consecutive `local_frame_index`; a discontinuity must be split into a new segment. Evaluation rate is the processed 3DPW rate, fixed at 30 Hz.

## 4. Frozen metric definitions

Let `p_t` and `g_t` be predicted and GT world pelvis positions. Let `J^p_{t,j}` and `J^g_{t,j}` be predicted and GT world H36M-J14 joints. Each definition is applied to one complete contiguous person track, never independently per sliding window.

The sole alignment vector is the complete-track initial-frame pelvis translation:

```text
d = g_0 - p_0
```

No metric performs scale alignment. No metric performs rotation alignment. Only WA-MPJPE and ATE use `d`.

| Metric | Definition per person track | Alignment | Unit |
|---|---|---|---|
| Global Pelvis Error | mean_t `||p_t-g_t||_2` | none | mm |
| W-MPJPE | mean_{t,j} `||J^p_{t,j}-J^g_{t,j}||_2` | none | mm |
| WA-MPJPE | mean_{t,j} `||(J^p_{t,j}+d)-J^g_{t,j}||_2` | one initial-frame pelvis translation for the entire track; no window reset | mm |
| Person-root ATE | `sqrt(mean_t ||(p_t+d)-g_t||_2^2)` | same entire-track initial translation | mm |
| RTE | mean_t `||(p_{t+1}-p_t)-(g_{t+1}-g_t)||_2` | none; translation invariant | mm / processed frame |
| World Velocity Error | mean_t `||v^p_t-v^g_t||_2`, `v_t=(x_{t+1}-x_t)*30` | none | m/s |
| World Acceleration Error | mean_t `||a^p_t-a^g_t||_2`, `a_t=(x_{t+2}-2x_{t+1}+x_t)*30^2` | none | m/s² |
| Trajectory Jitter | mean_t `||p_{t+3}-3p_{t+2}+3p_{t+1}-p_t||_2*30^3` | none; prediction-only diagnostic | m/s³ |
| Velocity magnitude ratio | pooled mean `||v^p||` / pooled mean `||v^g||` | none | unitless |
| Acceleration magnitude ratio | pooled mean `||a^p||` / pooled mean `||a^g||` | none | unitless |
| Motion Magnitude Preservation | `0.5*[min(r_v,1/r_v)+min(r_a,1/r_a)]` | none | unitless, [0,1] |

Thus WA-MPJPE is **sequence/person-track initial-frame translation alignment**, not window-first-frame alignment. It removes only the initial global translation bias and preserves all subsequent drift.

ATE is the RMS global-root trajectory error after that same one-time initial translation. RTE is adjacent-frame displacement error; it performs no explicit alignment and never resets at an inference-window boundary.

## 5. Aggregation and coverage

- Spatial metrics: pool their valid frame×joint or frame errors across the frozen test universe (instance-weighted primary result).
- RTE and velocity: pool valid adjacent-frame steps.
- Acceleration: pool valid three-frame triplets.
- Jitter: pool valid four-frame tuples.
- ATE: compute one RMS per complete person track and macro-average person tracks; also retain per-track values.
- Magnitude ratios: ratio of pooled predicted magnitude to pooled GT magnitude, never an average of per-track ratios.
- Report every sequence separately. As secondary summaries, report sequence-macro means in addition to primary pooled results.
- Report numerator/denominator coverage for every metric and require zero missing, unmatched, and duplicate identities.

## 6. Motion subgroups and over-smoothing safety

Assign each complete person track using GT-only mean pelvis velocity magnitude. Compute the 1/3 and 2/3 quantiles once on the frozen official-test universe: low is at or below the first quantile, mid is above the first and at or below the second, and high is above the second. The thresholds and memberships must be saved before reading ablation metrics and then reused for Raw, Frame-wise, and Full.

For every variant and subgroup, report velocity magnitude ratio, acceleration magnitude ratio, Motion Magnitude Preservation, temporal errors, and jitter. A jitter reduction is not evidence of improvement by itself. Any Full result that reduces temporal error while materially reducing motion ratios or preservation must be reported as an over-smoothing trade-off; it must not be hidden by aggregate jitter.

## 7. Gate tests

The evaluator may run formally only when all tests in `CAMERA_EVALUATOR_V1_TESTS.json` pass:

- barycentric row sums;
- translation equivariance;
- GT world→camera→world round trip;
- prediction camera→world→camera round trip;
- pelvis consistency;
- identity/frame mapping and duplicate rejection;
- synthetic perfect prediction gives approximately zero error;
- fixed translation perturbation produces the analytic result;
- fixed velocity perturbation produces the analytic temporal result.

Current gate result: `ALL_PASS`.

## 8. Prohibited changes

After this freeze, do not use formal test results to alter sequence membership, identity rules, root definition, mapping assets, alignment, aggregation, subgroup thresholds, units, or metric definitions. Do not add an empirical offset. Do not report Camera rotation error. Any necessary future protocol correction requires a new evaluator version and explicit disclosure; v1 results must remain reproducible under this document.
