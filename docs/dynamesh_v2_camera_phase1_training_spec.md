# DynaMesh4D-v2 Camera Branch Phase-1 Specification

## Scope and frozen state

Phase-1 validates whether the existing `CameraFeatureTemporalAdapter` can repair a real train-split camera-regression failure by micro-overfitting one strictly aligned temporal window. The Body Phase-2 checkpoint, Body adapter, MHRHead, PerspectiveHead, SAM backbone/decoder, DuoMo, and evaluators remain frozen. The only optimizer-owned parameters are those of `CameraFeatureTemporalAdapter`.

The frozen Body checkpoint is `/root/sam-body4d-0.2.0/outputs/dynamesh_v2_phase2_training/best.pt`; its SHA256 and the Phase-2 validation metrics SHA256 must be recorded before training. Camera Phase-1 checkpoints use a separate output directory.

## Data and deterministic equivalence gate

Failure mining is restricted to the 3DPW `train` split and to the clean condition. Validation and test inputs are hard rejected. Mining uses cached clean `pose_token`, its exactly aligned clean `mhr_init_pose`, and frozen MHR/Perspective heads. Original 3DPW annotations may supply `poses2d`, confidence, intrinsics, image size, and bbox inputs only through strict `sequence + person_id + frame_id` matching.

Before mining, a deterministic equivalence gate must prove that annotation-derived bbox, crop, camera intrinsics, projection coordinates, and frame mapping reproduce the formal pipeline contract. The gate compares reconstructed metadata/projection against an existing formal-pipeline record with explicit tolerances. If strict equivalence cannot be established, no approximate metadata is accepted: only a minimal Camera sanity cache may be generated for a few candidate train windows through the formal pipeline.

The four cache conditions must retain their own variant-specific token and init state. Clean init must never assist an occluded token unless the formal inference path itself does so.

## Clean-only failure mining

The first mining pass defines real camera failures from clean train frames only. Candidate scoring uses:

- raw `pred_cam` temporal innovation around a `normal -> abnormal -> normal` triplet;
- confidence-masked 12-joint bbox-normalized reprojection error;
- body-articulation error proxy, preferring low body error;
- stable visibility, bbox centre/scale, and mask-area observations.

No common-root error, fitted root offset, empirical scale, validation frame, or test frame may enter mining. The output contains ranked candidates and all diagnostic fields. One clear train window is selected for training, with at most two additional diagnostic candidates.

## Adapter and inference contract

The Phase-0 architecture is unchanged:

`raw pose_token -> LN -> 1024->128 -> reliability-conditioned bidirectional Mamba x2 -> LN -> 128->1024 -> gated residual -> frozen PerspectiveHead`.

The locked equations are:

`refined = raw + (1 - reliability) * delta`, where `reliability=1` trusts raw SAM and suppresses correction, and `reliability=0` permits full temporal correction.

Inference reliability inputs are limited to temporal latent disagreement/camera innovation, visibility or occlusion, and bbox/mask stability. GT 2D joints and GT reprojection residual are forbidden as inference inputs.

## Loss

No common-root loss is allowed. Phase-1 uses exactly four terms:

`L_camera = lambda_teacher*L_teacher + lambda_reproj*L_reproj + lambda_rel*L_reliability + lambda_anchor*L_anchor`.

- `L_teacher`: robust loss between refined synthetic-occluded `pred_cam` and the detached same-frame clean-token frozen-camera prediction. It is weighted by detached clean-camera reliability/stability; frames with poor clean reprojection receive weak or zero teacher weight.
- `L_reproj`: differentiable confidence-masked robust reprojection loss using the strict 12 MHR70/OpenPose correspondences, full-image intrinsics, and bbox normalization. This is the primary geometric supervision on frames whose clean camera is unreliable.
- `L_reliability`: calibrated soft-target loss. Training-only raw-SAM GT reprojection error maps monotonically to `1=raw reliable` and `0=raw unreliable`; the GT error is a target only, never an inference feature.
- `L_anchor`: high-reliability/stable-frame robust loss from refined `pred_cam` to detached raw SAM `pred_cam`, protecting normal and clean frames.

A forward-only loss-scale probe reports all four raw magnitudes before optimization. One fixed lambda set is chosen to avoid order-of-magnitude domination; at most one necessary scale correction is permitted.

## Micro-overfit and evaluation

Training uses only the selected train window, fixed seed, AdamW, approximately 400 bounded steps, and balanced clean/15/35/55 conditions. A real clean failure remains supervised. Best and final Camera-only checkpoints are saved separately.

Evaluation compares frozen SAM Camera with trained Camera-v0 at the failure centre, its normal neighbours, the full window, and all four conditions. It reports normalized reprojection, `pred_cam`, `pred_cam_t`, temporal innovation, predicted reliability, stable-frame anchor deviation, convergence, and whether clean/normal frames are preserved.

The verdict is exactly one of `CAMERA_TEMPORAL_ADAPTER_LEARNABLE`, `CAMERA_RECOVERY_WEAK`, `RELIABILITY_NOT_ALIGNED`, or `CAMERA_ARCHITECTURE_OR_LOSS_NEEDS_REDESIGN`. Phase-2 full Camera training is out of scope.
