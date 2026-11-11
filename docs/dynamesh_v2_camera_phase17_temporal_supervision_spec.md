# DynaMesh4D-v2 Camera Phase-1.7 Temporal Supervision Audit

## Probe-1

Train the unchanged CameraFeatureTemporalAdapter on only the fixed clean train window `courtyard_basketball_00/person1`, frames `506,508,...,536`. The correction gate is fixed open and the only loss is the validated 12-joint normalized reprojection loss. Start from the seeded identity initialization, use AdamW at `1e-3`, run 400 steps, and extend once to 800 only if the final 100-step clean loss is still decreasing materially. The 15/35/55 variants are evaluation-only.

## Probe-2

Reuse the existing aligned 50-frame `flat_guitar_01/person0` data and original 3DPW annotation. Do not run SAM or Meta Conversion. The primary physical candidate compares the MHR/SAM model origin placement (`pred_cam_t`) with the 3DPW SMPL model-origin placement (`R_cam @ trans + t_cam`). Diagnostic candidates compare the existing MHR hip midpoint and Meta-converted SMPL H36M/native pelvis roots. Report absolute, first-difference, and second-difference statistics without fitted offset or scale.

## Decisions

Probe-1 is capacity-confirmed only if both full-window and failure-frame reprojection improve materially beyond Phase-1.6. Relative supervision is validated only when its physical definition is compatible, differencing substantially reduces ambiguity, direction/magnitude correlations are high, and error is small relative to true motion. Otherwise it is auxiliary or unreliable. No Camera full training or Body modification follows automatically.
