# 3DPW-OC unified evaluator

This package implements the locked Phase-2B protocol without depending on a
particular HMR model.

## Canonical contracts

- Key: `(sequence, rgb_frame_idx, source_actor_id)`.
- `rgb_frame_idx` is the continuous 30-fps RGB index, not the 60-Hz mocap sync
  id stored in `sequenceFiles.img_frame_ids`.
- Prediction vertices are standard SMPL `(6890, 3)` in **meters**.
- GT comes from each `3DPW_oc.json` annotation's pose72, shape10, trans3 and
  gender, decoded by the corresponding gendered standard SMPL model.
- MPJPE/PA-MPJPE use the official, separately supplied
  `J_regressor_h36m_correct.npy`, H36M pelvis 0 centering, and J14 indices
  `(1,2,3,4,5,6,8,10,11,12,13,14,15,16)`.
- PA-MPJPE uses full scale/rotation/translation similarity alignment.
- MPVPE centers each mesh by native SMPL joint 0, not the H36M pelvis.
- Acceleration uses a deliberately separate temporal GT path: all three frames
  come from the same `sequenceFiles` source actor, are converted to camera-axis
  SMPL geometry, independently pelvis-centred as H36M J14, and must have mocap
  sync deltas `2,2` (the continuous 30-fps RGB interval).  The second
  difference is multiplied by `30**2` and reported in `m/s^2`.  Only the center
  key must be in the OC manifest; non-OC neighbours are allowed.  Absolute
  sequenceFiles translation is never used to replace OC-JSON spatial GT.

`evaluate_from_oc_json` is the fail-closed public entry point.  It never falls
back from `J_regressor_h36m_correct.npy` to the legacy
`J_regressor_h36m.npy`.

## Differences from the legacy Dyna evaluators

| Contract | This evaluator | Legacy evaluators |
|---|---|---|
| GT source | OC JSON gendered SMPL | sequenceFiles / prebuilt records |
| H36M resource | `J_regressor_h36m_correct.npy` | legacy `J_regressor_h36m.npy` |
| Joint subset | JOTR J14 above | reordered H36M J14 `(6,5,4,1,2,3,...)` |
| MPVPE pelvis | native SMPL joint 0 | absent or evaluator-dependent |
| Filtering | 20,243 OC person instances | ordinary 3DPW split/records |
| Acceleration | continuous RGB 30 fps, OC centers | processed-frame/window protocol |

The corrected and legacy regressors are distinct resources even though both
have shape `(17, 6890)`; shape equality is not provenance equivalence.
