# DynaMesh4D-v2 Camera Phase-1.8 Gate

This Gate tests whether strict common limb joints differ mainly by one camera
translation. It never fits rotation, scale, an offset, or a per-sequence
transform.

## Joint contract

The ordered joints are left/right shoulder, elbow, wrist, hip, knee, ankle.
The MHR side uses official MHR70 indices `5,6,7,8,62,41,9,10,11,12,13,14`.
The GT side uses native SMPL joints `16,17,18,19,20,21,1,2,4,5,7,8`.
Both arrays are expressed in metres. GT SMPL joints are transformed from world
to camera coordinates by the cached 3DPW `cam_poses` (`R @ joint + t`). MHR
joints remain in the local coordinate contract consumed by the frozen
PerspectiveHead; adding `pred_cam_t` places them in camera coordinates.

3DPW OpenPose-18 confidence is used only as a validity/weight mask after an
independent semantic audit. Its 2D coordinates are not used as 3D targets.

## Target and diagnostics

For valid weights `w_j`, `t = sum(w_j * (Y_j-X_j)) / sum(w_j)`. The same closed
form is evaluated for all 12, upper 6, and lower 6 joints. Diagnostics include
translation-only residuals, upper/lower disagreement, raw `pred_cam_t` error,
depth error, robust median/MAD estimates, body-teacher geodesic, temporal body
change, bbox scale, and 2D reprojection.

The full train split is eligible because cache-v2 contains aligned tokens,
variant-specific init state, GT SMPL parameters, camera extrinsics, and frame
metadata. Validation and test are rejected.

No GT SMPL-vs-MHR body-rotation error is reported: the articulation spaces are
not semantically one-to-one and using raw parameter alignment would violate the
Gate. The valid body diagnostic is each condition's legal MHR rotation relative
to its clean frozen-MHR teacher.

## Verdict policy

- `MULTI_JOINT_CAMERA_GT_VALIDATED`: low residual and regional disagreement,
  stable finite depth, failure-frame detection, and weak pose dependence.
- `MULTI_JOINT_CAMERA_GT_USABLE_WITH_RELIABILITY`: the low-residual population
  is usable, but some pose/model mismatch requires residual-based weighting.
- `MULTI_JOINT_CAMERA_GT_NOT_RELIABLE`: regional targets disagree, depth is
  unstable, or body mismatch cannot be separated.

If usable, the future-only proposal is
`w_target * SmoothL1(pred_cam_t, stopgrad(t_all))`, with `w_target` monotonically
decreasing with translation-fit residual. This Gate performs no training.
