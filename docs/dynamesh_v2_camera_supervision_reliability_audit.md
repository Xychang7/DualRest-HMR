# DynaMesh4D-v2 Camera/Root Supervision & Reliability Audit

## Scope and decision

This is a read-only design audit. It does not change or run the Body branch, geometry evaluator, DuoMo, or any model.

**GT decision: `ONLY_ROOT_TRAJECTORY_GT_AVAILABLE`.** 3DPW provides enough information to construct a strict camera-space SMPL root/pelvis trajectory and camera-space global orientation. It does not provide a label directly equal to SAM's `pred_cam` or `pred_cam_t` parameterization.

## A. 3DPW camera/root ground truth

Observed train annotation schema (person count `P`, frames `T`):

| Field | Shape | Meaning established by current code | Coordinates / units |
|---|---:|---|---|
| `trans` | `(P,T,3)` | Passed as `smplx.SMPL(..., transl=trans)` | SMPL/world coordinates; metres because SMPL vertices are metre-scale |
| `poses` | `(P,T,72)` | `[:3]` global axis-angle, `[3:72]` body pose | global orientation is world-frame before camera transform; radians |
| `cam_poses` | `(T,4,4)` | `V_cam = R_cam @ V_world + t_cam` in `eval_3dpw_standard.py:171-173` | world-to-camera rigid transform; translation in metres |
| `cam_intrinsics` | `(3,3)` | Passed to full-perspective projection | pixel focal lengths/principal point |
| `poses2d` | `(P,T,3,18)` | channel 0/1 are image x/y; channel 2 is validity/confidence (`build_dynamesh_v2_cache_sanity.py`) | original-image pixels + confidence |
| SMPL root | joint 0 under the selected SMPL joint definition | Must be generated from the same SMPL vertices/regressor used for supervision | metres |

Strict construction for frame `t`:

```python
V_world = SMPL(betas, body_pose, global_orient, transl).vertices
J_world = J_regressor @ V_world
J_root_cam = R_cam[t] @ J_world[root] + t_cam[t]
R_global_cam = R_cam[t] @ axis_angle_to_matrix(poses[t, :3])
camera_center_world = -R_cam[t].T @ t_cam[t]
```

`cam_poses` therefore also gives a strict physical camera trajectory, but that trajectory is not the output of `PerspectiveHead`.

SAM camera chain:

1. `sam3d_body.py:470-489`: `pose_token (B,1024)` and zero `init_camera (B,3)`.
2. `camera_head.py:39-52`: frozen head produces `pred_cam=(s,tx,ty)`, shape `(B,3)`.
3. `camera_head.py:70-96`: bbox/intrinsics convert it to `pred_cam_t=(tx+cx,ty+cy,tz)`, with `tz=2f/(bbox_size*s)`.
4. `camera_head.py:99`: `J_pred_cam = J_pred_local + pred_cam_t`.

Thus `pred_cam_t` is the translation placing the zero-global-translation MHR person in camera coordinates. It is neither camera extrinsics nor a weak-perspective camera trajectory. It equals a physical pelvis translation only if the chosen MHR local root is exactly zero.

Strict primary target:

```text
predicted common root = mapped_MHR_root_local + pred_cam_t
GT common root        = R_cam @ SMPL_root_world + t_cam
```

Use the same explicit root definition/mapping on both sides. `trans`, `cam_poses`, and raw `pred_cam_t` must not be compared directly. GT global orientation is constructible, but Camera-v0 does not predict orientation and should not be supervised with it.

## B. Camera reliability

Visibility is an observation cue, not camera reliability: frame 8 had normal bbox/crop/focal/mask yet `pred_cam ty` jumped `0.88 → 2.96 → 0.88`.

Locked Camera-v0 reliability semantics: `r=1` means trust raw SAM camera and suppress temporal correction; `r=0` means raw camera is unreliable and permits the full correction. All soft targets and losses use this direction.

Camera-v0 inference inputs:

1. **Camera temporal innovation:** learned embedding of raw `pred_cam`, first difference, and local bidirectional latent disagreement. An isolated camera-head failure is inconsistent with both neighbours.
2. **Observation stability/quality:** compact vector of visibility/occlusion ratio, normalized bbox centre/size derivatives, and mask-area change. It distinguishes genuine crop/occlusion changes from a camera-only regression spike.
3. **Temporal latent disagreement:** learned forward/backward disagreement available from the adapter without another model.

The current pipeline has no independent inference-time 2D observation. `pred_keypoints_2d` at `sam3d_body.py:2254-2271` is the projection of the same predicted 3D/camera state, while `kps_batch` at `sam3d_body.py:1131-1174` is an evaluation-only prompt path. Consequently reprojection residual is excluded from Camera-v0 reliability inputs. 3DPW `poses2d` remains legal for `L_reproj` and soft-target construction during training only.

## C. Camera Branch v0

```text
raw pose_token (B,T,1024)
  → LN → Linear(1024,128)
  + projected reliability cues
  → two independent forward + two independent backward Mamba/SSM layers
  → LN → zero-init Linear(128,1024) = delta_token
reliability r_t = sigmoid(MLP(camera innovation, observation quality, temporal disagreement))
refined_camera_token = raw_pose_token + (1-r_t) * delta_token
  → frozen PerspectiveHead(raw init_camera)
  → pred_cam → unchanged perspective_projection → pred_cam_t
```

- Inputs: raw token sequence, raw frozen-head camera, observed 2D/bbox/mask metadata, visibility; all aligned by person and global frame ID.
- Outputs: `(B,T,1024)` refined camera token and scalar reliability `(B,T,1)`.
- Identity: output projection weight/bias are zero; initialization exactly reproduces raw token regardless of gate.
- Frozen: backbone, prompt decoder, MHRHead, PerspectiveHead, Body adapter, camera projection.
- Trainable: Camera adapter plus reliability MLP only; expected roughly 0.7–1.0M parameters, asserted exactly before training.
- Bidirectional context requires future frames. Camera-v0 is offline-only; streaming would require a separately validated causal variant.

## D. Camera-v0 loss

```text
L_camera = 1.0 L_root3d + λ2d L_reproj + λrel L_reliability + λanchor L_anchor
```

1. `L_root3d`: robust metre-space loss between predicted and GT common camera-space roots. Uses physical GT, differentiable through frozen head/projection, and is the primary supervision.
2. `L_reproj`: confidence-weighted robust 2D joint reprojection in original pixels. Uses GT `poses2d`, is differentiable, and constrains crop/focal ambiguity.
3. `L_reliability`: supervise `r_t` from a soft target derived from normalized raw-SAM GT root/reprojection error, never a fixed jump threshold. Uses GT and makes the gate identifiable.
4. `L_anchor`: on high-GT-reliability/stable clean frames, keep refined `pred_cam` close to detached raw SAM `pred_cam`. It is auxiliary because raw SAM can itself fail.

No global acceleration-minimization term in v0. Sequence-level physical root GT already supervises temporal behaviour without suppressing real camera/person acceleration. A reliability-weighted temporal residual may be considered only after v0 ablation.

## E. Phase-0 and Phase-1

### Phase-0 — Identity equivalence

- Recover full ordered `(T,1024)` camera sequences and all projection metadata.
- Assert refined token equals raw token exactly at initialization.
- Assert `pred_cam`, `pred_cam_t`, projected 2D keypoints, and frame/person alignment equal baseline within strict float tolerance.
- Test visible, occluded, multi-person, and discontinuous tracks; never bridge discontinuities.

### Phase-1 — Sanity/micro-overfit

- Frame 7/8/9 is a diagnostic: the learned gate should lower reliability at frame 8, restore `ty` toward temporally/geometrically consistent values, and preserve frames 7/9.
- It may be trained only if it belongs to an allowed 3DPW train track. Otherwise it remains diagnostic-only.
- Train on a few 3DPW train person tracks with real and deterministic image-space occlusions; no test frames.
- Success: root/reprojection error decreases on known bad frames, camera spike is removed, stable-frame `pred_cam_t` changes negligibly, clean teacher is preserved, and gate correlates with raw physical error without hard thresholds.

## F. Relationship to Body branch

```text
shared raw pose_token + frame/person ordering
├─ Body: occlusion-conditioned latent restoration → frozen MHRHead → body articulation only
└─ Camera: reliability-aware latent restoration → frozen PerspectiveHead → camera/root translation only
```

They share the frozen visual representation and metadata alignment, but not temporal parameters, gates, checkpoints, losses, or output replacement. The separation is necessary because body corruption tracks visual occlusion, while camera regression can fail despite normal visibility and crop evidence.

## Risks and readiness

1. MHR and SMPL pelvis/root definitions must be explicitly mapped; otherwise `pred_cam_t` receives a constant or pose-dependent offset error.
2. 2D reprojection and bbox-conditioned camera scale have ambiguity; root GT must remain primary.
3. Bidirectional recovery can over-correct genuine fast camera/person motion; reliability supervision and clean anchoring must be validated per sequence.

**Status: `READY_FOR_CAMERA_PHASE0`**, conditional on freezing one explicit common-root mapping in the Phase-0 tests.
