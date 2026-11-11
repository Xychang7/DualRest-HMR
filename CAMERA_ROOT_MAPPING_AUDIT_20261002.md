# DynaMesh4D Camera Root Mapping Audit — 2026-10-02

## Final decision

**`ROOT_MAPPING_RELIABLE`**

**`COMMON_GEOMETRY_MAPPING_AVAILABLE`**

The earlier failure came from treating the MHR hip midpoint, MHR coordinate origin, native SMPL pelvis, and H36M pelvis as interchangeable. They are not interchangeable, and no fixed offset between them is legal. The resolved protocol instead constructs the same H36M pelvis definition on both prediction and GT geometry.

No model was trained or modified. Camera checkpoints and the frozen Camera-space smoke remain unchanged. The correspondence was defined and validated only on 3DPW train/validation data; no test sequence or test metric was used to select it.

## Frozen Camera artifacts

- Frame-wise checkpoint: `outputs/dynamesh_v2_camera_phase2/framewise/best.pt`
  - SHA256: `fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2`
- Temporal checkpoint: `outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt`
  - SHA256: `e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865`
- Frozen Camera-space result: `CAMERA_SPACE_SMOKE_20261002.json`

## Dataset split audit

- `courtyard_arguing_00`: train, persons 0 and 1
- `courtyard_relaxOnBench_00`: train, person 0
- `courtyard_jumpBench_01`: validation, person 0

The correspondence Gate used one deterministic central 16-frame window per track, totaling 4 tracks and 256 frame-condition records. `test_split_used=false`.

## MHR local origin and camera translation

`MHRHead.forward` forces `global_trans` to zero before evaluating the MHR model. Thus the local origin is the MHR rig/model coordinate `[0,0,0]`; the available source does not define it as an anatomical pelvis token. After the SAM camera-axis conversion, the camera head adds the same `pred_cam_t` to every local point:

```text
X_camera^MHR = X_local^MHR + pred_cam_t
```

Therefore `pred_cam_t` is a uniform translation of the MHR local model coordinate system, not a pelvis coordinate by itself.

Source evidence:

- `models/sam_3d_body/sam_3d_body/models/heads/mhr_head.py:290-343`
- `models/sam_3d_body/sam_3d_body/models/heads/camera_head.py:78-103`
- `/root/data1/MHR-main/tools/mhr_smpl_conversion/conversion.py:450-487`

## Unique common-geometry pelvis

The official MHR conversion asset defines every one of the 6890 SMPL-topology vertices as barycentric coordinates on one MHR triangle:

```text
V_pred^SMPL6890 = BaryMap_MHR→SMPL(V_pred^MHR18439)
P_pred_local = J_h36m_normalized[0] · V_pred^SMPL6890
P_gt_world   = J_h36m_normalized[0] · V_gt^SMPL6890
```

Prediction and GT therefore use the same surface topology and the same linear anatomical pelvis definition. This mapping is direct geometry interpolation; it does not fit SMPL parameters and does not use evaluation errors.

Assets:

- `/root/data1/MHR-main/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz`
  - shape: `triangle_ids[6890]`, `baryc_coords[6890,3]`
  - SHA256: `4bfe2fd6abb598f38ba3a3aa5506ac45b4efb3cdaed7fe4757cfbbf20bbe6ac7`
- `/root/DuoMo-main/data/body_models/J_regressor_h36m.npy`
  - shape: `[17,6890]`
  - SHA256: `c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0`

The released H36M regressor rows are numerically close to, but not exactly, affine: pelvis-row sum `0.9999127691`, maximum row-sum deviation `3.6661e-4`. Each row is deterministically divided by its sum before world-root evaluation. This changes no relative vertex weights, uses no data-derived tuning, and restores the required property that an anatomical point follows a uniform translation exactly. Maximum absolute individual weight change is `1.9522e-4`.

## Pelvis relationships and offset audit

Across 256 train/validation frame-condition records:

- MHR origin → common pelvis norm: mean `0.96920 m`, std `0.02818 m`
- MHR hip midpoint → common pelvis: mean `0.07775 m`, std `0.00301 m`, max `0.08703 m`
- 3DPW `transl` → native SMPL joint 0: mean `0.23278 m`
- native SMPL joint 0 → normalized H36M pelvis: mean `0.03619 m`, max `0.04577 m`
- hip/common offset magnitude vs pose magnitude correlation: `-0.45587`

Consequences:

1. MHR origin, MHR hip midpoint, SMPL native pelvis, H36M pelvis, and 3DPW `transl` remain distinct.
2. Their offset is not a legal universal constant; it varies with pose/orientation and can also vary with shape/person.
3. The final mapping does not estimate or apply any offset. It recomputes the common pelvis from the predicted geometry on every frame.

## Transform chain and numerical Gate

3DPW `cam_poses` is world→camera:

```text
X_camera = R_cw X_world + t_cw
X_world  = R_cw^T (X_camera - t_cw)
```

The unique prediction chain is:

```text
MHR local vertices
  -> official barycentric MHR→SMPL6890 surface mapping
  -> normalized H36M J17/J14 and pelvis in MHR-local coordinates
  -> add pred_cam_t in camera coordinates
  -> apply R_cw^T (X_camera - t_cw) to world coordinates
```

Gate measurements:

- barycentric row-sum maximum error: `2.220446049250313e-16`
- translation-equivariance maximum error: `4.440892098500626e-16 m`
- GT world→camera→world maximum closure error: `2.220446049250313e-15 m`
- prediction camera→world→camera maximum closure error: `8.881784197001252e-15 m`

These values validate units, rotation direction, translation convention, frame indexing, and the prediction/GT common-pelvis definition.

## Small world-space Camera smoke

Artifact: `CAMERA_WORLD_SMOKE_20261002.json`

Scope: one central 16-frame window per person track, 4 tracks, all four occlusion conditions, Raw/Frame-wise/Full. This is a development smoke, not a full benchmark.

Across the four conditions, sample-weighted means:

| Method | Pelvis error m | W-MPJPE m | WA-MPJPE m | ATE m | RTE m/frame | Velocity m/s | Acceleration m/s² | Jitter m/s³ | Motion ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 5.7931 | 5.7910 | 0.5832 | 5.8150 | 0.1624 | 4.8732 | 216.11 | 12148.2 | 6.053 |
| Frame-wise | 0.2885 | 0.2893 | 0.1666 | 0.3075 | 0.1268 | 3.8036 | 195.01 | 10972.5 | 4.935 |
| Full | 0.2995 | 0.2977 | 0.1456 | 0.3164 | 0.0750 | 2.2491 | 97.95 | 5078.9 | 3.013 |

Interpretation:

- Raw→Frame-wise strongly improves global spatial accuracy.
- Frame-wise→Full improves WA-MPJPE and every temporal metric, but slightly worsens aggregate direct pelvis/W-MPJPE/ATE.
- Full/Frame-wise motion magnitude is below `0.8` for every occlusion condition, so over-smoothing risk remains. Full motion magnitude nevertheless remains above GT (`2.01–4.14×` by condition), so the result does not show collapse below real motion.

## Evaluation authorization

The root-correspondence blocker is cleared. A formal 3DPW Camera evaluator may now use the chain above, provided its metric definitions are frozen before evaluation and it retains the official barycentric mapping, row-normalized H36M regressor, oracle 3DPW extrinsics, identity/frame alignment, and no test-based calibration.

This audit does not authorize automatic execution of a full benchmark.
