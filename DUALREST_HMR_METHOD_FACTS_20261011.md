# DualRest-HMR Method Implementation Fact Sheet

Date: 2026-10-11  
Scope: frozen implementation facts only; no claims inferred from benchmark outcomes.

## 1. Audited formal artifacts

| Artifact | Path | SHA256 | Status |
|---|---|---|---|
| Body restoration | `/root/sam-body4d-0.2.0/outputs/dynamesh_v2_phase2_training/best.pt` | `f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329` | exact match |
| Frame-wise camera | `/root/sam-body4d-0.2.0/outputs/dynamesh_v2_camera_phase2/framewise/best.pt` | `fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2` | exact match |
| Temporal camera | `/root/sam-body4d-0.2.0/outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt` | `e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865` | exact match |

Camera data audit verdict: `NO_TEST_LEAKAGE`. Both Camera checkpoints optimize only on the official 3DPW train split and use the disjoint official validation split for checkpoint selection/early stopping. Test is rejected by loader guards and no test cache file exists.

## 2. Full inference pipeline

`video frames -> frozen SAM-Body4D image/prompt encoder and transformer decoder -> one 1024-D decoder pose token per person/frame -> {Body residual restoration, Camera restoration} -> frozen MHRHead/PerspectiveHead -> late parameter composition -> local MHR mesh + camera-space translation/output`.

- Frozen foundation: the complete loaded SAM-Body4D estimator, including visual backbone, prompt/transformer decoder, `MHRHead` (`head_pose`), `PerspectiveHead` (`head_camera`), and learned initial estimates.
- Both restoration paths originate from the same raw SAM decoder pose token; they do not exchange learned features.
- Body path: raw pose token -> Body Bi-Mamba adapter -> frozen `head_pose`; only body articulation is retained from this refined decode.
- Camera path: raw pose token -> frame-wise residual MLP -> frozen `head_camera` -> frame-wise `pred_cam/pred_cam_t`; then frame-wise refined token -> conditioned temporal Bi-Mamba residual -> frozen `head_camera` -> final `pred_cam/pred_cam_t`.
- `compose_parameters` replaces only `body_pose`/`body_pose_params` and `pred_cam`/`pred_cam_t`. Global rotation, hands, scale, shape, face/expression and jaw remain raw.
- There is no learned Body-Camera fusion. The unification is late, deterministic parameter composition followed by local MHR decoding and camera translation.

Primary call sites: `sam3d_body.py:decoder`, `dynamesh_v2_temporal.py:BodyFeatureTemporalAdapter`, `camera_temporal.py:CameraFeatureTemporalAdapter`, `train_dynamesh_v2_camera_phase112.py:temporal_residual_forward`, and `dynamesh_v2_integration.py:compose_parameters/rebuild_local_vertices`.

## 3. Body representation and decoding

The accurate name is **SAM-Body4D decoder pose token** (or **1024-D pose-query token**). `sam3d_body.py` enforces one pose token, selects `tokens[:, 0]`, sends the same token to `head_pose` and `head_camera`, and exports it as `pose_output["pose_token"]`.

It is not a class token, separate global token, decoded pose vector, or a collection of continuous body parameters. The cached token has shape `(condition=4, T, 1024)`; a training batch has `(B, 4, 16, 1024)` and the adapter receives `(B, 16, 1024)`. The decoded MHR initialization is `(B,16,519)`. The 260-D continuous body articulation slice is `head_pose.proj(token)+init` at indices `[6:266]`; it is a loss/decode representation, not the temporal adapter input.

The frozen `MHRHead` predicts a 519-D parameter vector: 6-D global rotation, 260-D continuous body articulation, 45 shape, 28 scale, 108 hand, and 72 face coefficients. It converts these to MHR parameters and returns 18,439 vertices plus joints and pose fields. In the formal Body composition, refined articulation is combined with raw global rotation, hand, scale, shape, and expression fields.

The official barycentric MHR-to-SMPL6890 conversion and H36M regression are evaluation bridges only, not network components.

## 4. Body temporal restoration architecture

Class: `BodyFeatureTemporalAdapter`.

For token sequence `z` and scalar condition `a`:

`h = Linear_1024->256(LayerNorm(z)) + MLP_1->256->256(a)`  
`u = BiMamba(h) = Mamba_fwd^2(h) + reverse(Mamba_bwd^2(reverse(h)))`  
`delta_z = Linear_256->1024(LayerNorm(u))`  
`z_restored = z + delta_z`.

The condition MLP is `Linear(1,256)-SiLU-Linear(256,256)`. Each direction has two Mamba blocks with `d_model=256`, `d_state=16`, `d_conv=4`, `expand=2`. The output layer was zero-initialized before training. Output is all frames, not center-frame only.

Formal full-track Body inference uses window 16, stride 8. Every frame is assigned the prediction from the window where it is most central; ties select the earlier window. A short contiguous segment is edge-replicated, and the final length-16 window is included.

Parameters: 2,345,472, all trainable during Body training and frozen for formal inference.

## 5. Restoration strength

The implementation argument is `occlusion_ratio`; formal inference constructs a tensor shaped `(B,T)` filled with the fixed scalar `0.15`. It is embedded per frame by the condition MLP and added to the projected temporal latent before both directional Mamba stacks.

The formal inference value is a fixed restoration-strength condition. No measured or ground-truth occlusion ratio is required at inference. Training uses synthetic ratios `{0,.15,.35,.55}`, but this historical supervision does not make the real test-time occlusion ratio an input requirement.

`alpha=0` is **not** a structural bypass: it still evaluates the learned condition MLP (including biases), temporal blocks and residual head. The clean anchor encourages identity preservation, but exact raw identity at zero is not guaranteed after training.

## 6. Body training and formal loss

- Data: official 3DPW train split; 34 person tracks, 21,330 stride-1 windows. Validation: disjoint official 3DPW validation split; 16 tracks, 614 fixed non-overlap windows.
- Corruption: clean plus deterministic bbox-relative filled rectangle occlusions at 15%, 35%, 55%. All four conditions train and validate the model.
- Window/batch: length 16, training stride 1, batch 16.
- AdamW, LR `1e-3`, weight decay `1e-4`, maximum 12 epochs, patience 3. Training stopped after epoch 10; best checkpoint is epoch 7.
- Trainable: Body adapter only. Foundation and frozen `head_pose` are not optimized.

Formal objective:

- `L_feature = SmoothL1(z_restored, z_clean)`, weight 1.0.
- `L_body = mean geodesic rotation loss(decode260(z_restored, init_condition), decode260(clean_teacher))`, weight 0.1.
- `L_anchor = SmoothL1(Adapter(z_clean, 0), z_clean)`, weight 1.0.
- `L = L_feature + 0.1 L_body + L_anchor`.

Velocity and acceleration consistency losses belong only to the Dynamics-Low ablation (`ABLATION_ONLY`) and are not part of this formal Body checkpoint.

## 7. Frame-wise camera calibration

Class: `FramewiseLatentAdapter` in `ablate_dynamesh_v2_camera_phase111.py`.

`z_fw = z + Linear_128->1024(GELU(Linear_1024->128(LayerNorm(z))))`.

Input/output are `(B,T,1024)` (or any leading dimensions accepted by PyTorch linear layers). It is applied independently to every frame and contains no temporal operation or reliability cue. The frozen `head_camera(z_fw, init_camera)` produces `(s, tx, ty)`, and the frozen perspective conversion produces `pred_cam_t` and reprojection. This supports the precise phrase “frame-wise latent camera calibration”; it does not directly regress a free 3-D trajectory.

Parameters: 265,344.

## 8. Camera conditioning cues

The temporal Camera path packs six values plus six presence indicators (12-D condition per frame):

- `visibility` (1): code value `1-ratio`; for formal natural-video inference ratio is fixed to 0, hence this is a constant 1, not measured visibility.
- normalized bbox-center displacement (2): `(center_t-center_{t-1}) / bbox_size_t`, first frame zero.
- log bbox-scale change (1): `log(size_t/size_{t-1})`, first frame zero.
- mask-area change (1): absent in the formal pipeline, zero-valued with presence mask 0.
- temporal latent disagreement (1): norm of the token minus the mean of its two neighbors, divided by `sqrt(1024)`, robustly normalized by its window 75th percentile and clipped to `[0,4]`; boundary frames zero.

These are inference-available conditioning features, not a demonstrated explicit reliability estimator. They use no GT, no segmentation mask, no independent keypoint confidence, and no reprojection residual. The frame-wise adapter uses none of them.

Important implementation fact: `CameraFeatureTemporalAdapter` defines a 449-parameter `reliability_head` and a gated class `forward`, but the formal Phase-2 checkpoint training and formal evaluation call `temporal_residual_forward`, which bypasses both the reliability head and gate. Therefore no learned reliability scalar modulates the formal correction.

## 9. Temporal camera refinement

Formal input is the frozen frame-wise refined token `(B,16,1024)`, not the raw token. The 12-D cue vector is linearly projected to 128 and additively fused with `Linear_1024->128(LayerNorm(z_fw))`. Two forward and two reverse Mamba blocks use `d_model=128`, `d_state=16`, `d_conv=4`, `expand=2`. Their sum is normalized and projected to 1024:

`h_c = Linear(LN(z_fw)) + Linear([cue, presence_mask])`  
`delta_c = Linear_128->1024(LN(BiMamba(h_c)))`  
`z_cam = z_fw + delta_c`  
`pred_cam_final = frozen_head_camera(z_cam, init_camera)`.

Temporal-only is not a valid formal ablation because this checkpoint was optimized on outputs from the specific frozen frame-wise checkpoint. The held-out full runner partitions each contiguous track into non-overlapping chunks of at most 16 and covers each frame exactly once; it does not overlap/merge Camera windows.

Parameters stored/trained: 733,633, including the 449 reliability-head parameters that are optimizer-owned but receive no gradient through the formal residual helper. Executed formal residual path excludes that head.

## 10. Camera variables and projection definition

`pred_cam = (s, tx, ty)` has shape `(N,3)`. Before conversion, the perspective head flips `s` and `ty` for the camera-system convention. With bbox size `b`, per-frame focal length `f=K[0,0]`, image size `(W,H)`, bbox center `(bx,by)`, and `bs=b*s*default_scale_factor+1e-8`:

`tz = 2f/bs`,  
`cx = 2(bx-W/2)/bs`, `cy = 2(by-H/2)/bs` (or intrinsic principal point when enabled),  
`pred_cam_t = (tx+cx, ty+cy, tz)` after the sign conversion.

The focal length comes from each frame's camera intrinsic matrix; projection is in original-image coordinates, so there is no single fixed image resolution in this definition. MHR local geometry is in meters and `pred_cam_t` is the corresponding meter-valued translation. Source directly applies `X_camera = X_local + pred_cam_t` before `K X / Z` projection.

The Camera branch predicts no camera rotation, no body/global orientation, and no independent world trajectory. World coordinates in evaluation are obtained only by applying provided 3DPW extrinsics to camera-space points.

## 11. Camera training and losses

Both stages use official 3DPW train only (24 sequences, 34 tracks) and the disjoint official validation split (12 sequences, 16 tracks). The prepared training set has 1,370 non-overlap length-16 windows; validation has 614 fixed non-overlap windows. Each step uses one window, cycling clean/15/35/55 conditions. AdamW uses LR `1e-3`, weight decay `1e-4`, maximum 12 epochs and patience 3.

Frame-wise stage:

- Foundation/head frozen; only `FramewiseLatentAdapter` trainable.
- Best epoch 11 (step 15,070); run completed 12 epochs.
- `L_cam3d`: reliability-weighted SmoothL1 on `pred_cam_t` versus fitted `t_all`, weight 1.
- `L_reproj`: confidence-masked mean 2-D joint error normalized by bbox size, weight `22.407989736107268`.
- Formal loss is `L_cam3d + 22.407989736107268 L_reproj`. Config entries named reliability and anchor are inherited probe metadata and are not called by the final Phase-2 frame loss.

Temporal stage:

- Loads and freezes the selected frame-wise adapter; only `CameraFeatureTemporalAdapter` is optimizer-owned.
- Best epoch 6 (step 8,220); early stopping after epoch 9.
- `L_velocity`: reliability-pair-weighted SmoothL1 of first-difference translation discrepancy, weight 1.
- `L_acceleration`: reliability-triplet-weighted SmoothL1 of second-difference translation discrepancy, weight `0.16511516225894365`.
- `L_absolute`: SmoothL1 to `w*t_all + (1-w)*pred_cam_t_framewise`, weight `0.9057701220278093`.
- `L_reproj`: same confidence-masked bbox-normalized reprojection loss, weight `0.11466139446581076`.

There is no formal Camera latent loss, reliability-head loss, explicit jitter loss, or separate motion-preservation loss. Jitter and motion ratios are evaluation metrics, not training objectives.

## 12. Training schedule and ownership

The implemented schedule is sequential, not joint end-to-end:

1. Train Body temporal restoration against cached frozen-foundation outputs; freeze selected Body checkpoint.
2. Train frame-wise Camera calibration with frozen SAM/head; select on validation.
3. Load and freeze that exact frame-wise checkpoint; train temporal Camera refinement; select on validation.

The Camera stages reference/audit the Body checkpoint hash but do not jointly optimize the Body adapter. At inference the two branches share the source SAM token and are composed late.

## 13. Tensor and ownership table

| Component | Input | Input shape | Output | Output shape | Training status |
|---|---|---:|---|---:|---|
| Frozen SAM decoder | encoded frame/person prompt | model-internal | `pose_token` | `(N,1024)` | frozen |
| Body adapter | pose token + strength | `(B,16,1024)`, `(B,16)` | restored pose token | `(B,16,1024)` | Body stage only |
| Frozen `head_pose` | restored token + init | `(N,1024)`, `(N,519)` | MHR params/vertices | body cont 260; vertices `(N,18439,3)` | frozen |
| Raw camera representation | raw pose token | `(N,1024)` | raw `pred_cam` | `(N,3)` | frozen head |
| Frame-wise adapter | raw pose token | `(B,T,1024)` | `z_fw` | `(B,T,1024)` | Camera stage A only |
| Frozen `head_camera` | camera token + init | `(N,1024)`, `(N,3)` | `pred_cam` | `(N,3)` | frozen |
| Temporal Camera adapter | `z_fw` + cues/masks | `(B,16,1024)`, `(B,16,12)` | `z_cam` | `(B,16,1024)` | Camera stage B only |
| Perspective conversion | local 3-D + `pred_cam` + metadata | `(N,J,3)`, `(N,3)` | `pred_cam_t`, 2-D | `(N,3)`, `(N,J,2)` | frozen/deterministic |

## 14. Parameter and cost audit

| Module | Parameters | Added/trainable in its stage |
|---|---:|---:|
| Frozen SAM-Body4D model | 1,285,412,468 | 0 |
| Body restoration | 2,345,472 | 2,345,472 |
| Frame-wise Camera | 265,344 | 265,344 |
| Temporal Camera | 733,633 | 733,633 |
| Total added | 3,344,449 | 3,344,449 |

Added parameters are 0.2601849% of the frozen foundation parameter count. Frozen `head_pose` contains 173,530,148 parameters and frozen `head_camera` 1,052,675. No trustworthy FLOP count or latency overhead is stored in the formal artifacts; this audit did not run a performance benchmark.

## 15. DO NOT CLAIM

- Do not claim prediction of camera rotation, global/body orientation, or an independent world trajectory.
- Do not claim end-to-end or joint Body-Camera training.
- Do not claim learned cross-branch fusion, cross-attention, or shared restoration weights.
- Do not call the adapter input decoded pose parameters, a CLS token, or a separate global token; it is the decoder pose token.
- Do not claim the formal Camera frame-wise adapter uses reliability cues.
- Do not claim the formal temporal output is reliability-gated or produced by a learned reliability estimator; the formal helper bypasses the reliability head/gate.
- Do not claim mask area, keypoint confidence, observed visibility, GT, or reprojection residual is used as a formal inference cue.
- Do not claim a real test-time occlusion ratio is required. Body uses fixed strength 0.15; Camera natural-video evaluation uses ratio 0 (visibility cue 1).
- Do not claim alpha 0 is an exact identity/bypass after training.
- Do not claim Temporal-only is a valid ablation.
- Do not claim Camera windows are overlap-merged in the held-out formal runner.
- Do not present MHR-to-SMPL conversion or H36M regression as a DualRest-HMR network component.
- Do not include Dynamics-Low velocity/acceleration losses in the main Body objective.
- Do not present Camera evaluation metrics (ATE/RTE/jitter/motion ratios) as training losses.
- Do not claim a fixed universal focal length or input resolution in `pred_cam_t` conversion; formal conversion uses per-frame intrinsics and original-image metadata.
- Do not claim all 733,633 temporal Camera parameters affect the formal forward: 449 reliability-head parameters are bypassed.
- Do not claim measured FLOPs or latency from this audit.

## 16. Remaining implementation caveats

No unresolved core tensor, loss, checkpoint, ownership, or coordinate-definition ambiguity remains for Method writing. Two reporting caveats remain: (1) exact FLOPs/latency were not measured; (2) the stored temporal Camera checkpoint includes an unused reliability head, so papers should describe the executed residual helper, not the class's unused default gated `forward`.

## 17. Source locations

- `/root/sam-body4d-0.2.0/models/sam_3d_body/sam_3d_body/models/meta_arch/sam3d_body.py`
- `/root/sam-body4d-0.2.0/models/sam_3d_body/sam_3d_body/models/heads/mhr_head.py`
- `/root/sam-body4d-0.2.0/models/sam_3d_body/sam_3d_body/models/heads/camera_head.py`
- `/root/sam-body4d-0.2.0/models/sam_3d_body/sam_3d_body/models/modules/dynamesh_v2_temporal.py`
- `/root/sam-body4d-0.2.0/models/sam_3d_body/sam_3d_body/models/modules/camera_temporal.py`
- `/root/sam-body4d-0.2.0/models/sam_3d_body/sam_3d_body/models/modules/dynamesh_v2_integration.py`
- `/root/sam-body4d-0.2.0/scripts/train_dynamesh_v2_phase2.py`
- `/root/sam-body4d-0.2.0/scripts/ablate_dynamesh_v2_camera_phase111.py`
- `/root/sam-body4d-0.2.0/scripts/train_dynamesh_v2_camera_phase112.py`
- `/root/sam-body4d-0.2.0/scripts/train_dynamesh_v2_camera_phase2.py`
- `/root/sam-body4d-0.2.0/scripts/evaluate_dynamesh_v2_unified_3dpw.py`
- `/root/sam-body4d-0.2.0/scripts/run_camera_branch_3dpw_test_final.py`
- `/root/sam-body4d-0.2.0/CAMERA_EVALUATOR_V1_PROTOCOL.md`
- `/root/sam-body4d-0.2.0/CAMERA_DATA_LEAKAGE_AUDIT.json`

## 18. Requested precision supplement (2026-10-11)

### 18.1 Exact 260-D articulation and geodesic supervision

The 260-D vector is the `body_cont` slice `[6:266]` of the frozen MHR head's 519-D prediction. Its exact packed layout is:

- dimensions `0:138`: 23 three-DoF rotations, each represented by a continuous 6-D rotation representation (`23 x 6 = 138`);
- dimensions `138:254`: 58 one-DoF rotations, each represented by a 2-D sine/cosine pair (`58 x 2 = 116`);
- dimensions `254:260`: six scalar translation parameters corresponding to MHR model-parameter indices `124..129`.

`compact_cont_to_rotmat_body` converts all 23 6-D entries to `3x3` matrices and all 58 sine/cosine entries to `2x2` rotations. The formal Body loss then selects only non-hand/non-jaw articulation:

- 12 selected three-DoF rotations;
- 34 selected one-DoF rotations, embedded into `SO(3)` as a top-left `2x2` block with final diagonal value 1;
- total: 46 rotation matrices per person/frame.

It excludes every rotation whose MHR parameter indices intersect hand indices `62..115`, excludes jaw indices `(130,131,132)`, and never applies geodesic loss to the six translation values `124..129`. For each of the 46 selected rotations, `R_rel = R_pred R_teacher^T`, `theta = acos(clamp((trace(R_rel)-1)/2, -1+1e-6, 1-1e-6))`; `L_body` is the mean `theta` over batch, time, and selected rotations.

### 18.2 Formal synthetic corruption and prediction-target pairing

The clean image is condition 0. Conditions 1/2/3 contain one deterministic black rectangle (`fill_value=0`) entirely inside the person bbox, with requested bbox-area ratios `.15/.35/.55`. For each image, the rectangle aspect ratio is sampled log-uniformly from `[0.6,1.6]`; target area is `ratio * bbox_width * bbox_height`; integer width/height are derived from this area, clipped to the bbox, and the top-left corner is sampled uniformly over legal positions. The seed is `20260829 + level*100000 + frame_id`, so corruption is deterministic per severity/frame. The bbox itself comes from valid GT 2-D joints, expanded independently in width/height by 1.2 and clipped to the image.

The cache stores four aligned variants per frame: `pose_token[4,T,1024]`, `mhr_init_pose[4,T,519]`, and decoded `body_cont[4,T,260]`. Training windows are shuffled, but condition choice is balanced by shuffled blocks of `[0,1,2,3]`: every four consecutive optimizer steps use each condition exactly once, and every sample in one batch uses that step's same condition. Validation evaluates every fixed window under all four conditions.

For a selected condition `c`, prediction uses `pose_token[c]`, `mhr_init_pose[c]`, and a constant per-window ratio from `{0,.15,.35,.55}`. The feature target is always `pose_token[clean]`; the rotation teacher is always `body_cont[clean]`; the refined rotation decode deliberately retains `mhr_init_pose[c]`. Separately, the anchor path predicts `Adapter(pose_token[clean], ratio=0)` and targets `pose_token[clean]`.

### 18.3 Exact Temporal Camera 12-D condition

The value vector order is:

`[v0 visibility, v1 bbox_dx, v2 bbox_dy, v3 bbox_dlog_scale, v4 mask_darea, v5 latent_disagreement]`.

- `v0 = 1-ratio`.
- `(v1,v2) = (center_t-center_{t-1}) / size_t`, with frame 0 equal to zero.
- `v3 = log(size_t/size_{t-1})`, with frame 0 equal to zero.
- `v4` is zero because `mask_area_delta=None` in the formal pipeline.
- `v5 = ||z_t - 0.5(z_{t-1}+z_{t+1})||_2/sqrt(1024)`, divided by the window 75th percentile (minimum denominator `1e-6`) and clipped to `[0,4]`; boundary frames are zero.

The next six dimensions are aligned presence masks `[m0..m5]`. A supplied field contributes ones for all its scalar dimensions; an absent field contributes zeros. The formal vector is therefore `concat(values, masks)` with mask pattern `[1,1,1,1,0,1]` per frame. Formally: indices `0..5` are values and indices `6..11` are their presence masks.

### 18.4 Complete `pred_cam` to `pred_cam_t` formula

Let the frozen camera head output be `pred_cam_raw=(s_raw, tx_raw, ty_raw)`. The perspective head first applies the camera-system sign conversion

`s=-s_raw`, `tx=tx_raw`, `ty=-ty_raw`.

Let `b` be the scalar crop bbox size, `d` the head's `default_scale_factor` (formal default 1), `f=K[0,0]`, `(bx,by)` the bbox center, and `(W,H)` the original image size. Define

`q = b*s*d + 1e-8`, `Tz=2f/q`, `Cx=2(bx-W/2)/q`, `Cy=2(by-H/2)/q`,

then

`pred_cam_t=(Tx,Ty,Tz)=(tx+Cx, ty+Cy, 2f/q)`.

The formal wrapper does not set `use_intrin_center`, so the image center `(W/2,H/2)` is used. The alternative unused branch would replace it with the intrinsic principal point `(K[0,2],K[1,2])`. In Camera Phase-2 metadata, a GT-2D-joint bbox is first formed with 1.2 width/height expansion; `(bx,by)` is its midpoint, while `b=1.25*(x2-x1)` uses its expanded/clipped width. `K` is the sequence camera intrinsic matrix and `(W,H)` are original-image dimensions. Projection then uses `X_camera=X_local+pred_cam_t` and `u~K X_camera`; returned 2-D coordinates are in original-image pixels.

### 18.5 Exact training hyperparameters

| Stage | Optimizer | LR / weight decay | Batch | Epochs and selection | Scheduler / stopping | Other optimization details |
|---|---|---|---|---|---|---|
| Body | AdamW, PyTorch default betas/eps | `1e-3` / `1e-4` | 16 windows, `drop_last=False` | max 12; actually 10; best epoch 7 by mean total loss over all four validation conditions | no scheduler; constant LR; patience 3; improvement threshold `1e-8` | no gradient clipping in formal loop; train windows shuffled with seed `20260831`; length 16, stride 1 |
| Frame-wise Camera | AdamW, PyTorch default betas/eps | `1e-3` / `1e-4` | one length-16 window per optimizer step | max 12; actually 12; best epoch 11, step 15,070 | no scheduler; constant LR; patience 3, but maximum epoch reached first | global-norm clipping at 1; deterministic cache order; condition `step mod 4` |
| Temporal Camera | AdamW, PyTorch default betas/eps | `1e-3` / `1e-4` | one length-16 window per optimizer step | max 12; actually 9; best epoch 6, step 8,220 | no scheduler; constant LR; patience 3; stopped after epochs 7--9 failed to improve | global-norm clipping at 1; frame-wise adapter frozen; deterministic cache order; condition `step mod 4` |
