# DynaMesh4D-v2 Unified 3DPW Evaluation Protocol B

Status: frozen before Protocol-B Test metric generation.

- Coverage: every `campose_valid` person-frame in 3DPW Test, keyed strictly by sequence, person ID, and original frame ID.
- Inputs: existing frozen feature caches; the visual backbone is never rerun.
- Conditions: clean, 15%, 35%, and 55% synthetic occlusion.
- Methods: frozen Raw SAM Body4D and frozen DynaMesh4D-v2 Body adapter. Camera checkpoints are hashed but do not affect pelvis-aligned body geometry.
- Temporal contract: fixed 16-frame windows, stride 8. Overlaps select the prediction farthest from a window boundary; ties select the earlier window. The final window is always included. Segments shorter than 16 use edge replication. Windows never cross a discontinuity, person, or sequence.
- Conversion: each sequence/person/condition/method is independently fitted by the official Meta MHR-to-SMPL converter with `single_identity=True`. No identity state is shared across methods or conditions.
- Joint protocol: SMPL 6890 vertices, `J_regressor_h36m.npy`, and H36M-to-J14 indices `[6,5,4,1,2,3,16,15,14,11,12,13,8,10]`.
- MPJPE: H36M pelvis translation alignment only; no rotation or scale alignment.
- PA-MPJPE: per-frame similarity Procrustes alignment on the same J14 joints.
- MPVPE: prediction and GT SMPL vertices are independently centered at their H36M pelvis; no rotation or scale alignment.
- Acceleration error: second-difference error on pelvis-centered H36M J14, reported as `mm/processed-frame^2`; processed data are 30 fps and no `fps^2` multiplier is applied.
- Persistence: fitted SMPL parameters, converter fit error, H36M/J14 joints, frame IDs, source/cache/GT hashes, protocol hash, and checkpoint/code hashes are stored. Full SMPL/MHR vertices are temporary and removed after metrics are atomically committed.
- Resume: a completed unit is skipped only after state checksum, protocol/input hashes, representation hashes/schema/finite checks, and metric checksum/finite checks all pass.
- Aggregation: frame/triplet-weighted means plus paired person-track bootstrap confidence intervals (10,000 resamples, fixed seed).
- Selection policy: Protocol B exists only for protocol alignment and metric completeness. No model parameter or hyperparameter is selected from its Test results.
- Test safety: a validation-only preflight must pass before the immutable lock is written and formal Test evaluation begins.
