# DynaMesh4D-v2 Camera Phase-1.6 Loss Rebalancing Ablation

Use only `courtyard_basketball_00/person1`, frames `506,508,...,536`, failure frame 522. Reuse the frozen SAM camera heads, unchanged CameraFeatureTemporalAdapter architecture, validated 12-joint projection contract, and existing Phase-1/1.5 artifacts. Do not rerun either prior phase, modify Body, or start full training.

Measure four-condition mean raw gradient norms at the Phase-1 best checkpoint. Keep `lambda_reproj=1`; target weighted teacher/anchor gradients at 1.5 times reprojection and reliability at 0.25 times reprojection. This deterministic recipe balances the core losses without a lambda search.

- Exp-A: original Phase-1 term definitions with derived global lambdas.
- Exp-B: teacher/anchor frame weights use detached predicted reliability times observation stability. GT 2D is only for reprojection loss and soft reliability supervision.
- Exp-C: Exp-B with correction gate fixed open while reliability remains trained and reported.

Each experiment starts independently from the same seeded identity adapter, trains 400 balanced-condition AdamW steps, and saves separate checkpoints. Evaluation reports frame 522, neighbors 520/524, full-window/four-condition metrics, reliability, innovation, stable deviation, and depth/scale drift. Outputs live only under `outputs/dynamesh_v2_camera_phase16_loss_rebalancing/`.
