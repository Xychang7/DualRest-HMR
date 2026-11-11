# DynaMesh4D-v2 Camera Phase-1.5 Oracle Capacity Diagnosis

## Scope

Use only the selected `courtyard_basketball_00/person1` 16-frame window and existing cache-v2/frozen-head outputs. Do not run SAM visual inference, modify Body, train on other windows, or enter full Camera training.

## Independent reprojection oracles

For each of clean/15/35/55 independently, initialize a dedicated variable from that condition's own raw prediction. No variable, initialization, teacher, or identity information is shared across conditions.

1. Direct-camera oracle optimizes a free `(16,3)` `pred_cam` tensor initialized from the condition's raw `pred_cam`.
2. Free-token oracle optimizes a free `(16,1024)` camera token initialized from the condition's raw cached token and decodes it through the frozen PerspectiveHead.

Both use the identical strict 12-joint correspondence, confidence mask, frozen per-condition MHR70 points, PerspectiveHead projection, normalized `L_reproj`, Adam, 400 maximum steps, and best-loss stopping/checkpoint rule. They use no teacher, anchor, reliability, root, temporal, or cross-condition loss.

The primary capacity conclusion uses clean frame 522 and the clean full-window mean/P95. Synthetic conditions diagnose consistency only.

## Gradient decomposition

Load the current Phase-1 Camera-v0 best checkpoint. Average each of the four Phase-1 loss terms independently across the four balanced conditions. For every term report its scalar loss, raw full-adapter gradient L2 norm, configured lambda, and weighted gradient norm `lambda * raw_norm`. Do not update parameters.

## Comparison and verdict

Compare Raw SAM, Direct pred_cam Oracle, Free-token Oracle, and current Camera-v0 at frame 522 and full-window mean/P95. Interpret:

- strong direct and token gains with weak Camera-v0: adapter/gating/loss optimization bottleneck;
- strong direct but weak token: frozen PerspectiveHead token-space controllability bottleneck;
- weak direct: reprojection/objective/data contract capacity limit;
- highly imbalanced weighted gradients: Phase-1 loss competition bottleneck.

Optional forced-gate-zero reprojection-only adapter overfit is run only if the two main oracles establish useful capacity but the source of Camera-v0 failure remains ambiguous.

