"""Lightweight protocol acceptance; never runs an HMR model or converter."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.three_dpw_oc.evaluate import evaluate_records
from evaluation.three_dpw_oc.gt import (GenderedSMPLGT, build_oc_gt_meshes,
                                         load_corrected_h36m_regressor)
from evaluation.three_dpw_oc.manifest import validate_predictions
from evaluation.three_dpw_oc.metrics import (audit_oc_acceleration_keys,
                                               spatial_metrics)


def _first_consecutive_triple(instances):
    indexed = {(row["sequence"], int(row["rgb_frame_idx"]),
                int(row["source_actor_id"])): row for row in instances}
    for sequence, frame, actor in sorted(indexed):
        keys = [(sequence, frame + offset, actor) for offset in (-1, 0, 1)]
        if frame > 0 and all(key in indexed for key in keys):
            return [indexed[key] for key in keys]
    raise RuntimeError("No same-actor consecutive OC RGB triplet")


def _records(instances, meshes):
    result = []
    for row in instances:
        key = (row["sequence"], int(row["rgb_frame_idx"]),
               int(row["source_actor_id"]))
        result.append({"sequence": key[0], "rgb_frame_idx": key[1],
                       "pred_track_id": key[2], "matched_actor_id": key[2],
                       "vertices": meshes[key], "valid": True})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--corrected-regressor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assert manifest["oc_instances"] == 20243
    assert manifest["oc_sequences"] == 22
    corrected = load_corrected_h36m_regressor(args.corrected_regressor)
    builder = GenderedSMPLGT(args.smpl_model_dir)
    triple = _first_consecutive_triple(manifest["instances"])
    gt = build_oc_gt_meshes(args.dataset_root, triple, builder)
    tiny_manifest = {"instances": triple}

    identical = evaluate_records(tiny_manifest, _records(triple, gt), gt,
                                  corrected, builder.native_pelvis_regressor)
    translated = {key: mesh + np.array([0.31, -0.17, 0.42]) for key, mesh in gt.items()}
    translation = evaluate_records(tiny_manifest, _records(triple, translated), gt,
                                    corrected, builder.native_pelvis_regressor)
    center_key = (triple[1]["sequence"], int(triple[1]["rgb_frame_idx"]),
                  int(triple[1]["source_actor_id"]))
    center_gt = gt[center_key]
    pelvis = builder.native_pelvis_regressor[0] @ center_gt
    scaled = (center_gt - pelvis) * 1.2 + pelvis
    scale = spatial_metrics(scaled, center_gt, corrected,
                            builder.native_pelvis_regressor)

    order_rejected = False
    try:
        validate_predictions(list(reversed(_records(triple, gt))))
    except ValueError:
        order_rejected = True
    if not order_rejected:
        raise AssertionError("shuffled prediction records were not rejected")

    audit = audit_oc_acceleration_keys(manifest)
    report = {
        "resource": {"file": str(Path(args.corrected_regressor).resolve()),
                     "sha256": hashlib.sha256(Path(args.corrected_regressor).read_bytes()).hexdigest(),
                     "shape": list(corrected.shape), "dtype": str(corrected.dtype),
                     "max_row_sum_error": float(np.max(np.abs(corrected.sum(axis=1) - 1.0)))},
        "manifest": {"spatial_instances": manifest["oc_instances"],
                     "oc_sequences": manifest["oc_sequences"], **audit},
        "test_a_gt_as_prediction": {key: identical[key] for key in
                                     ("mpjpe_mm", "pa_mpjpe_mm", "mpvpe_mm", "acceleration")},
        "test_b_constant_translation": {key: translation[key] for key in
                                          ("mpjpe_mm", "pa_mpjpe_mm", "mpvpe_mm")},
        "test_c_scale_1p2": scale,
        "test_d_shuffled_order_rejected": order_rejected,
        "real_triplet": [{"sequence": row["sequence"],
                          "rgb_frame_idx": row["rgb_frame_idx"],
                          "source_actor_id": row["source_actor_id"]} for row in triple],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    assert max(identical["mpjpe_mm"], identical["pa_mpjpe_mm"],
               identical["mpvpe_mm"], identical["acceleration"]["mean_m_s2"]) < 1e-8
    # The official corrected regressor rows are close to, but not exactly,
    # affine-normalized (max row-sum error is recorded above).  Consequently a
    # large translated mesh leaves a sub-millimetre MPJPE residual after joint
    # 0 centering; this must not be "fixed" by modifying the official matrix.
    assert max(translation["mpjpe_mm"], translation["pa_mpjpe_mm"],
               translation["mpvpe_mm"]) < 0.5
    assert scale["pa_mpjpe_mm"] < 0.1 and scale["mpjpe_mm"] > 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
