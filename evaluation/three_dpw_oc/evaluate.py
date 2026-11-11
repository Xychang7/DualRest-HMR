"""Model-agnostic evaluation on OC actor instances and full RGB video tracks."""

import numpy as np

from .manifest import validate_predictions
from .metrics import spatial_metrics, acceleration_metrics
from .gt import (GenderedSMPLGT, build_oc_gt_meshes,
                 build_sequence_temporal_gt, load_corrected_h36m_regressor)


def evaluate_records(manifest: dict, prediction_records, gt_full_meshes: dict,
                     corrected_h36m_regressor, native_smpl_regressor,
                     gt_temporal_meshes=None, temporal_center_status=None) -> dict:
    """Compute OC spatial metrics and 30 fps Accel without dropping missing instances.

    Both prediction_records and gt_full_meshes must include complete valid RGB
    actor trajectories for Accel. Only OC manifest centers enter reported means.
    Missing OC center predictions or GT are errors, not silent exclusions.
    """
    indexed = validate_predictions(prediction_records)
    oc_keys = set()
    per_instance = []
    for instance in manifest["instances"]:
        key = (str(instance["sequence"]), int(instance["rgb_frame_idx"]),
               int(instance["source_actor_id"]))
        if key in oc_keys:
            raise ValueError(f"duplicate OC canonical key: {key}")
        oc_keys.add(key)
        if key not in indexed or not indexed[key]["valid"]:
            raise ValueError(f"missing OC prediction: {key}")
        if key not in gt_full_meshes:
            raise ValueError(f"missing OC GT: {key}")
        metrics = spatial_metrics(indexed[key]["vertices"], gt_full_meshes[key],
                                  corrected_h36m_regressor, native_smpl_regressor)
        per_instance.append({"sequence": key[0], "rgb_frame_idx": key[1],
                             "source_actor_id": key[2],
                             "annotation_id": int(instance["annotation_id"]), **metrics})
    if not per_instance:
        raise ValueError("empty OC instance manifest")
    prediction_full = {key: row["vertices"] for key, row in indexed.items() if row["valid"]}
    temporal_gt = gt_full_meshes if gt_temporal_meshes is None else gt_temporal_meshes
    accel = acceleration_metrics(prediction_full, temporal_gt, oc_keys,
                                 corrected_h36m_regressor,
                                 center_status=temporal_center_status)
    return {"spatial_instances": len(per_instance),
            "mpjpe_mm": float(np.mean([row["mpjpe_mm"] for row in per_instance])),
            "pa_mpjpe_mm": float(np.mean([row["pa_mpjpe_mm"] for row in per_instance])),
            "mpvpe_mm": float(np.mean([row["mpvpe_mm"] for row in per_instance])),
            "acceleration": accel, "per_instance": per_instance}


def evaluate_from_oc_json(manifest: dict, prediction_records, dataset_root,
                          smpl_model_directory, corrected_regressor_path) -> dict:
    """JOTR-aligned public entry point with fail-closed resource contracts.

    GT is built only from ``3DPW_oc.json`` gendered SMPL parameters.  The
    corrected H36M regressor is loaded before any expensive SMPL construction,
    and the legacy similarly-shaped regressor is never accepted as fallback.
    """
    corrected = load_corrected_h36m_regressor(corrected_regressor_path)
    builder = GenderedSMPLGT(smpl_model_directory)
    gt_meshes = build_oc_gt_meshes(dataset_root, manifest["instances"], builder)
    temporal_meshes, temporal_status, temporal_audit = build_sequence_temporal_gt(
        dataset_root, manifest["instances"], builder)
    result = evaluate_records(manifest, prediction_records, gt_meshes, corrected,
                              builder.native_pelvis_regressor, temporal_meshes,
                              temporal_status)
    result["temporal_gt_audit"] = temporal_audit
    return result
