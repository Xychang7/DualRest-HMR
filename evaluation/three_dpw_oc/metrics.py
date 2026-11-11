"""JOTR-aligned spatial metrics and OC-filtered full-video acceleration."""

from collections import Counter

import numpy as np


J14 = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)


def _mesh(mesh, name):
    array = np.asarray(mesh, dtype=np.float64)
    if array.shape != (6890, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (6890,3) in meters")
    return array


def _regressor(regressor, rows, name):
    array = np.asarray(regressor, dtype=np.float64)
    if array.shape != (rows, 6890) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite ({rows},6890)")
    return array


def _j14(mesh, h36m_regressor):
    joints = h36m_regressor @ mesh
    return (joints - joints[:1])[list(J14)]


def similarity_align(prediction, target):
    """Full scale/rotation/translation Procrustes on 14 joints."""
    pred_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = prediction - pred_mean
    y = target - target_mean
    variance = np.sum(x*x)
    if variance < 1e-16:
        raise ValueError("Degenerate predicted joints for Procrustes")
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    scale = np.sum(singular * np.diag(correction)) / variance
    return scale * x @ rotation + target_mean


def spatial_metrics(prediction, target, h36m_regressor, native_smpl_regressor):
    prediction = _mesh(prediction, "prediction")
    target = _mesh(target, "target")
    h36m = _regressor(h36m_regressor, 17, "corrected H36M regressor")
    native = np.asarray(native_smpl_regressor, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 6890 or native.shape[0] < 1:
        raise ValueError("native SMPL regressor must include joint 0 pelvis")
    pred_j14, gt_j14 = _j14(prediction, h36m), _j14(target, h36m)
    pred_native_pelvis, gt_native_pelvis = native[0] @ prediction, native[0] @ target
    return {"mpjpe_mm": float(np.linalg.norm(pred_j14 - gt_j14, axis=1).mean()*1000),
            "pa_mpjpe_mm": float(np.linalg.norm(similarity_align(pred_j14, gt_j14) - gt_j14,
                                               axis=1).mean()*1000),
            "mpvpe_mm": float(np.linalg.norm((prediction - pred_native_pelvis) -
                                             (target - gt_native_pelvis), axis=1).mean()*1000)}


def acceleration_metrics(prediction_meshes, gt_meshes, oc_keys, h36m_regressor, fps=30,
                         center_status=None):
    """Use complete RGB actor tracks; only OC center frames enter the mean."""
    if fps != 30:
        raise ValueError("3DPW-OC main protocol is fixed at continuous 30 fps")
    h36m = _regressor(h36m_regressor, 17, "corrected H36M regressor")
    reasons = Counter()
    values = []
    for sequence, center, actor in sorted(oc_keys):
        key = (sequence, center, actor)
        status = "valid" if center_status is None else center_status.get(key, "missing_source_actor_frame")
        if status != "valid":
            reasons[status] += 1
            continue
        triple = [(sequence, center + offset, actor) for offset in (-1, 0, 1)]
        if center == 0:
            reasons["boundary"] += 1
            continue
        if any(key not in prediction_meshes for key in triple):
            reasons["missing_adjacent_frame"] += 1
            continue
        if any(key not in gt_meshes for key in triple):
            reasons["missing_gt"] += 1
            continue
        pred = np.stack([_j14(_mesh(prediction_meshes[key], "prediction"), h36m)
                         for key in triple])
        gt = np.stack([_j14(_mesh(gt_meshes[key], "GT"), h36m)
                       for key in triple])
        pred_acc = pred[2] - 2*pred[1] + pred[0]
        gt_acc = gt[2] - 2*gt[1] + gt[0]
        values.append(float(np.linalg.norm(pred_acc - gt_acc, axis=1).mean() * fps**2))
    return {"mean_m_s2": float(np.mean(values)) if values else None,
            "valid_triplets": len(values), "invalid_reasons": dict(sorted(reasons.items()))}


def audit_oc_acceleration_keys(manifest: dict) -> dict:
    """Count OC centers whose RGB-frame neighbours also have OC JSON GT.

    This is a metadata-only upper bound.  Runtime prediction validity is still
    enforced by :func:`acceleration_metrics` and can only reduce this count.
    """
    keys = {(str(row["sequence"]), int(row["rgb_frame_idx"]),
             int(row["source_actor_id"])) for row in manifest["instances"]}
    reasons = Counter()
    valid = 0
    for sequence, center, actor in sorted(keys):
        if center == 0:
            reasons["boundary"] += 1
            continue
        neighbours = ((sequence, center - 1, actor),
                      (sequence, center + 1, actor))
        if any(key not in keys for key in neighbours):
            reasons["missing_oc_gt_neighbor"] += 1
            continue
        valid += 1
    return {"candidate_centers": len(keys), "valid_triplets": valid,
            "invalid_reasons": dict(sorted(reasons.items()))}
