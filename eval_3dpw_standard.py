"""Strict canonical 3DPW evaluator using one shared H36M joint regressor."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


H36M_TO_J14 = (6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10)
REQUIRED_FIELDS = ("sequence", "person_id", "frame_id", "smpl_vertices")


def _key(record: Mapping) -> tuple[str, int, int]:
    return str(record["sequence"]), int(record["person_id"]), int(record["frame_id"])


def validate_records(records: Sequence[Mapping], *, source: str) -> dict[tuple[str, int, int], Mapping]:
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError(f"{source} records must be a non-empty list or tuple")
    indexed = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}[{index}] must be a mapping")
        for field in REQUIRED_FIELDS:
            if field not in record:
                raise ValueError(f"{source}[{index}] missing required field {field}")
        if not isinstance(record["sequence"], str) or not record["sequence"]:
            raise ValueError(f"{source}[{index}] sequence must be a non-empty string")
        for field in ("person_id", "frame_id"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{source}[{index}] {field} must be an integer")
        vertices = np.asarray(record["smpl_vertices"])
        if vertices.shape != (6890, 3):
            raise ValueError(f"{source}[{index}] smpl_vertices must have shape (6890, 3), got {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{source}[{index}] smpl_vertices contains NaN or Inf")
        key = _key(record)
        if key in indexed:
            raise ValueError(f"{source} duplicate key {key}")
        indexed[key] = record
    return indexed


def strict_match_records(predictions: Sequence[Mapping], ground_truth: Sequence[Mapping]):
    pred_index = validate_records(predictions, source="prediction")
    gt_index = validate_records(ground_truth, source="ground_truth")
    pred_keys, gt_keys = set(pred_index), set(gt_index)
    if pred_keys != gt_keys:
        missing = sorted(pred_keys - gt_keys)
        extra = sorted(gt_keys - pred_keys)
        raise ValueError(f"prediction/ground_truth key mismatch: missing_gt={missing}, extra_gt={extra}")
    keys = sorted(pred_keys)
    return [pred_index[k] for k in keys], [gt_index[k] for k in keys], keys


def regress_h36m_j14(vertices: np.ndarray, joint_regressor: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 2:
        vertices = vertices[None]
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3):
        raise ValueError(f"vertices must have shape (N, 6890, 3), got {vertices.shape}")
    regressor = np.asarray(joint_regressor, dtype=np.float64)
    if regressor.shape != (17, 6890):
        raise ValueError(f"J_regressor_h36m must have shape (17, 6890), got {regressor.shape}")
    joints_h36m = np.einsum("jv,nvc->njc", regressor, vertices)
    centered = joints_h36m - joints_h36m[:, 0:1]
    return centered[:, H36M_TO_J14]


def _similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = np.sum(source_centered**2)
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("cannot Procrustes-align a degenerate joint set")
    covariance = source_centered.T @ target_centered
    u, _, vh = np.linalg.svd(covariance)
    v = vh.T
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ v.T))
    rotation = v @ correction @ u.T
    scale = np.trace(rotation @ covariance) / variance
    translation = target_mean - scale * rotation @ source_mean
    return (scale * rotation @ source.T).T + translation


def evaluate_records(predictions: Sequence[Mapping], ground_truth: Sequence[Mapping], joint_regressor: np.ndarray) -> dict:
    pred_records, gt_records, keys = strict_match_records(predictions, ground_truth)
    pred_vertices = np.stack([np.asarray(r["smpl_vertices"], dtype=np.float64) for r in pred_records])
    gt_vertices = np.stack([np.asarray(r["smpl_vertices"], dtype=np.float64) for r in gt_records])
    pred_j14 = regress_h36m_j14(pred_vertices, joint_regressor)
    gt_j14 = regress_h36m_j14(gt_vertices, joint_regressor)
    per_frame_mpjpe_mm = np.linalg.norm(pred_j14 - gt_j14, axis=-1).mean(axis=-1) * 1000.0
    aligned = np.stack([_similarity_align(p, g) for p, g in zip(pred_j14, gt_j14)])
    per_frame_pa_mpjpe_mm = np.linalg.norm(aligned - gt_j14, axis=-1).mean(axis=-1) * 1000.0
    return {
        "num_records": len(keys),
        "keys": keys,
        "mpjpe_mm": float(per_frame_mpjpe_mm.mean()),
        "pa_mpjpe_mm": float(per_frame_pa_mpjpe_mm.mean()),
        "per_frame_mpjpe_mm": per_frame_mpjpe_mm,
        "per_frame_pa_mpjpe_mm": per_frame_pa_mpjpe_mm,
    }


def _load_pickle_records(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("records") if isinstance(payload, Mapping) else payload
    validate_records(records, source="prediction")
    return records


def select_person_betas(betas_by_person, person_id: int) -> np.ndarray:
    """Select an actor before ndarray conversion; 3DPW actor vectors may be ragged."""
    if person_id < 0 or person_id >= len(betas_by_person):
        raise ValueError(f"GT betas has no person_id={person_id}")
    betas = np.asarray(betas_by_person[person_id])
    if betas.ndim != 1:
        raise ValueError(f"GT person betas must be 1D, got shape {betas.shape}")
    return betas


def build_gt_records(keys: Iterable[tuple[str, int, int]], *, gt_dir: Path, smpl_model_path: Path, device: str = "cpu"):
    import smplx
    import torch

    requested = sorted(set(keys))
    grouped = {}
    for sequence, person_id, frame_id in requested:
        grouped.setdefault((sequence, person_id), []).append(frame_id)
    records = []
    for (sequence, person_id), frame_ids in grouped.items():
        gt_path = gt_dir / f"{sequence}.pkl"
        if not gt_path.exists():
            raise ValueError(f"missing GT sequence file {gt_path}")
        with gt_path.open("rb") as handle:
            try:
                gt = pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                gt = pickle.load(handle, encoding="latin1")
        poses_all = np.asarray(gt["poses"])
        if person_id < 0 or person_id >= poses_all.shape[0]:
            raise ValueError(f"GT {sequence} has no person_id={person_id}")
        original_ids = np.asarray(gt["img_frame_ids"]).astype(int)
        if len(np.unique(original_ids)) != len(original_ids):
            raise ValueError(f"GT {sequence} contains duplicate original frame IDs")
        frame_lookup = {int(fid): index for index, fid in enumerate(original_ids)}
        missing = [fid for fid in frame_ids if fid not in frame_lookup]
        if missing:
            raise ValueError(f"GT {sequence} person {person_id} missing frame IDs {missing}")
        local_indices = np.asarray([frame_lookup[fid] for fid in frame_ids], dtype=int)
        valid = np.asarray(gt.get("campose_valid", np.ones(poses_all.shape[:2], dtype=bool)))[person_id].astype(bool)
        invalid = [fid for fid, index in zip(frame_ids, local_indices) if not valid[index]]
        if invalid:
            raise ValueError(f"GT {sequence} person {person_id} has invalid camera pose at frame IDs {invalid}")
        poses = poses_all[person_id, local_indices]
        betas_source = select_person_betas(gt["betas"], person_id)
        betas = betas_source[local_indices, :10] if betas_source.ndim == 2 else np.tile(betas_source[:10], (len(frame_ids), 1))
        transl = np.asarray(gt["trans"])[person_id, local_indices]
        model = smplx.SMPL(model_path=str(smpl_model_path), batch_size=len(frame_ids)).to(device)
        with torch.no_grad():
            output = model(
                betas=torch.as_tensor(betas, dtype=torch.float32, device=device),
                body_pose=torch.as_tensor(poses[:, 3:72], dtype=torch.float32, device=device),
                global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float32, device=device),
                transl=torch.as_tensor(transl, dtype=torch.float32, device=device),
            )
        vertices_world = output.vertices.detach().cpu().numpy()
        camera = np.asarray(gt["cam_poses"])[local_indices]
        vertices_camera = np.einsum("nij,nvj->nvi", camera[:, :3, :3], vertices_world) + camera[:, None, :3, 3]
        for frame_id, vertices in zip(frame_ids, vertices_camera):
            records.append({"sequence": sequence, "person_id": person_id, "frame_id": frame_id, "smpl_vertices": vertices})
    validate_records(records, source="ground_truth")
    return records


def main():
    parser = argparse.ArgumentParser(description="Strict canonical 3DPW SMPL-vertex evaluator")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--smpl-model-path", type=Path, required=True)
    parser.add_argument("--joint-regressor", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    predictions = _load_pickle_records(args.predictions)
    keys = sorted(validate_records(predictions, source="prediction"))
    ground_truth = build_gt_records(keys, gt_dir=args.gt_dir, smpl_model_path=args.smpl_model_path, device=args.device)
    result = evaluate_records(predictions, ground_truth, np.load(args.joint_regressor))
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    serializable["keys"] = [list(k) for k in result["keys"]]
    text = json.dumps(serializable, indent=2)
    if args.output_json:
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
