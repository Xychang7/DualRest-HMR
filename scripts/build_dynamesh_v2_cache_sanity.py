"""Build Phase-1 clean/synthetic feature-cache sanity samples only.

This script intentionally processes one fixed 16-frame window from one train
and one validation sequence.  It never accepts the 3DPW test split.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path("/root/sam-body4d-0.2.0")
DATA_ROOT = Path("/root/data1/sam-body4d-master/dataset/3DPW")
OUTPUT_ROOT = ROOT / "outputs/dynamesh_v2_phase1_cache_sanity"
MANIFEST_ROOT = ROOT / "data/dynamesh_v2/manifests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.data.dynamesh_v2_cache import (
    SCHEMA_VERSION,
    apply_bbox_rectangle_occlusion,
    build_split_manifests,
    validate_cache_payload,
    validate_cache_split,
)

SAMPLES = {
    "train": ("outdoors_slalom_00", 0),
    "validation": ("outdoors_freestyle_01", 0),
}
OCCLUSION_LEVELS = np.asarray([0.15, 0.35, 0.55], dtype=np.float32)
WINDOW_LENGTH = 16
BASE_SEED = 20260829


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def bbox_from_pose2d(pose2d, image_width, image_height, expansion=1.2):
    pose2d = np.asarray(pose2d, dtype=np.float32)
    valid = np.isfinite(pose2d).all(axis=0) & (pose2d[2] > 0)
    if valid.sum() < 2:
        raise ValueError("Fewer than two valid GT 2D joints")
    x, y = pose2d[0, valid], pose2d[1, valid]
    cx, cy = float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2)
    width, height = float(x.max() - x.min()), float(y.max() - y.min())
    width = max(width * expansion, 16.0); height = max(height * expansion, 16.0)
    return np.asarray([
        max(0.0, cx - width / 2), max(0.0, cy - height / 2),
        min(float(image_width), cx + width / 2), min(float(image_height), cy + height / 2),
    ], dtype=np.float32)


def run_sequence(estimator, image_paths, bboxes, cam_int, shape_scale_cache=None):
    outputs = []
    shape_scale_cache = {} if shape_scale_cache is None else shape_scale_cache
    for start in range(0, len(image_paths), 4):
        stop = min(start + 4, len(image_paths))
        batch = estimator.process_frames(
            image_paths[start:stop],
            bboxes=[box.reshape(1, 4) for box in bboxes[start:stop]],
            masks=None,
            id_batch=[[1] for _ in range(stop - start)],
            idx_path={}, idx_dict={}, mhr_shape_scale_dict=shape_scale_cache,
            occ_dict={1: [1] * (stop - start)}, cam_int=cam_int,
            use_mask=False,
        )
        outputs.extend(frame[0] for frame in batch)
    if len(outputs) != len(image_paths):
        raise RuntimeError("Inference output count mismatch")
    return outputs


def save_manifests():
    manifests = build_split_manifests(DATA_ROOT / "sequenceFiles")
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        (MANIFEST_ROOT / f"3dpw_{split}_manifest.json").write_text(
            json.dumps(manifests[split], indent=2)
        )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "train_validation_overlap": sorted(set(manifests["train"]["sequences"]) & set(manifests["validation"]["sequences"])),
        "train_test_overlap": sorted(set(manifests["train"]["sequences"]) & set(manifests["test"]["sequences"])),
        "validation_test_overlap": sorted(set(manifests["validation"]["sequences"]) & set(manifests["test"]["sequences"])),
        "test_sequences_for_leakage_guard_only": manifests["test"]["sequences"],
    }
    (MANIFEST_ROOT / "3dpw_split_audit.json").write_text(json.dumps(audit, indent=2))
    return manifests


def build_sample(estimator, split, sequence, person_id):
    validate_cache_split(split)
    annotation = load_pickle(DATA_ROOT / "sequenceFiles" / split / f"{sequence}.pkl")
    frame_ids = np.asarray(annotation["img_frame_ids"], dtype=np.int64)
    valid_all = np.asarray(annotation["campose_valid"][person_id], dtype=bool)
    selected = np.flatnonzero(valid_all)[:WINDOW_LENGTH]
    if selected.size != WINDOW_LENGTH:
        raise RuntimeError(f"{sequence} lacks {WINDOW_LENGTH} valid frames")
    selected_frame_ids = frame_ids[selected]
    image_dir = DATA_ROOT / "imageFiles" / sequence
    image_paths = [str(image_dir / f"image_{index:05d}.jpg") for index in selected]
    first = cv2.imread(image_paths[0])
    if first is None:
        raise FileNotFoundError(image_paths[0])
    height, width = first.shape[:2]
    poses2d = np.asarray(annotation["poses2d"][person_id])
    bboxes = np.stack([
        bbox_from_pose2d(poses2d[index], width, height) for index in selected
    ])
    cam_int_np = np.asarray(annotation["cam_intrinsics"], dtype=np.float32)
    cam_int = torch.as_tensor(cam_int_np, device="cuda").unsqueeze(0)

    clean = run_sequence(estimator, image_paths, bboxes, cam_int)
    clean_tokens = np.stack([output["pose_token"] for output in clean]).astype(np.float32)
    clean_vertices = np.stack([output["pred_vertices"] for output in clean]).astype(np.float32)

    occluded_tokens, occluded_vertices, occluded_body, rectangle_meta = [], [], [], []
    with tempfile.TemporaryDirectory(prefix=f"dynav2_{split}_") as temp_dir:
        temp_dir = Path(temp_dir)
        for level_index, ratio in enumerate(OCCLUSION_LEVELS):
            paths, level_meta = [], []
            for local_index, (path, bbox) in enumerate(zip(image_paths, bboxes)):
                image = cv2.imread(path)
                seed = BASE_SEED + level_index * 100000 + int(selected_frame_ids[local_index])
                image_occ, meta = apply_bbox_rectangle_occlusion(
                    image, bbox, float(ratio), seed=seed, fill_value=0
                )
                output_path = temp_dir / f"level{level_index}_{local_index:05d}.png"
                cv2.imwrite(str(output_path), image_occ)
                paths.append(str(output_path)); level_meta.append(meta)
            outputs = run_sequence(estimator, paths, bboxes, cam_int)
            occluded_tokens.append(np.stack([x["pose_token"] for x in outputs]).astype(np.float32))
            occluded_vertices.append(np.stack([x["pred_vertices"] for x in outputs]).astype(np.float32))
            occluded_body.append(np.stack([x["body_pose_params"] for x in outputs]).astype(np.float32))
            rectangle_meta.append(level_meta)

    poses = np.asarray(annotation["poses"][person_id], dtype=np.float32)[selected]
    transl = np.asarray(annotation["trans"][person_id], dtype=np.float32)[selected]
    betas = np.asarray(annotation["betas"][person_id], dtype=np.float32)[:10]
    joints = np.asarray(annotation["jointPositions"][person_id], dtype=np.float32)[selected].reshape(-1, 24, 3)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "sequence": sequence,
        "person_id": int(person_id),
        "frame_id": selected_frame_ids.astype(np.int64),
        "local_frame_index": selected.astype(np.int64),
        "is_partial_sequence": np.asarray(True),
        "pose_token_clean": clean_tokens,
        "pose_token_occluded": np.stack(occluded_tokens),
        "visibility": np.full(WINDOW_LENGTH, -1, dtype=np.int8),
        "visibility_semantics": np.asarray("-1=unknown; real amodal visibility unavailable for 3DPW cache path"),
        "valid": valid_all[selected],
        "baseline_body_pose": np.stack([x["body_pose_params"] for x in clean]).astype(np.float32),
        "baseline_global_rot": np.stack([x["global_rot"] for x in clean]).astype(np.float32),
        "baseline_pred_vertices": clean_vertices,
        "occluded_body_pose": np.stack(occluded_body),
        "occluded_pred_vertices": np.stack(occluded_vertices),
        "gt_body_pose": poses[:, 3:72],
        "gt_global_orient": poses[:, :3],
        "gt_betas": np.repeat(betas[None], WINDOW_LENGTH, axis=0),
        "gt_transl": transl,
        "gt_joints3d": joints,
        "bbox": bboxes,
        "camera_intrinsics": np.repeat(cam_int_np[None], WINDOW_LENGTH, axis=0),
        "camera_extrinsics": np.asarray(annotation["cam_poses"], dtype=np.float32)[selected],
        "occlusion_ratio": OCCLUSION_LEVELS,
        "synthetic_visibility": np.repeat((1.0 - OCCLUSION_LEVELS)[:, None], WINDOW_LENGTH, axis=1).astype(np.float32),
        "synthetic_seed": np.asarray(BASE_SEED, dtype=np.int64),
    }
    validate_cache_payload(payload)
    if not np.isfinite(payload["pose_token_occluded"]).all():
        raise FloatingPointError("Occluded tokens are not finite")
    distances = np.linalg.norm(payload["pose_token_occluded"] - clean_tokens[None], axis=-1)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "split": split, "sequence": sequence, "person_id": person_id,
        "frame_ids": selected_frame_ids.tolist(), "is_partial_sequence": True,
        "window_length": WINDOW_LENGTH, "occlusion_levels": OCCLUSION_LEVELS.tolist(),
        "mean_token_l2_by_level": distances.mean(axis=1).tolist(),
        "min_token_l2": float(distances.min()), "max_token_l2": float(distances.max()),
        "all_finite": True, "rectangle_metadata": rectangle_meta,
    }
    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{sequence}_person{person_id}_16f.npz", **payload)
    (output_dir / f"{sequence}_person{person_id}_16f.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    manifests = save_manifests()
    cfg = OmegaConf.load(ROOT / "configs/body4d.yaml")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body.ckpt_path, device=torch.device("cuda"),
        mhr_path=cfg.sam_3d_body.mhr_path,
    )
    estimator = SAM3DBodyEstimator(
        model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None
    )
    summaries = {}
    for split, (sequence, person_id) in SAMPLES.items():
        if sequence not in manifests[split]["sequences"]:
            raise RuntimeError(f"{sequence} not in {split} manifest")
        summaries[split] = build_sample(estimator, split, sequence, person_id)
    (OUTPUT_ROOT / "sanity_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
