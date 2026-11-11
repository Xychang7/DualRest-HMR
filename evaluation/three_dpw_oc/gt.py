"""JOTR-style gendered OC-JSON SMPL GT and corrected H36M resource guard."""

from pathlib import Path
import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
import smplx


def load_corrected_h36m_regressor(path: str | Path) -> np.ndarray:
    """Never silently substitute the legacy J_regressor_h36m.npy."""
    path = Path(path)
    if path.is_dir():
        path = path / "J_regressor_h36m_correct.npy"
    if path.name != "J_regressor_h36m_correct.npy" or not path.is_file():
        raise FileNotFoundError(f"Official J_regressor_h36m_correct.npy is required: {path}")
    regressor = np.load(path, allow_pickle=False)
    if regressor.shape != (17, 6890) or not np.isfinite(regressor).all():
        raise ValueError(f"Invalid corrected H36M joint regressor: {path}/{regressor.shape}")
    return regressor.astype(np.float64, copy=False)


class GenderedSMPLGT:
    """Generate camera-space 6890-vertex GT from each OC JSON annotation."""

    def __init__(self, smpl_directory: str | Path):
        directory = Path(smpl_directory)
        self.models = {}
        for gender in ("male", "female", "neutral"):
            path = directory / f"SMPL_{gender.upper()}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing gendered official SMPL model: {path}")
            self.models[gender] = smplx.SMPL(model_path=str(path), batch_size=1).eval()
            for parameter in self.models[gender].parameters():
                parameter.requires_grad_(False)
        self.native_pelvis_regressor = self.models["neutral"].J_regressor.detach().cpu().numpy()
        if self.native_pelvis_regressor.shape != (24, 6890):
            raise ValueError("SMPL native joint regressor has unexpected topology")

    def vertices(self, annotation: dict) -> np.ndarray:
        params = annotation["smpl_param"]
        gender = str(params["gender"]).lower()
        if gender not in self.models:
            raise ValueError(f"Unsupported SMPL gender: {gender}")
        pose = np.asarray(params["pose"], dtype=np.float32)
        shape = np.asarray(params["shape"], dtype=np.float32)
        trans = np.asarray(params["trans"], dtype=np.float32)
        if pose.shape != (72,) or shape.shape != (10,) or trans.shape != (3,):
            raise ValueError("OC annotation requires pose72/shape10/trans3")
        if not (np.isfinite(pose).all() and np.isfinite(shape).all() and np.isfinite(trans).all()):
            raise ValueError("non-finite OC SMPL GT parameters")
        vertices = self.vertices_batch(gender, pose.reshape(1, 72),
                                       shape.reshape(1, 10), trans.reshape(1, 3))[0]
        if vertices.shape != (6890, 3) or not np.isfinite(vertices).all():
            raise ValueError("invalid gendered SMPL GT mesh")
        return vertices.astype(np.float64, copy=False)

    def vertices_batch(self, gender, pose, shape, trans) -> np.ndarray:
        gender = str(gender).lower()
        if gender not in self.models:
            raise ValueError(f"Unsupported SMPL gender: {gender}")
        pose = np.asarray(pose, dtype=np.float32)
        shape = np.asarray(shape, dtype=np.float32)
        trans = np.asarray(trans, dtype=np.float32)
        count = len(pose)
        if pose.shape != (count, 72) or shape.shape != (count, 10) or trans.shape != (count, 3):
            raise ValueError("batched SMPL parameters require pose72/shape10/trans3")
        if not (np.isfinite(pose).all() and np.isfinite(shape).all() and np.isfinite(trans).all()):
            raise ValueError("non-finite batched SMPL parameters")
        with torch.no_grad():
            pose_tensor = torch.from_numpy(pose)
            output = self.models[gender](global_orient=pose_tensor[:, :3],
                                         body_pose=pose_tensor[:, 3:],
                                         betas=torch.from_numpy(shape),
                                         transl=torch.from_numpy(trans))
        vertices = output.vertices.detach().cpu().numpy()
        if vertices.shape != (count, 6890, 3) or not np.isfinite(vertices).all():
            raise ValueError("invalid batched gendered SMPL GT mesh")
        return vertices.astype(np.float64, copy=False)


def build_oc_gt_meshes(dataset_root: str | Path, instances, builder: GenderedSMPLGT) -> dict:
    """Read GT exclusively from OC JSON annotation IDs; no sequenceFiles fallback."""
    root = Path(dataset_root)
    dataset = json.loads((root / "3DPW_oc.json").read_text(encoding="utf-8"))
    annotations = {int(row["id"]): row for row in dataset["annotations"]}
    images = {int(row["id"]): row for row in dataset["images"]}
    result = {}
    for instance in instances:
        ann_id = int(instance["annotation_id"])
        if ann_id not in annotations:
            raise ValueError(f"OC annotation ID missing: {ann_id}")
        annotation = annotations[ann_id]
        image = images[int(annotation["image_id"])]
        key = (str(instance["sequence"]), int(instance["rgb_frame_idx"]),
               int(instance["source_actor_id"]))
        if str(image["sequence"]) != key[0] or int(image["frame_idx"]) != key[1]:
            raise ValueError(f"OC annotation/image/manifest RGB frame mismatch: {key}")
        if "person_id" in instance and int(annotation["person_id"]) != int(instance["person_id"]):
            raise ValueError(f"OC annotation/person/manifest mismatch: {key}")
        if key in result:
            raise ValueError(f"duplicate OC GT canonical key: {key}")
        result[key] = builder.vertices(annotation)
    return result


def _load_pickle(path):
    with Path(path).open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _person_betas(source, actor, frames):
    value = np.asarray(source["betas"])[actor]
    if value.ndim == 1:
        return np.repeat(value[None, :10], len(frames), axis=0)
    return value[np.asarray(frames), :10]


def build_sequence_temporal_gt(dataset_root: str | Path, instances,
                               builder: GenderedSMPLGT, batch_size=128):
    """Build full-neighbour root-relative-compatible GT from sequenceFiles.

    Absolute translations deliberately remain in the returned meshes, but the
    acceleration metric always regresses and pelvis-centres each frame before
    differencing.  OC JSON is used only to define center keys.
    """
    root = Path(dataset_root)
    sources = {}
    status = {}
    needed = defaultdict(set)
    genders = {}
    for row in instances:
        sequence, frame, actor = (str(row["sequence"]), int(row["rgb_frame_idx"]),
                                  int(row["source_actor_id"]))
        key = (sequence, frame, actor)
        matches = sorted((root / "sequenceFiles").glob(f"*/{sequence}.pkl"))
        if len(matches) != 1:
            status[key] = "missing_source_actor_frame"
            continue
        if sequence not in sources:
            sources[sequence] = _load_pickle(matches[0])
        source = sources[sequence]
        poses = np.asarray(source["poses"])
        if actor < 0 or actor >= poses.shape[0] or frame <= 0 or frame >= poses.shape[1] - 1:
            status[key] = "rgb_sequence_boundary" if 0 <= actor < poses.shape[0] and (frame == 0 or frame == poses.shape[1] - 1) else "missing_source_actor_frame"
            continue
        image_paths = [root / "imageFiles" / sequence / f"image_{index:05d}.jpg"
                       for index in (frame - 1, frame, frame + 1)]
        if not all(path.is_file() for path in image_paths):
            status[key] = "non_contiguous_rgb"
            continue
        sync = np.asarray(source["img_frame_ids"], dtype=np.int64)
        if len(sync) <= frame + 1:
            status[key] = "missing_source_actor_frame"
            continue
        if int(sync[frame] - sync[frame - 1]) != 2 or int(sync[frame + 1] - sync[frame]) != 2:
            status[key] = "irregular_timestamp"
            continue
        triple = np.arange(frame - 1, frame + 2)
        cameras = np.asarray(source["cam_poses"])[triple]
        valid = np.asarray(source.get("campose_valid", np.ones(poses.shape[:2], bool)))[actor, triple]
        if not bool(np.all(valid)) or not np.isfinite(poses[actor, triple]).all() or not np.isfinite(cameras).all():
            status[key] = "missing_source_actor_frame"
            continue
        status[key] = "valid"
        group = (sequence, actor, str(row["gender"]).lower())
        needed[group].update(int(index) for index in triple)
        genders[(sequence, actor)] = group[2]

    meshes = {}
    for (sequence, actor, gender), frame_set in sorted(needed.items()):
        source = sources[sequence]
        source_gender = str(np.asarray(source.get("genders", [gender]))[actor]).lower()
        normalized_source_gender = {"m": "male", "f": "female"}.get(source_gender, source_gender)
        if normalized_source_gender != gender:
            raise ValueError(f"gender mismatch for {sequence}/actor{actor}: {gender}/{source_gender}")
        frames = np.asarray(sorted(frame_set), dtype=np.int64)
        for start in range(0, len(frames), batch_size):
            selected = frames[start:start + batch_size]
            pose = np.asarray(source["poses"])[actor, selected]
            trans = np.asarray(source["trans"])[actor, selected]
            beta = _person_betas(source, actor, selected)
            world = builder.vertices_batch(gender, pose, beta, trans)
            camera = np.asarray(source["cam_poses"], dtype=np.float64)[selected]
            camera_mesh = np.einsum("nij,nvj->nvi", camera[:, :3, :3], world)
            camera_mesh += camera[:, None, :3, 3]
            for frame, vertices in zip(selected, camera_mesh):
                meshes[(sequence, int(frame), actor)] = vertices
    counts = Counter(status.values())
    total = len(instances)
    unavailable = counts["missing_source_actor_frame"] + counts["non_contiguous_rgb"]
    audit = {"total_oc_centers": total,
             "rgb_sequence_boundary": counts["rgb_sequence_boundary"],
             "missing_source_actor_frame": counts["missing_source_actor_frame"],
             "non_contiguous_rgb": counts["non_contiguous_rgb"],
             "irregular_timestamp": counts["irregular_timestamp"],
             "theoretical_valid_before_timestamp": total - counts["rgb_sequence_boundary"] - unavailable,
             "valid_acceleration_triplets": counts["valid"]}
    return meshes, status, audit
