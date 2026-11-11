"""Deterministic OC instance manifest; 60 Hz mocap IDs are metadata only."""

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


SCHEMA = "three_dpw_oc_manifest_v1"


def _source_sequence_file(root: Path, sequence: str) -> Path:
    matches = sorted((root / "sequenceFiles").glob(f"*/{sequence}.pkl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one sequenceFiles record for {sequence}, found {matches}")
    return matches[0]


def _load_source(root: Path, sequence: str) -> dict:
    with _source_sequence_file(root, sequence).open("rb") as file:
        source = pickle.load(file, encoding="latin1")
    poses = np.asarray(source["poses"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[2] != 72:
        raise ValueError(f"Invalid source actor poses for {sequence}: {poses.shape}")
    return source


def build_manifest(root: str | Path) -> dict:
    root = Path(root)
    annotation_file = root / "3DPW_oc.json"
    raw = annotation_file.read_bytes()
    dataset = json.loads(raw)
    images = {int(image["id"]): image for image in dataset["images"]}
    if len(images) != len(dataset["images"]):
        raise ValueError("duplicate image ID")
    sources = {}
    rgb_frames = set()
    for image in images.values():
        sequence = str(image["sequence"])
        frame = int(image["frame_idx"])
        filename = str(image["file_name"])
        if filename != f"image_{frame:05d}.jpg":
            raise ValueError(f"RGB frame/file mismatch: {sequence}/{frame}/{filename}")
        rgb_key = (sequence, frame)
        if rgb_key in rgb_frames:
            raise ValueError(f"duplicate sequence/RGB frame: {rgb_key}")
        rgb_frames.add(rgb_key)
        if not (root / "imageFiles" / sequence / filename).is_file():
            raise ValueError(f"Missing referenced RGB image: {sequence}/{filename}")
        if sequence not in sources:
            sources[sequence] = _load_source(root, sequence)
        if frame < 0 or frame >= np.asarray(sources[sequence]["poses"]).shape[1]:
            raise ValueError(f"RGB frame outside source timeline: {rgb_key}")
    for sequence in {key[0] for key in rgb_frames}:
        sequence_frames = sorted(frame for name, frame in rgb_frames if name == sequence)
        if sequence_frames != list(range(len(sequence_frames))):
            raise ValueError(f"non-contiguous RGB image timeline: {sequence}")
    keys = set()
    person_actor = {}
    instances = []
    for annotation in sorted(dataset["annotations"], key=lambda row: int(row["id"])):
        image = images[int(annotation["image_id"])]
        sequence = str(image["sequence"])
        frame = int(image["frame_idx"])
        filename = str(image["file_name"])
        if filename != f"image_{frame:05d}.jpg":
            raise ValueError(f"RGB frame/file mismatch: {sequence}/{frame}/{filename}")
        if not (root / "imageFiles" / sequence / filename).is_file():
            raise ValueError(f"Missing referenced RGB image: {sequence}/{filename}")
        source = sources[sequence]
        source_poses = np.asarray(source["poses"], dtype=np.float64)
        if frame < 0 or frame >= source_poses.shape[1]:
            raise ValueError(f"RGB frame outside source actor timeline: {sequence}/{frame}")
        smpl = annotation["smpl_param"]
        pose = np.asarray(smpl["pose"], dtype=np.float64)
        shape = np.asarray(smpl["shape"], dtype=np.float64)
        trans = np.asarray(smpl["trans"], dtype=np.float64)
        if pose.shape != (72,) or shape.shape != (10,) or trans.shape != (3,):
            raise ValueError(f"Invalid annotation SMPL parameters: {annotation['id']}")
        errors = np.linalg.norm(source_poses[:, frame, 3:] - pose[None, 3:], axis=1)
        actor = int(np.argmin(errors))
        if errors[actor] > 1e-3 or (len(errors) > 1 and np.partition(errors, 1)[1] < 1e-2):
            raise ValueError(f"ambiguous source actor mapping: {sequence}/{frame}/{annotation['id']}: {errors}")
        person = int(annotation["person_id"])
        mapping_key = (sequence, person)
        if mapping_key in person_actor and person_actor[mapping_key] != actor:
            raise ValueError(f"inconsistent person/source actor mapping: {mapping_key}")
        person_actor[mapping_key] = actor
        key = (sequence, frame, actor)
        if key in keys:
            raise ValueError(f"duplicate canonical key: {key}")
        keys.add(key)
        gender = str(smpl["gender"]).lower()
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unknown SMPL gender: {gender}")
        sync = np.asarray(source["img_frame_ids"])
        instances.append({"sequence": sequence, "rgb_frame_idx": frame,
                          "source_actor_id": actor, "annotation_id": int(annotation["id"]),
                          "person_id": person, "gender": gender, "oc": True,
                          "image_file": filename, "mocap_sync_id": int(sync[frame])})
    instances.sort(key=lambda row: (row["sequence"], row["rgb_frame_idx"],
                                    row["source_actor_id"], row["annotation_id"]))
    return {"schema": SCHEMA, "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "image_records": len(images), "image_sequences": len({key[0] for key in rgb_frames}),
            "oc_instances": len(instances),
            "oc_sequences": len({row["sequence"] for row in instances}),
            "instances": instances}


def write_manifest(manifest: dict, destination: str | Path) -> None:
    """Persist a deterministic, atomic manifest without changing the source JSON."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("invalid OC manifest schema")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                     prefix=destination.name + ".", suffix=".tmp",
                                     delete=False) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)


def validate_predictions(records) -> dict:
    """Fail closed on metadata, duplicate keys, units and invalid meshes."""
    indexed = {}
    last_frame = {}
    required = ("sequence", "rgb_frame_idx", "pred_track_id", "matched_actor_id",
                "vertices", "valid")
    for record in records:
        for field in required:
            if field not in record:
                raise ValueError(f"Missing prediction field: {field}")
        key = (str(record["sequence"]), int(record["rgb_frame_idx"]),
               int(record["matched_actor_id"]))
        if key in indexed:
            raise ValueError(f"duplicate prediction canonical key: {key}")
        track = (key[0], key[2])
        if track in last_frame and key[1] <= last_frame[track]:
            raise ValueError(f"prediction frame order must increase within track: {key}")
        last_frame[track] = key[1]
        if key[1] < 0 or key[2] < 0 or int(record["pred_track_id"]) < 0:
            raise ValueError(f"invalid prediction frame/actor: {key}")
        vertices = np.asarray(record["vertices"])
        if vertices.shape != (6890, 3):
            raise ValueError(f"Prediction vertices must be (6890,3) meters: {key}/{vertices.shape}")
        if bool(record["valid"]) and not np.isfinite(vertices).all():
            raise ValueError(f"non-finite prediction vertices: {key}")
        indexed[key] = {"vertices": vertices.astype(np.float64, copy=False),
                        "valid": bool(record["valid"])}
    return indexed
