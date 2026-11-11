"""Model-neutral paired SAM/Dyna prediction adapter for 3DPW-OC."""

import numpy as np


def _vertices(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (6890, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (T,6890,3) standard SMPL vertices")
    extent = np.ptp(array, axis=1).max(axis=1)
    if np.any(extent < 0.5) or np.any(extent > 3.0):
        raise ValueError(f"{name} is not plausibly meter-scale SMPL geometry")
    return array


def make_paired_prediction_records(sequence, rgb_frame_indices, source_actor_id,
                                   sam_vertices, dyna_vertices, pred_track_id=None):
    """Create key-identical records after one shared converter contract."""
    frames = np.asarray(rgb_frame_indices, dtype=np.int64)
    sam = _vertices(sam_vertices, "SAM vertices")
    dyna = _vertices(dyna_vertices, "Dyna vertices")
    if len(frames) != len(sam) or sam.shape != dyna.shape:
        raise ValueError("paired frames/SAM/Dyna lengths or topology differ")
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("RGB frame indices must be unique and strictly increasing")
    actor = int(source_actor_id)
    track = actor if pred_track_id is None else int(pred_track_id)
    if actor < 0 or track < 0:
        raise ValueError("actor/track IDs must be non-negative")
    shared = [{"sequence": str(sequence), "rgb_frame_idx": int(frame),
               "pred_track_id": track, "matched_actor_id": actor,
               "valid": True} for frame in frames]
    sam_records = [{**row, "vertices": vertices} for row, vertices in zip(shared, sam)]
    dyna_records = [{**row, "vertices": vertices} for row, vertices in zip(shared, dyna)]
    sam_keys = [(row["sequence"], row["rgb_frame_idx"], row["matched_actor_id"])
                for row in sam_records]
    dyna_keys = [(row["sequence"], row["rgb_frame_idx"], row["matched_actor_id"])
                 for row in dyna_records]
    if sam_keys != dyna_keys:
        raise AssertionError("paired SAM/Dyna canonical keys differ")
    return sam_records, dyna_records


def paired_intersection_manifest(manifest, sam_records, dyna_records):
    sam_keys = {(row["sequence"], int(row["rgb_frame_idx"]),
                 int(row["matched_actor_id"])) for row in sam_records if row["valid"]}
    dyna_keys = {(row["sequence"], int(row["rgb_frame_idx"]),
                  int(row["matched_actor_id"])) for row in dyna_records if row["valid"]}
    common = sam_keys & dyna_keys
    selected = [row for row in manifest["instances"] if
                (row["sequence"], int(row["rgb_frame_idx"]),
                 int(row["source_actor_id"])) in common]
    expected = len(manifest["instances"])
    return {**{key: value for key, value in manifest.items() if key != "instances"},
            "instances": selected}, {
                "expected_oc_instances": expected,
                "matched_intersection_instances": len(selected),
                "missing_prediction_instances": expected - len(selected),
                "sam_only_keys": len(sam_keys - dyna_keys),
                "dyna_only_keys": len(dyna_keys - sam_keys),
                "duplicate_keys": 0,
            }
