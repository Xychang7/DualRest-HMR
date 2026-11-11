"""Metadata-only validation of the frozen unified 30-fps temporal GT protocol."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def load_pickle(path):
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--equivalence-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    manifest = json.loads(Path(args.manifest).read_text())
    sources = {}
    counts = Counter()
    for row in manifest["instances"]:
        sequence, actor, frame = (str(row["sequence"]), int(row["source_actor_id"]),
                                  int(row["rgb_frame_idx"]))
        paths = sorted((root / "sequenceFiles").glob(f"*/{sequence}.pkl"))
        if len(paths) != 1:
            counts["missing_source_actor_frame"] += 1
            continue
        if sequence not in sources:
            sources[sequence] = load_pickle(paths[0])
        source = sources[sequence]
        poses = np.asarray(source["poses"])
        if actor < 0 or actor >= poses.shape[0] or frame < 0 or frame >= poses.shape[1]:
            counts["missing_source_actor_frame"] += 1
            continue
        if frame == 0 or frame == poses.shape[1] - 1:
            counts["rgb_sequence_boundary"] += 1
            continue
        images = [root / "imageFiles" / sequence / f"image_{value:05d}.jpg"
                  for value in (frame - 1, frame, frame + 1)]
        if not all(path.is_file() for path in images):
            counts["non_contiguous_rgb"] += 1
            continue
        sync = np.asarray(source["img_frame_ids"], dtype=np.int64)
        if len(sync) <= frame + 1:
            counts["missing_source_actor_frame"] += 1
            continue
        if int(sync[frame] - sync[frame - 1]) != 2 or int(sync[frame + 1] - sync[frame]) != 2:
            counts["irregular_timestamp"] += 1
            continue
        triple = np.arange(frame - 1, frame + 2)
        valid = np.asarray(source.get("campose_valid", np.ones(poses.shape[:2], bool)))[actor, triple]
        if (not bool(np.all(valid)) or not np.isfinite(poses[actor, triple]).all()
                or not np.isfinite(np.asarray(source["cam_poses"])[triple]).all()):
            counts["missing_source_actor_frame"] += 1
            continue
        counts["valid"] += 1
    equivalence = json.loads(Path(args.equivalence_audit).read_text())
    relative = equivalence["pelvis_centered_h36m_j14_mean_distance_mm"]
    total = len(manifest["instances"])
    report = {
        "state": "passed" if relative["max"] < 0.01 else "failed",
        "protocol": "sequenceFiles root-relative H36M J14 at continuous 30fps; OC centers only",
        "total_oc_center_instances": total,
        "rgb_sequence_boundary": counts["rgb_sequence_boundary"],
        "missing_source_actor_frame": counts["missing_source_actor_frame"],
        "non_contiguous_rgb": counts["non_contiguous_rgb"],
        "irregular_timestamp": counts["irregular_timestamp"],
        "theoretical_valid_before_timestamp": total - counts["rgb_sequence_boundary"]
            - counts["missing_source_actor_frame"] - counts["non_contiguous_rgb"],
        "final_valid_acceleration_triplets": counts["valid"],
        "root_relative_j14_difference_mm": relative,
    }
    if sum(counts.values()) != total:
        raise AssertionError("temporal center accounting is not exhaustive")
    if report["state"] != "passed":
        raise RuntimeError("Temporal GT Gate failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
