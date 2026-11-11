"""Train-clean-only camera failure mining from aligned precomputed diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED = (
    "sequence", "person_id", "frame_id", "condition", "pred_cam",
    "reprojection_error", "body_temporal_innovation", "visibility",
    "bbox_center_delta", "bbox_scale_delta", "mask_area_delta",
)


def validate_training_split(split):
    """Backward-compatible Phase-0 guard; still rejects validation/test."""
    normalized=str(split).lower()
    if normalized!='train':
        raise ValueError('camera failure mining is restricted to the 3DPW train split')
    return normalized


def validate_mining_contract(split, condition):
    split, condition = validate_training_split(split), str(condition).lower()
    if condition != "clean":
        raise ValueError("first-pass camera failure mining is restricted to clean condition")
    return split, condition


def _norm(value):
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def mine_records(records, split, condition="clean"):
    validate_mining_contract(split, condition)
    groups, seen = defaultdict(list), set()
    for item in records:
        if "raw_common_root_error" in item:
            raise ValueError("common-root diagnostics are forbidden in Camera Phase-1 mining")
        missing = [name for name in REQUIRED if name not in item]
        if missing:
            raise ValueError(f"mining record missing required fields: {missing}")
        if str(item["condition"]).lower() != "clean":
            raise ValueError("mining records must all come from clean condition")
        key = (str(item["sequence"]), int(item["person_id"]), int(item["frame_id"]))
        if key in seen:
            raise ValueError(f"duplicate person/frame {key}")
        seen.add(key); groups[key[:2]].append(item)
    candidates = []
    for (sequence, person), items in groups.items():
        items.sort(key=lambda x: int(x["frame_id"]))
        frames = np.asarray([int(x["frame_id"]) for x in items])
        for index in range(1, len(items) - 1):
            if frames[index] - frames[index - 1] != frames[index + 1] - frames[index]:
                continue
            prev, cur, nxt = items[index - 1], items[index], items[index + 1]
            cam = np.asarray(cur["pred_cam"], np.float64)
            neighbour_cam = .5 * (np.asarray(prev["pred_cam"], np.float64) + np.asarray(nxt["pred_cam"], np.float64))
            innovation = float(np.linalg.norm(cam - neighbour_cam))
            neighbour_reproj = max(float(prev["reprojection_error"]), float(nxt["reprojection_error"]))
            reproj_spike = float(cur["reprojection_error"]) - neighbour_reproj
            bbox_dynamics = _norm(cur["bbox_center_delta"]) + abs(float(cur["bbox_scale_delta"]))
            body = float(cur["body_temporal_innovation"])
            # Higher is better: large isolated camera/reprojection event, small
            # body and bbox innovations. This is diagnostic ranking, not a loss.
            score = max(reproj_spike, 0.0) * max(innovation, 0.0) / (1e-6 + body + bbox_dynamics)
            candidates.append({
                "sequence": sequence, "person_id": person,
                "frame_id": int(frames[index]),
                "window_frame_ids": frames[index - 1:index + 2].tolist(),
                "reprojection_error": float(cur["reprojection_error"]),
                "reprojection_spike": reproj_spike,
                "camera_temporal_innovation": innovation,
                "body_temporal_innovation": body,
                "visibility": int(cur["visibility"]),
                "bbox_dynamics": bbox_dynamics,
                "mask_area_delta": cur["mask_area_delta"],
                "score": score,
            })
    return sorted(candidates, key=lambda x: (x["score"], x["reprojection_spike"], x["camera_temporal_innovation"]), reverse=True)


def is_camera_specific_candidate(candidate, *, max_body=0.10, max_bbox=0.15,
                                 min_reprojection_spike=0.05,
                                 min_camera_innovation=0.20):
    """Fixed post-ranking gate; rejects body- or bbox-driven events."""
    return (
        float(candidate["body_temporal_innovation"]) <= max_body
        and float(candidate["bbox_dynamics"]) <= max_bbox
        and float(candidate["reprojection_spike"]) >= min_reprojection_spike
        and float(candidate["camera_temporal_innovation"]) >= min_camera_innovation
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_mining_contract(args.split, args.condition)
    payload = json.loads(args.input.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    result = {"split": "train", "condition": "clean", "source": str(args.input), "candidates": mine_records(records, args.split, args.condition)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

