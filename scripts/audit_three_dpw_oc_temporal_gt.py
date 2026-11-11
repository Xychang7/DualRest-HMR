"""Read-only Gate: OC-JSON GT versus full sequenceFiles GT in camera space."""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import smplx


J14 = np.asarray((1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16))


def _load_pickle(path):
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _betas_for(source, actor, frames):
    value = np.asarray(source["betas"])
    selected = value[actor]
    if selected.ndim == 1:
        return np.repeat(selected[None, :10], len(frames), axis=0)
    return selected[np.asarray(frames), :10]


def _model_vertices(model, poses, betas, transl):
    pose = torch.as_tensor(poses, dtype=torch.float32)
    with torch.no_grad():
        output = model(global_orient=pose[:, :3], body_pose=pose[:, 3:72],
                       betas=torch.as_tensor(betas, dtype=torch.float32),
                       transl=torch.as_tensor(transl, dtype=torch.float32))
    return output.vertices.detach().cpu().numpy().astype(np.float64)


def _stats(values):
    value = np.asarray(values, dtype=np.float64)
    return {"count": int(value.size), "mean": float(value.mean()),
            "median": float(np.median(value)), "p95": float(np.percentile(value, 95)),
            "max": float(value.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--smpl-model-dir", required=True)
    parser.add_argument("--corrected-regressor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    manifest = json.loads(Path(args.manifest).read_text())
    raw = json.loads((root / "3DPW_oc.json").read_text())
    annotations = {int(row["id"]): row for row in raw["annotations"]}
    reg = np.load(args.corrected_regressor).astype(np.float64)
    models = {gender: smplx.SMPL(
        model_path=str(Path(args.smpl_model_dir) / f"SMPL_{gender.upper()}.pkl"),
        batch_size=1).eval() for gender in ("male", "female")}
    native = models["male"].J_regressor[0].detach().cpu().numpy().astype(np.float64)

    grouped = defaultdict(list)
    for row in manifest["instances"]:
        grouped[(row["sequence"], int(row["source_actor_id"]), row["gender"])].append(row)
    vertex_frame_mean_mm, vertex_frame_median_mm = [], []
    vertex_global_max_mm, j14_frame_mean_mm, centered_j14_frame_mean_mm = 0.0, [], []
    centered_vertex_frame_mean_mm = []
    signed_offsets_mm = []
    sync_gaps = defaultdict(int)
    processed = 0
    started = time.time()
    for (sequence, actor, gender), rows in sorted(grouped.items()):
        paths = list((root / "sequenceFiles").glob(f"*/{sequence}.pkl"))
        if len(paths) != 1:
            raise ValueError(f"sequenceFiles lookup failed for {sequence}: {paths}")
        source = _load_pickle(paths[0])
        rows.sort(key=lambda row: int(row["rgb_frame_idx"]))
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            frames = np.asarray([int(row["rgb_frame_idx"]) for row in chunk])
            annotation_chunk = [annotations[int(row["annotation_id"])] for row in chunk]
            json_pose = np.asarray([row["smpl_param"]["pose"] for row in annotation_chunk])
            json_betas = np.asarray([row["smpl_param"]["shape"] for row in annotation_chunk])
            json_trans = np.asarray([row["smpl_param"]["trans"] for row in annotation_chunk])
            oc_vertices = _model_vertices(models[gender], json_pose, json_betas, json_trans)

            seq_pose = np.asarray(source["poses"])[actor, frames]
            seq_trans = np.asarray(source["trans"])[actor, frames]
            seq_betas = _betas_for(source, actor, frames)
            world_vertices = _model_vertices(models[gender], seq_pose, seq_betas, seq_trans)
            cameras = np.asarray(source["cam_poses"], dtype=np.float64)[frames]
            seq_vertices = np.einsum("nij,nvj->nvi", cameras[:, :3, :3], world_vertices)
            seq_vertices += cameras[:, None, :3, 3]

            delta = oc_vertices - seq_vertices
            vertex_distance = np.linalg.norm(delta, axis=2) * 1000.0
            vertex_frame_mean_mm.extend(vertex_distance.mean(axis=1))
            vertex_frame_median_mm.extend(np.median(vertex_distance, axis=1))
            vertex_global_max_mm = max(vertex_global_max_mm, float(vertex_distance.max()))
            signed_offsets_mm.extend(delta.mean(axis=1) * 1000.0)

            oc_native = np.einsum("v,nvc->nc", native, oc_vertices)
            seq_native = np.einsum("v,nvc->nc", native, seq_vertices)
            centered_vertex_distance = np.linalg.norm(
                (oc_vertices - oc_native[:, None]) - (seq_vertices - seq_native[:, None]), axis=2)
            centered_vertex_frame_mean_mm.extend(centered_vertex_distance.mean(axis=1) * 1000.0)
            oc_h36m = np.einsum("jv,nvc->njc", reg, oc_vertices)
            seq_h36m = np.einsum("jv,nvc->njc", reg, seq_vertices)
            j14_frame_mean_mm.extend(np.linalg.norm(oc_h36m[:, J14] - seq_h36m[:, J14], axis=2).mean(axis=1) * 1000.0)
            oc_j14 = (oc_h36m - oc_h36m[:, :1])[:, J14]
            seq_j14 = (seq_h36m - seq_h36m[:, :1])[:, J14]
            centered_j14_frame_mean_mm.extend(np.linalg.norm(oc_j14 - seq_j14, axis=2).mean(axis=1) * 1000.0)

            sync = np.asarray(source["img_frame_ids"], dtype=np.int64)
            for frame in frames:
                if frame > 0 and int(sync[frame] - sync[frame - 1]) != 2:
                    sync_gaps[f"delta_{int(sync[frame] - sync[frame - 1])}"] += 1
            processed += len(chunk)

    offsets = np.asarray(signed_offsets_mm)
    report = {
        "instances": processed,
        "elapsed_seconds": time.time() - started,
        "absolute_vertex_distance_mm": {**_stats(vertex_frame_mean_mm),
                                         "frame_median_stats": _stats(vertex_frame_median_mm),
                                         "global_vertex_max": vertex_global_max_mm},
        "absolute_h36m_j14_mean_distance_mm": _stats(j14_frame_mean_mm),
        "native_pelvis_centered_vertex_mean_distance_mm": _stats(centered_vertex_frame_mean_mm),
        "pelvis_centered_h36m_j14_mean_distance_mm": _stats(centered_j14_frame_mean_mm),
        "signed_mesh_offset_mm": {"mean_xyz": offsets.mean(axis=0).tolist(),
                                  "std_xyz": offsets.std(axis=0).tolist(),
                                  "min_xyz": offsets.min(axis=0).tolist(),
                                  "max_xyz": offsets.max(axis=0).tolist()},
        "mocap_sync_non_two_gaps_at_oc_frames": dict(sorted(sync_gaps.items())),
        "equivalent_threshold": "not decided by script; report raw evidence",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
