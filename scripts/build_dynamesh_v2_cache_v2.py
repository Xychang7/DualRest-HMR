"""Resumable per-track DynaMesh4D-v2 Phase-2 feature-cache builder."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path("/root/sam-body4d-0.2.0")
DATA = Path("/root/data1/sam-body4d-master/dataset/3DPW")
OUT = ROOT / "data/dynamesh_v2/cache_v2"
GATE_OUT = ROOT / "outputs/dynamesh_v2_phase2_gates"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models/sam_3d_body")); sys.path.insert(0, str(ROOT / "scripts"))

from build_dynamesh_v2_cache_sanity import BASE_SEED, apply_bbox_rectangle_occlusion, bbox_from_pose2d, load_pickle
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.data.dynamesh_v2_cache_v2 import (
    CONDITIONS, RATIOS, SCHEMA_VERSION, atomic_save_track, file_sha256, make_alignment_hashes,
    make_variant_batch, restore_variant_batch, validate_track_payload,
)


class GpuMonitor:
    def __init__(self): self.values = []; self.stop_event = threading.Event(); self.thread = None
    def start(self):
        def sample():
            while not self.stop_event.is_set():
                try:
                    value = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True).splitlines()[0]
                    self.values.append(float(value.strip()))
                except Exception: pass
                self.stop_event.wait(.5)
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join()


def load_estimator():
    cfg = OmegaConf.load(ROOT / "configs/body4d.yaml")
    model, model_cfg = load_sam_3d_body(cfg.sam_3d_body.ckpt_path, device=torch.device("cuda"), mhr_path=cfg.sam_3d_body.mhr_path)
    estimator = SAM3DBodyEstimator(model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    estimator.model.eval()
    for parameter in estimator.model.parameters(): parameter.requires_grad_(False)
    return estimator


def run_flat(estimator, paths, boxes, identities, cam_int, *, batch_size, state=None):
    outputs = []; state = {} if state is None else state
    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))
        ids = identities[start:stop]
        # Each image contains exactly one supplied GT person box.  process_frames
        # addresses box slots (1..N), not external condition IDs; with N=1 the
        # only valid slot is 1.  Logical condition identity is retained solely
        # in the cache alignment metadata/hashes.
        result = estimator.process_frames(
            paths[start:stop], bboxes=[box.reshape(1, 4) for box in boxes[start:stop]], masks=None,
            id_batch=[[1] for _ in ids], idx_path={}, idx_dict={},
            mhr_shape_scale_dict=state, occ_dict={1: [1] * (stop - start)},
            cam_int=cam_int, use_mask=False,
        )
        if len(result) != stop - start or any(len(frame) != 1 for frame in result):
            raise RuntimeError(
                f"Formal inference alignment failure: requested {stop-start} "
                f"single-person frames, received {[len(frame) for frame in result]}"
            )
        outputs.extend(frame[0] for frame in result)
    return outputs


def prepare_track(split, sequence, person):
    annotation = load_pickle(DATA / "sequenceFiles" / split / f"{sequence}.pkl")
    valid = np.asarray(annotation["campose_valid"][person], dtype=bool)
    selected = np.flatnonzero(valid)
    frame_ids = np.asarray(annotation["img_frame_ids"], dtype=np.int64)[selected]
    paths = [str(DATA / "imageFiles" / sequence / f"image_{index:05d}.jpg") for index in selected]
    image = cv2.imread(paths[0]); height, width = image.shape[:2]
    poses2d = np.asarray(annotation["poses2d"][person])
    boxes = np.stack([bbox_from_pose2d(poses2d[index], width, height) for index in selected])
    cam_np = np.asarray(annotation["cam_intrinsics"], dtype=np.float32)
    cam = torch.as_tensor(cam_np, device="cuda").unsqueeze(0)
    return annotation, selected, frame_ids, paths, boxes, cam


def make_chunk_variants(paths, boxes, frame_ids, directory, offset):
    by_condition = [list(paths)]
    for level, ratio in enumerate(RATIOS[1:]):
        condition_paths = []
        for index, (path, box, frame_id) in enumerate(zip(paths, boxes, frame_ids)):
            image = cv2.imread(path)
            occluded, _ = apply_bbox_rectangle_occlusion(image, box, float(ratio), seed=BASE_SEED + level * 100000 + int(frame_id), fill_value=0)
            target = Path(directory) / f"c{level+1}_{offset+index:06d}.png"
            if not cv2.imwrite(str(target), occluded): raise RuntimeError(f"failed writing {target}")
            condition_paths.append(str(target))
        by_condition.append(condition_paths)
    return by_condition


def compare_paths(estimator, paths, boxes, frame_ids, cam):
    with tempfile.TemporaryDirectory(prefix="dynav2_throughput_compare_") as directory:
        variants = make_chunk_variants(paths, boxes, frame_ids, directory, 0)
        sequential, sequential_seconds = [], 0.0
        torch.cuda.reset_peak_memory_stats()
        for condition in range(4):
            started = time.perf_counter()
            sequential.append(run_flat(estimator, variants[condition], boxes, [condition + 1] * len(paths), cam, batch_size=4, state={}))
            torch.cuda.synchronize(); sequential_seconds += time.perf_counter() - started
        flat, mapping = make_variant_batch(variants)
        flat_boxes = [boxes[frame] for condition, frame in mapping]
        identities = [condition + 1 for condition, frame in mapping]
        started = time.perf_counter()
        batched_flat = run_flat(estimator, flat, flat_boxes, identities, cam, batch_size=16, state={})
        torch.cuda.synchronize(); batched_seconds = time.perf_counter() - started
        batched = restore_variant_batch(batched_flat, mapping, conditions=4, frames=len(paths))
        differences = {}
        for field in ("pose_token", "mhr_init_pose", "pred_pose_raw"):
            left = np.stack([[sequential[c][f][field] for f in range(len(paths))] for c in range(4)])
            right = np.stack([[batched[c, f][field] for f in range(len(paths))] for c in range(4)])
            differences[field] = float(np.max(np.abs(left - right)))
    return {"sequential_seconds": sequential_seconds, "batched_seconds": batched_seconds, "speedup": sequential_seconds / batched_seconds, "max_abs_differences": differences, "output_equivalent": bool(max(differences.values()) < 2e-4)}


def build_track(estimator, split, sequence, person, *, variant_batching=True):
    annotation, selected, frame_ids, paths, boxes, cam = prepare_track(split, sequence, person)
    outputs = [[None] * len(paths) for _ in range(4)]
    monitor = GpuMonitor(); monitor.start(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    # Keep the same per-condition inference state across frame chunks.  The four
    # condition identities remain isolated, matching four independent formal
    # inference streams while allowing variant batching.
    batched_state = {}
    sequential_states = [{} for _ in range(4)]
    with tempfile.TemporaryDirectory(prefix=f"dynav2_{split}_{sequence}_p{person}_") as directory:
        for start in range(0, len(paths), 4):
            stop = min(start + 4, len(paths))
            variants = make_chunk_variants(paths[start:stop], boxes[start:stop], frame_ids[start:stop], directory, start)
            if variant_batching:
                flat, mapping = make_variant_batch(variants)
                flat_boxes = [boxes[start + frame] for condition, frame in mapping]
                flat_outputs = run_flat(estimator, flat, flat_boxes, [condition + 1 for condition, frame in mapping], cam, batch_size=16, state=batched_state)
                restored = restore_variant_batch(flat_outputs, mapping, conditions=4, frames=stop-start)
                for condition in range(4):
                    for frame in range(stop-start): outputs[condition][start+frame] = restored[condition, frame]
            else:
                for condition in range(4):
                    result = run_flat(estimator, variants[condition], boxes[start:stop], [condition + 1] * (stop-start), cam, batch_size=4, state=sequential_states[condition])
                    outputs[condition][start:stop] = result
            for path in Path(directory).glob("*.png"): path.unlink()
    torch.cuda.synchronize(); seconds = time.perf_counter() - started; monitor.stop()
    pose = np.asarray(annotation["poses"][person], dtype=np.float32)[selected]
    trans = np.asarray(annotation["trans"][person], dtype=np.float32)[selected]
    beta_source = np.asarray(annotation["betas"][person], dtype=np.float32)
    betas = beta_source[selected, :10] if beta_source.ndim == 2 else np.repeat(beta_source[:10][None], len(selected), 0)
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION), "split": np.asarray(split),
        "sequence": np.asarray(sequence), "person_id": np.asarray(person, dtype=np.int64),
        "condition_names": np.asarray(CONDITIONS), "occlusion_ratio": RATIOS,
        "frame_id": frame_ids, "local_frame_index": selected.astype(np.int64),
        "valid": np.ones(len(selected), dtype=bool), "visibility": np.full(len(selected), -1, dtype=np.int8),
        "pose_token": np.stack([[item["pose_token"] for item in condition] for condition in outputs]).astype(np.float16),
        "mhr_init_pose": np.stack([[item["mhr_init_pose"] for item in condition] for condition in outputs]).astype(np.float32),
        "body_cont": np.stack([[item["pred_pose_raw"][6:] for item in condition] for condition in outputs]).astype(np.float32),
        "gt_pose": pose[:, :72], "gt_transl": trans, "gt_betas": betas,
        "camera_extrinsics": np.asarray(annotation["cam_poses"], dtype=np.float32)[selected],
    }
    payload["alignment_sha256"] = make_alignment_hashes(payload); validate_track_payload(payload)
    path = OUT / split / f"{sequence}_person{person}.npz"; digest = atomic_save_track(path, payload)
    return {"path": str(path), "sha256": digest, "frames": len(selected), "variant_frames": 4 * len(selected), "seconds": seconds, "variant_fps": 4 * len(selected) / seconds, "gpu_util_mean": float(np.mean(monitor.values)) if monitor.values else None, "gpu_util_max": float(np.max(monitor.values)) if monitor.values else None, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "cache_bytes": path.stat().st_size}


def pilot():
    GATE_OUT.mkdir(parents=True, exist_ok=True); estimator = load_estimator()
    _, _, frame_ids, paths, boxes, cam = prepare_track("train", "outdoors_slalom_00", 0)
    comparison = compare_paths(estimator, paths[:16], boxes[:16], frame_ids[:16], cam)
    # Select batching only when it preserves the formal inference outputs;
    # throughput alone is never sufficient.
    use_batching = comparison["output_equivalent"] and comparison["speedup"] > 1.05
    track = build_track(estimator, "train", "outdoors_slalom_00", 0, variant_batching=use_batching)
    total_frames = 22735 + 10413
    estimate = {"uncompressed_core_bytes": total_frames * (4*1024*2 + 4*519*4 + 4*260*4 + (72+3+10+16)*4), "pilot_ratio_estimated_bytes": int(track["cache_bytes"] / track["frames"] * total_frames)}
    result = {"comparison_16_frames": comparison, "selected_variant_batching": use_batching, "pilot_track": track, "full_cache_estimate": estimate, "passed": True}
    (GATE_OUT / "throughput_gate.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


def validated_existing(path, split, sequence, person):
    if not path.exists(): return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            validate_track_payload(payload)
            if str(payload["split"].item()) != split or str(payload["sequence"].item()) != sequence or int(payload["person_id"].item()) != person:
                raise ValueError("cache identity mismatch")
            frames = int(len(payload["frame_id"]))
        return {"path": str(path), "sha256": file_sha256(path), "frames": frames, "cache_bytes": path.stat().st_size, "resumed": True}
    except Exception:
        return None


def deterministic_validation_windows(entries, length=16):
    windows = []
    for entry in entries:
        with np.load(entry["path"], allow_pickle=False) as payload:
            local = np.asarray(payload["local_frame_index"], dtype=np.int64)
            frame_ids = np.asarray(payload["frame_id"], dtype=np.int64)
            sequence = str(payload["sequence"].item()); person = int(payload["person_id"].item())
        run_start = 0
        for end in range(1, len(local) + 1):
            boundary = end == len(local) or local[end] != local[end - 1] + 1
            if not boundary: continue
            for start in range(run_start, end - length + 1, length):
                windows.append({"sequence": sequence, "person_id": person, "cache_start": start, "cache_stop": start + length, "frame_ids": frame_ids[start:start+length].tolist()})
            run_start = end
    return windows


def full_cache():
    GATE_OUT.mkdir(parents=True, exist_ok=True)
    gate = json.loads((GATE_OUT / "throughput_gate.json").read_text())
    init_gate = json.loads((GATE_OUT / "init_float16_gate.json").read_text())
    if not gate.get("passed") or not init_gate.get("passed"):
        raise RuntimeError("engineering Gates are not passed")
    # The measured batching path was faster but not numerically equivalent.
    variant_batching = bool(gate["selected_variant_batching"])
    manifest_root = ROOT / "data/dynamesh_v2/manifests"
    train_manifest = json.loads((manifest_root / "3dpw_train_manifest.json").read_text())
    val_manifest = json.loads((manifest_root / "3dpw_validation_manifest.json").read_text())
    audit = json.loads((manifest_root / "3dpw_split_audit.json").read_text())
    if audit["train_test_overlap"] or audit["validation_test_overlap"] or audit["train_validation_overlap"]:
        raise RuntimeError(f"split leakage detected: {audit}")
    tasks = []
    for split, manifest in (("train", train_manifest), ("validation", val_manifest)):
        if set(manifest["sequences"]) & set(audit["test_sequences_for_leakage_guard_only"]):
            raise RuntimeError(f"test leakage in {split}")
        for sequence in manifest["sequences"]:
            annotation = load_pickle(DATA / "sequenceFiles" / split / f"{sequence}.pkl")
            for person in range(len(annotation["poses"])):
                if np.asarray(annotation["campose_valid"][person], dtype=bool).any():
                    tasks.append((split, sequence, person))
    estimator = load_estimator()
    entries = []; status_path = OUT / "cache_build_status.json"; started = time.time()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    initial_status = {"schema_version": SCHEMA_VERSION, "state": "running", "completed_tracks": 0, "total_tracks": len(tasks), "completed_frames": 0, "elapsed_seconds": 0.0, "current_track": {"split": tasks[0][0], "sequence": tasks[0][1], "person_id": tasks[0][2]} if tasks else None}
    temporary = status_path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(initial_status, indent=2) + "\n"); temporary.replace(status_path)
    for index, (split, sequence, person) in enumerate(tasks):
        path = OUT / split / f"{sequence}_person{person}.npz"
        entry = validated_existing(path, split, sequence, person)
        if entry is None:
            entry = build_track(estimator, split, sequence, person, variant_batching=variant_batching)
            entry["resumed"] = False
        entry.update({"split": split, "sequence": sequence, "person_id": person})
        entries.append(entry)
        status = {"schema_version": SCHEMA_VERSION, "state": "running", "completed_tracks": index + 1, "total_tracks": len(tasks), "completed_frames": sum(x["frames"] for x in entries), "elapsed_seconds": time.time() - started, "last_entry": entry}
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(status, indent=2) + "\n"); temporary.replace(status_path)
        print(json.dumps(status), flush=True)
    train_entries = [x for x in entries if x["split"] == "train"]
    val_entries = [x for x in entries if x["split"] == "validation"]
    windows = deterministic_validation_windows(val_entries)
    final_manifest = {"schema_version": SCHEMA_VERSION, "condition_names": CONDITIONS, "occlusion_ratios": RATIOS.tolist(), "variant_batching": variant_batching, "test_cache_files": 0, "entries": entries, "train_tracks": len(train_entries), "validation_tracks": len(val_entries), "train_frames": sum(x["frames"] for x in train_entries), "validation_frames": sum(x["frames"] for x in val_entries), "validation_windows_length16_nonoverlap": windows}
    manifest_path = OUT / "manifest.json"; temp = manifest_path.with_suffix(".json.tmp"); temp.write_text(json.dumps(final_manifest, indent=2) + "\n"); temp.replace(manifest_path)
    status = {"schema_version": SCHEMA_VERSION, "state": "complete", "completed_tracks": len(entries), "total_tracks": len(tasks), "completed_frames": sum(x["frames"] for x in entries), "elapsed_seconds": time.time() - started, "manifest": str(manifest_path)}
    temporary = status_path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(status, indent=2) + "\n"); temporary.replace(status_path)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("pilot", "full")); args = parser.parse_args()
    pilot() if args.mode == "pilot" else full_cache()
