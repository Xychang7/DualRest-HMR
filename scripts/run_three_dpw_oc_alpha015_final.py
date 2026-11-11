"""Formal resumable 3DPW-OC benchmark at frozen restoration strength alpha=.15."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import run_three_dpw_oc_full as base
import run_three_dpw_oc_paired_smoke as smoke
from evaluate_dynamesh_v2_unified_3dpw import decode_mhr, refine_all

ROOT = Path("/root/sam-body4d-0.2.0")
OUT = ROOT / "outputs/dynamesh_v2_3dpw_oc_alpha015_final"
OLD = ROOT / "outputs/dynamesh_v2_3dpw_oc_full"
ALPHA = 0.15
CHECKPOINT_SHA256 = "f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_track(sequence, actor, cache, estimator, adapter):
    work = OUT / "tracks" / f"{sequence}_person{actor}"
    work.mkdir(parents=True, exist_ok=True)
    sam_conv = work / "sam_smpl.npz"
    old_sam = OLD / "tracks" / f"{sequence}_person{actor}" / "sam_smpl.npz"
    if not old_sam.is_file():
        raise FileNotFoundError(f"missing frozen SAM conversion: {old_sam}")
    if not sam_conv.exists():
        sam_conv.symlink_to(old_sam)
    dyna_conv = work / "dyna_smpl.npz"
    frames = np.asarray(cache["local_frame_index"], dtype=np.int64)
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError(f"cache RGB frames are not unique/increasing: {sequence}/actor{actor}")
    contiguous_segments = 1 + int(np.sum(np.diff(frames) != 1))
    token = np.asarray(cache["pose_token"][0], dtype=np.float32)
    init = np.asarray(cache["mhr_init_pose"][0], dtype=np.float32)
    equivalence = smoke.raw_cache_equivalence(
        estimator.model.head_pose, token, init, cache["body_cont"][0]
    )
    sam_mhr_path = work / "sam_mhr.npy"
    dyna_mhr_path = work / "dyna_mhr.npy"
    if not dyna_conv.is_file() and not dyna_mhr_path.is_file():
        refined, coverage = refine_all(adapter, token, ALPHA, frames)
        _, dyna_mhr = decode_mhr(estimator.model.head_pose, token, refined, init)
        if dyna_mhr.shape[1:] != (18439, 3) or not np.isfinite(dyna_mhr).all():
            raise ValueError("Dyna MHR topology/finite check failed")
        smoke.atomic_npy(dyna_mhr_path, dyna_mhr)
        coverage_windows = len(coverage["windows"])
    else:
        coverage_windows = None
    return {
        "sequence": sequence,
        "actor": actor,
        "frames": frames,
        "sam_mhr": sam_mhr_path,
        "dyna_mhr": dyna_mhr_path,
        "sam_conv": sam_conv,
        "dyna_conv": dyna_conv,
        "equivalence": equivalence,
        "coverage_windows": coverage_windows,
        "contiguous_segments": contiguous_segments,
    }


def write_protocol_and_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "conversions", "tracks", "sequences"):
        (OUT / name).mkdir(exist_ok=True)
    checkpoint = ROOT / "outputs/dynamesh_v2_phase2_training/best.pt"
    manifest = ROOT / "evaluation/three_dpw_oc/3dpw_oc_manifest.json"
    actual_checkpoint_hash = sha256(checkpoint)
    if actual_checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("formal Body checkpoint hash mismatch")
    frozen_files = {
        "body_checkpoint": checkpoint,
        "oc_manifest": manifest,
        "decoder_and_refinement": ROOT / "scripts/evaluate_dynamesh_v2_unified_3dpw.py",
        "converter_adapter": ROOT / "adapters/mhr_to_smpl_persistent_worker.py",
        "evaluator": ROOT / "evaluation/three_dpw_oc/evaluate.py",
        "runner": Path(__file__),
    }
    protocol = {
        "schema": "dynamesh_v2_3dpw_oc_alpha015_final_v1",
        "temporal_restoration_strength_alpha": ALPHA,
        "window_length": 16,
        "stride": 8,
        "checkpoint": str(checkpoint),
        "hashes": {key: sha256(path) for key, path in frozen_files.items()},
        "model_input_contract": "clean continuous 30fps RGB cache; no OC annotation/severity/GT input",
        "spatial_protocol": "frozen JOTR-aligned OC JSON GT",
        "temporal_protocol": "sequenceFiles root-relative J14; OC centers; 30fps delta2",
    }
    smoke.atomic_json(OUT / "protocol.json", protocol)
    manifest_data = json.loads(manifest.read_text())
    sequences = sorted({row["sequence"] for row in manifest_data["instances"]})
    actors = {
        sequence: sorted({int(row["source_actor_id"]) for row in manifest_data["instances"] if row["sequence"] == sequence})
        for sequence in sequences
    }
    cache_index = base.load_cache_index()
    reuse = []
    cache_rows = []
    for sequence in sequences:
        for actor in actors[sequence]:
            cache = base.cache_path(cache_index, sequence, actor)
            old_sam = OLD / "tracks" / f"{sequence}_person{actor}" / "sam_smpl.npz"
            if not old_sam.is_file():
                raise FileNotFoundError(old_sam)
            cache_rows.append({"sequence": sequence, "actor": actor, "path": str(cache), "size": cache.stat().st_size})
            reuse.append({
                "sequence": sequence,
                "actor": actor,
                "sam": {"status": "reused", "source": str(old_sam), "reason": "frozen paired pipeline and evaluator"},
                "dyna": {"status": "recomputed", "source": None, "reason": "no prior alpha=.15 artifact with complete required provenance"},
            })
    smoke.atomic_json(OUT / "cache_audit.json", {
        "visual_cache_tracks_available": len(cache_rows),
        "sequence_count": len(sequences),
        "all_readable": all(Path(row["path"]).is_file() for row in cache_rows),
        "tracks": cache_rows,
    })
    smoke.atomic_json(OUT / "reuse_manifest.json", {"entries": reuse})
    smoke.atomic_json(OUT / "sequence_status.json", {
        "state": "prepared", "completed_sequences": 0, "total_sequences": len(sequences)
    })


def main():
    write_protocol_and_audit()
    base.OUT = OUT
    smoke.OUT = OUT
    base.prepare_track = prepare_track
    original_atomic = smoke.atomic_json

    def audited_atomic(path, value):
        original_atomic(path, value)
        path = Path(path)
        if path.parent.name == "sequences" and value.get("completion") is True:
            completed = len(list((OUT / "sequences").glob("*.json")))
            original_atomic(OUT / "sequence_status.json", {
                "state": "running", "completed_sequences": completed,
                "total_sequences": 22, "last_completed": value.get("sequence")
            })

    smoke.atomic_json = audited_atomic
    base.main()
    original_atomic(OUT / "sequence_status.json", {
        "state": "complete", "completed_sequences": 22, "total_sequences": 22
    })


if __name__ == "__main__":
    main()
