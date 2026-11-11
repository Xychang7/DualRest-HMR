"""Paired SAM/Dyna-v2 3DPW-OC smoke test on exactly two sequences."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/sam-body4d-0.2.0")
DATA = Path("/root/data1/sam-body4d-master/dataset/3DPW")
CACHE = ROOT / "data/dynamesh_v2/cache_v2/train"
OUT = ROOT / "outputs/dynamesh_v2_3dpw_oc_phase4_smoke"
SEQUENCES = ("courtyard_backpack_00", "courtyard_basketball_00")
CORRECTED = DATA / "eval_assets/J_regressor_h36m_correct.npy"
SMPL_MODELS = Path("/root/DuoMo-main/data/body_models/smpl")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "models/sam_3d_body")]

from evaluate_dynamesh_v2_unified_3dpw import (decode_mhr, load_models, refine_all,
                                                run_converter)
from evaluation.three_dpw_oc.evaluate import evaluate_records
from evaluation.three_dpw_oc.gt import (GenderedSMPLGT, build_oc_gt_meshes,
                                         build_sequence_temporal_gt,
                                         load_corrected_h36m_regressor)
from evaluation.three_dpw_oc.prediction_adapter import (make_paired_prediction_records,
                                                          paired_intersection_manifest)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_npy(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_track(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


@torch.no_grad()
def raw_cache_equivalence(head, token, init, expected_body):
    count = min(32, len(token))
    output = head(torch.as_tensor(token[:count], device="cuda", dtype=torch.float32),
                  torch.as_tensor(init[:count], device="cuda", dtype=torch.float32))
    if "pred_pose_raw" not in output:
        return {"available": False}
    actual = output["pred_pose_raw"][:, 6:].detach().cpu().numpy()
    expected = np.asarray(expected_body[:count], dtype=np.float32)
    return {"available": True, "frames": count,
            "max_abs": float(np.max(np.abs(actual - expected))),
            "mean_abs": float(np.mean(np.abs(actual - expected)))}


def prepare_track(sequence, actor, cache, estimator, adapter):
    work = OUT / "tracks" / f"{sequence}_person{actor}"
    work.mkdir(parents=True, exist_ok=True)
    sam_conv = work / "sam_smpl.npz"
    dyna_conv = work / "dyna_smpl.npz"
    frames = np.asarray(cache["local_frame_index"], dtype=np.int64)
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError(f"cache RGB frames are not unique/increasing: {sequence}/actor{actor}")
    contiguous_segments = 1 + int(np.sum(np.diff(frames) != 1))
    token = np.asarray(cache["pose_token"][0], dtype=np.float32)
    init = np.asarray(cache["mhr_init_pose"][0], dtype=np.float32)
    equivalence = raw_cache_equivalence(estimator.model.head_pose, token, init,
                                        cache["body_cont"][0])
    sam_mhr_path, dyna_mhr_path = work / "sam_mhr.npy", work / "dyna_mhr.npy"
    if ((not sam_conv.is_file() and not sam_mhr_path.is_file()) or
            (not dyna_conv.is_file() and not dyna_mhr_path.is_file())):
        refined, coverage = refine_all(adapter, token, 0.0, frames)
        sam_mhr, dyna_mhr = decode_mhr(estimator.model.head_pose, token, refined, init)
        if sam_mhr.shape != dyna_mhr.shape or sam_mhr.shape[1:] != (18439, 3):
            raise ValueError("paired MHR topology mismatch")
        if not sam_conv.is_file(): atomic_npy(sam_mhr_path, sam_mhr)
        if not dyna_conv.is_file(): atomic_npy(dyna_mhr_path, dyna_mhr)
        coverage_windows = len(coverage["windows"])
    else:
        coverage_windows = None
    return {"sequence": sequence, "actor": actor, "frames": frames,
            "sam_mhr": sam_mhr_path, "dyna_mhr": dyna_mhr_path,
            "sam_conv": sam_conv, "dyna_conv": dyna_conv,
            "equivalence": equivalence, "coverage_windows": coverage_windows,
            "contiguous_segments": contiguous_segments}


def finish_track(prepared):
    sequence, actor, frames = prepared["sequence"], prepared["actor"], prepared["frames"]
    with np.load(prepared["sam_conv"], allow_pickle=False) as sam_data, np.load(prepared["dyna_conv"], allow_pickle=False) as dyna_data:
        sam_vertices = np.asarray(sam_data["vertices"], dtype=np.float32)
        dyna_vertices = np.asarray(dyna_data["vertices"], dtype=np.float32)
        sam_fit = np.asarray(sam_data["errors"], dtype=np.float64)
        dyna_fit = np.asarray(dyna_data["errors"], dtype=np.float64)
    sam_records, dyna_records = make_paired_prediction_records(
        sequence, frames, actor, sam_vertices, dyna_vertices, pred_track_id=actor)
    prepared["sam_mhr"].unlink(missing_ok=True)
    prepared["dyna_mhr"].unlink(missing_ok=True)
    return sam_records, dyna_records, {
        "sequence": sequence, "source_actor_id": actor, "frames": len(frames),
        "first_rgb_frame": int(frames[0]), "last_rgb_frame": int(frames[-1]),
        "coverage_windows": prepared["coverage_windows"],
        "contiguous_rgb_segments": prepared["contiguous_segments"],
        "raw_cache_equivalence": prepared["equivalence"],
        "sam_converter_fit_mm": float(sam_fit.mean() * 1000.0),
        "dyna_converter_fit_mm": float(dyna_fit.mean() * 1000.0),
        "shared_input_contract": "same clean pose token, variant init, frame, actor and crop cache",
        "final_parameter_sources": {"body_articulation": "refined token only",
                                    "global_rotation": "raw SAM",
                                    "hand_shape_scale_face": "raw SAM"},
    }


def compact_metrics(result):
    return {"evaluated_oc_instances": result["spatial_instances"],
            "mpjpe_mm": result["mpjpe_mm"],
            "pa_mpjpe_mm": result["pa_mpjpe_mm"],
            "mpvpe_mm": result["mpvpe_mm"],
            "accel_m_s2": result["acceleration"]["mean_m_s2"],
            "valid_accel_triplets": result["acceleration"]["valid_triplets"],
            "accel_invalid_reasons": result["acceleration"]["invalid_reasons"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "evaluation/three_dpw_oc/3dpw_oc_manifest.json").read_text())
    corrected = load_corrected_h36m_regressor(CORRECTED)
    builder = GenderedSMPLGT(SMPL_MODELS)
    entries = []
    cache_manifest = json.loads((ROOT / "data/dynamesh_v2/cache_v2/manifest.json").read_text())
    for row in cache_manifest["entries"]:
        if row["split"] == "train" and row["sequence"] in SEQUENCES:
            entries.append(row)
    expected_tracks = {(SEQUENCES[0], 0), (SEQUENCES[1], 0), (SEQUENCES[1], 1)}
    actual_tracks = {(row["sequence"], int(row["person_id"])) for row in entries}
    if actual_tracks != expected_tracks:
        raise ValueError(f"paired cache track mismatch: {actual_tracks}")
    estimator, adapter = load_models()
    all_sam, all_dyna, track_reports, prepared_tracks = {}, {}, [], []
    for index, row in enumerate(sorted(entries, key=lambda item: (item["sequence"], item["person_id"]))):
        atomic_json(OUT / "status.json", {"state": "running", "completed_tracks": index,
                    "total_tracks": len(entries), "current_sequence": row["sequence"],
                    "current_actor": row["person_id"], "started_at": time.time()})
        cache = load_track(row["path"])
        prepared_tracks.append(prepare_track(row["sequence"], int(row["person_id"]),
                                             cache, estimator, adapter))
    del estimator, adapter
    torch.cuda.empty_cache()
    conversion_jobs = []
    for prepared in prepared_tracks:
        for method in ("sam", "dyna"):
            if not prepared[f"{method}_conv"].is_file():
                conversion_jobs.append((prepared[f"{method}_mhr"], prepared[f"{method}_conv"]))
    atomic_json(OUT / "status.json", {"state": "running", "stage": "parallel_conversion",
                "completed_conversions": 0, "total_conversions": len(conversion_jobs),
                "max_parallel_workers": 6})
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(run_converter, source, target): (source, target)
                   for source, target in conversion_jobs}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            atomic_json(OUT / "status.json", {"state": "running", "stage": "parallel_conversion",
                        "completed_conversions": completed, "total_conversions": len(conversion_jobs),
                        "max_parallel_workers": 6})
    for prepared in prepared_tracks:
        sam, dyna, track_report = finish_track(prepared)
        sequence = prepared["sequence"]
        all_sam.setdefault(sequence, []).extend(sam)
        all_dyna.setdefault(sequence, []).extend(dyna)
        track_reports.append(track_report)

    sequence_reports = {}
    for sequence in SEQUENCES:
        sequence_manifest = {**{key: value for key, value in manifest.items() if key != "instances"},
                             "instances": [row for row in manifest["instances"]
                                           if row["sequence"] == sequence]}
        paired_manifest, matching = paired_intersection_manifest(
            sequence_manifest, all_sam[sequence], all_dyna[sequence])
        expected_actors = {int(row["source_actor_id"]) for row in sequence_manifest["instances"]}
        predicted_actors = {int(row["matched_actor_id"]) for row in all_sam[sequence]}
        matching["unmatched_person_count"] = len(expected_actors - predicted_actors)
        matching["frame_coverage"] = {str(actor): {
            "predicted_frames": sum(int(row["matched_actor_id"]) == actor for row in all_sam[sequence]),
            "first": min(int(row["rgb_frame_idx"]) for row in all_sam[sequence]
                         if int(row["matched_actor_id"]) == actor),
            "last": max(int(row["rgb_frame_idx"]) for row in all_sam[sequence]
                        if int(row["matched_actor_id"]) == actor)} for actor in predicted_actors}
        gt_spatial = build_oc_gt_meshes(DATA, paired_manifest["instances"], builder)
        gt_temporal, temporal_status, temporal_audit = build_sequence_temporal_gt(
            DATA, paired_manifest["instances"], builder)
        sam_result = evaluate_records(paired_manifest, all_sam[sequence], gt_spatial,
                                      corrected, builder.native_pelvis_regressor,
                                      gt_temporal, temporal_status)
        dyna_result = evaluate_records(paired_manifest, all_dyna[sequence], gt_spatial,
                                       corrected, builder.native_pelvis_regressor,
                                       gt_temporal, temporal_status)
        sequence_reports[sequence] = {"matching": matching,
                                      "temporal_gt": temporal_audit,
                                      "sam": compact_metrics(sam_result),
                                      "dyna": compact_metrics(dyna_result),
                                      "paired_keys_identical": True,
                                      "conversion": "same Meta MHR-to-SMPL worker/model"}
    report = {"state": "complete", "sequences": sequence_reports,
              "tracks": track_reports,
              "spatial_protocol": "frozen JOTR-aligned OC JSON GT",
              "temporal_protocol": "sequenceFiles root-relative J14; OC centers; 30fps delta2",
              "camera_note": "root-relative OC metrics intentionally do not evaluate Camera restoration"}
    atomic_json(OUT / "summary.json", report)
    atomic_json(OUT / "status.json", {"state": "complete", "completed_tracks": len(entries),
                "total_tracks": len(entries), "current_sequence": None, "current_actor": None})
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
