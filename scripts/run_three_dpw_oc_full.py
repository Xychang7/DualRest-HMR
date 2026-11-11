"""Full paired SAM/Dyna-v2 evaluation on all annotated 3DPW-OC sequences."""

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/sam-body4d-0.2.0")
DATA = Path("/root/data1/sam-body4d-master/dataset/3DPW")
OUT = ROOT / "outputs/dynamesh_v2_3dpw_oc_full"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "models/sam_3d_body")]

import run_three_dpw_oc_paired_smoke as smoke
from evaluation.three_dpw_oc.evaluate import evaluate_records
from evaluation.three_dpw_oc.gt import (
    GenderedSMPLGT,
    build_oc_gt_meshes,
    build_sequence_temporal_gt,
    load_corrected_h36m_regressor,
)
from evaluation.three_dpw_oc.prediction_adapter import paired_intersection_manifest
from evaluate_dynamesh_v2_unified_3dpw import load_models, run_converter

smoke.OUT = OUT

METRICS = ("mpjpe_mm", "pa_mpjpe_mm", "mpvpe_mm", "accel_m_s2")


def metric_comparison(sam, dyna):
    dyna_minus = {key: float(dyna[key] - sam[key]) for key in METRICS}
    return {
        "dyna_minus_sam": dyna_minus,
        "sam_minus_dyna_improvement": {key: -value for key, value in dyna_minus.items()},
    }


def _method_aggregate(reports, method, per_sequence):
    values = {}
    for metric in METRICS:
        denominator = "valid_accel_triplets" if metric == "accel_m_s2" else "evaluated_oc_instances"
        if per_sequence:
            values[metric] = float(np.mean([row[method][metric] for row in reports.values()]))
        else:
            total = sum(row[method][denominator] for row in reports.values())
            values[metric] = float(sum(row[method][metric] * row[method][denominator]
                                       for row in reports.values()) / total)
    values["evaluated_oc_instances"] = int(sum(row[method]["evaluated_oc_instances"]
                                                 for row in reports.values()))
    values["valid_accel_triplets"] = int(sum(row[method]["valid_accel_triplets"]
                                               for row in reports.values()))
    return values


def aggregate_reports(reports):
    instance = {method: _method_aggregate(reports, method, False) for method in ("sam", "dyna")}
    sequence = {method: _method_aggregate(reports, method, True) for method in ("sam", "dyna")}
    return {
        "instance_weighted": {**instance, **metric_comparison(instance["sam"], instance["dyna"])},
        "per_sequence_mean": {**sequence, **metric_comparison(sequence["sam"], sequence["dyna"])},
    }


def load_cache_index():
    manifest = json.loads((ROOT / "data/dynamesh_v2/cache_v2/manifest.json").read_text())
    index = {(row["sequence"], int(row["person_id"])): Path(row["path"])
             for row in manifest["entries"]}
    return index


def cache_path(index, sequence, actor):
    path = index.get((sequence, actor))
    if path is None:
        path = ROOT / "outputs/dynamesh_v2_final_test/cache" / f"{sequence}_person{actor}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing paired clean cache: {sequence}/person{actor}: {path}")
    return path


def compact_metrics(result):
    return smoke.compact_metrics(result)


def evaluate_sequence(sequence, actors, manifest, cache_index, estimator, adapter,
                      corrected, builder, prepare_track_fn=smoke.prepare_track,
                      finish_track_fn=smoke.finish_track):
    prepared = []
    for actor in actors:
        cache = smoke.load_track(cache_path(cache_index, sequence, actor))
        prepared.append(prepare_track_fn(sequence, actor, cache, estimator, adapter))

    jobs = []
    for track in prepared:
        for method in ("sam", "dyna"):
            if not track[f"{method}_conv"].is_file():
                jobs.append((track[f"{method}_mhr"], track[f"{method}_conv"]))
    if jobs:
        with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
            futures = [pool.submit(run_converter, source, target) for source, target in jobs]
            for future in as_completed(futures):
                future.result()

    sam_records, dyna_records, track_reports = [], [], []
    for track in prepared:
        sam, dyna, report = finish_track_fn(track)
        sam_records.extend(sam)
        dyna_records.extend(dyna)
        track_reports.append(report)

    sequence_manifest = {
        **{key: value for key, value in manifest.items() if key != "instances"},
        "instances": [row for row in manifest["instances"] if row["sequence"] == sequence],
    }
    paired_manifest, matching = paired_intersection_manifest(
        sequence_manifest, sam_records, dyna_records)
    expected_actors = {int(row["source_actor_id"]) for row in sequence_manifest["instances"]}
    predicted_actors = {int(row["matched_actor_id"]) for row in sam_records}
    matching["unmatched_person_count"] = len(expected_actors - predicted_actors)
    matching["frame_coverage"] = {
        str(actor): {
            "predicted_frames": sum(int(row["matched_actor_id"]) == actor for row in sam_records),
            "first": min(int(row["rgb_frame_idx"]) for row in sam_records
                         if int(row["matched_actor_id"]) == actor),
            "last": max(int(row["rgb_frame_idx"]) for row in sam_records
                        if int(row["matched_actor_id"]) == actor),
        } for actor in predicted_actors
    }
    gt_spatial = build_oc_gt_meshes(DATA, paired_manifest["instances"], builder)
    gt_temporal, temporal_status, temporal_audit = build_sequence_temporal_gt(
        DATA, paired_manifest["instances"], builder)
    sam_result = evaluate_records(paired_manifest, sam_records, gt_spatial, corrected,
                                  builder.native_pelvis_regressor, gt_temporal, temporal_status)
    dyna_result = evaluate_records(paired_manifest, dyna_records, gt_spatial, corrected,
                                   builder.native_pelvis_regressor, gt_temporal, temporal_status)
    sam_metrics, dyna_metrics = compact_metrics(sam_result), compact_metrics(dyna_result)
    return {
        "completion": True,
        "sequence": sequence,
        "matching": matching,
        "temporal_gt": temporal_audit,
        "sam": sam_metrics,
        "dyna": dyna_metrics,
        "paired": metric_comparison(sam_metrics, dyna_metrics),
        "paired_keys_identical": True,
        "conversion": "same Meta MHR-to-SMPL worker/model",
        "tracks": track_reports,
    }


def save_csv(reports):
    path = OUT / "per_sequence_results.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["sequence", "expected_oc_instances", "intersection_instances",
              "missing_prediction_instances", "unmatched_person_count", "duplicate_keys"]
    for method in ("sam", "dyna"):
        fields.extend(f"{method}_{metric}" for metric in METRICS)
        fields.append(f"{method}_valid_accel_triplets")
    fields.extend(f"improvement_{metric}" for metric in METRICS)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence, report in sorted(reports.items()):
            matching = report["matching"]
            row = {
                "sequence": sequence,
                "expected_oc_instances": matching["expected_oc_instances"],
                "intersection_instances": matching["matched_intersection_instances"],
                "missing_prediction_instances": matching["missing_prediction_instances"],
                "unmatched_person_count": matching["unmatched_person_count"],
                "duplicate_keys": matching["duplicate_keys"],
            }
            for method in ("sam", "dyna"):
                row.update({f"{method}_{metric}": report[method][metric] for metric in METRICS})
                row[f"{method}_valid_accel_triplets"] = report[method]["valid_accel_triplets"]
            row.update({f"improvement_{metric}": value for metric, value in
                        report["paired"]["sam_minus_dyna_improvement"].items()})
            writer.writerow(row)
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "evaluation/three_dpw_oc/3dpw_oc_manifest.json").read_text())
    sequences = sorted({row["sequence"] for row in manifest["instances"]})
    if len(sequences) != 22 or len(manifest["instances"]) != 20243:
        raise ValueError("frozen OC manifest count mismatch")
    actor_map = {sequence: sorted({int(row["source_actor_id"])
                                   for row in manifest["instances"]
                                   if row["sequence"] == sequence}) for sequence in sequences}
    cache_index = load_cache_index()
    for sequence in sequences:
        for actor in actor_map[sequence]:
            cache_path(cache_index, sequence, actor)

    corrected = load_corrected_h36m_regressor(smoke.CORRECTED)
    builder = GenderedSMPLGT(smoke.SMPL_MODELS)
    estimator, adapter = load_models()
    completed = {}
    for sequence in sequences:
        path = OUT / "sequences" / f"{sequence}.json"
        if path.is_file():
            report = json.loads(path.read_text())
            if report.get("completion") is not True:
                raise ValueError(f"incomplete prior sequence result: {sequence}")
            completed[sequence] = report

    for sequence in sequences:
        if sequence in completed:
            continue
        smoke.atomic_json(OUT / "status.json", {
            "state": "running", "stage": "paired_sequence_evaluation",
            "completed_sequences": len(completed), "total_sequences": len(sequences),
            "current_sequence": sequence, "started_or_resumed_at": time.time(),
            "failed_sequences": [],
        })
        report = evaluate_sequence(sequence, actor_map[sequence], manifest, cache_index,
                                   estimator, adapter, corrected, builder)
        smoke.atomic_json(OUT / "sequences" / f"{sequence}.json", report)
        completed[sequence] = report
        smoke.atomic_json(OUT / "per_sequence_results.json", completed)
        save_csv(completed)

    del estimator, adapter
    torch.cuda.empty_cache()
    aggregation = aggregate_reports(completed)
    summary = {
        "state": "complete",
        "sequence_count": len(completed),
        "expected_oc_instances": sum(row["matching"]["expected_oc_instances"]
                                     for row in completed.values()),
        "intersection_instances": sum(row["matching"]["matched_intersection_instances"]
                                      for row in completed.values()),
        "sam_evaluated_instances": sum(row["sam"]["evaluated_oc_instances"]
                                       for row in completed.values()),
        "dyna_evaluated_instances": sum(row["dyna"]["evaluated_oc_instances"]
                                        for row in completed.values()),
        "total_missing_predictions": sum(row["matching"]["missing_prediction_instances"]
                                         for row in completed.values()),
        "total_unmatched_people": sum(row["matching"]["unmatched_person_count"]
                                      for row in completed.values()),
        "total_duplicate_keys": sum(row["matching"]["duplicate_keys"]
                                    for row in completed.values()),
        "aggregation": aggregation,
        "spatial_protocol": "frozen JOTR-aligned OC JSON GT",
        "temporal_protocol": "sequenceFiles root-relative J14; OC centers; 30fps delta2",
    }
    smoke.atomic_json(OUT / "summary.json", summary)
    smoke.atomic_json(OUT / "status.json", {
        "state": "complete", "stage": "complete", "completed_sequences": len(completed),
        "total_sequences": len(sequences), "current_sequence": None, "failed_sequences": [],
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        smoke.atomic_json(OUT / "status.json", {
            "state": "failed", "stage": "failed", "error": repr(error),
            "failed_at": time.time(),
        })
        raise
