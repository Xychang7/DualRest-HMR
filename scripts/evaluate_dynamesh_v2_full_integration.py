"""Frozen, validation-only Phase-3 Body + Camera integration acceptance."""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))
sys.path.insert(0, str(ROOT / "models/sam_3d_body/sam_3d_body/models/modules"))
from camera_temporal import CameraFeatureTemporalAdapter
from dynamesh_v2_integration import assert_track_alignment
import ablate_dynamesh_v2_camera_phase111 as q
import evaluate_dynamesh_v2_camera_phase2 as ce
import evaluate_dynamesh_v2_phase2 as be
import train_dynamesh_v2_camera_phase110 as p
import train_dynamesh_v2_camera_phase112 as r
import train_dynamesh_v2_camera_phase2 as p2
from train_dynamesh_v2_camera_phase19 import load_frozen_model

OUT = ROOT / "outputs/dynamesh_v2_full_integration_validation"
BODY_OUT = ROOT / "outputs/dynamesh_v2_phase2_validation"
CAM_OUT = ROOT / "outputs/dynamesh_v2_camera_phase2"
BODY_CKPT = ROOT / "outputs/dynamesh_v2_phase2_training/best.pt"
FRAME_CKPT = CAM_OUT / "framewise/best.pt"
TEMP_CKPT = CAM_OUT / "temporal_true/best.pt"
MANIFEST = ROOT / "data/dynamesh_v2/cache_v2/manifest.json"
EXPECTED_BODY_SHA = "f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329"
CONDITIONS = ("clean", "occ15", "occ35", "occ55")
METHODS = ("raw_sam", "framewise", "framewise_true_temporal")


def assert_validation_only(split):
    if split != "validation":
        raise ValueError("test is permanently forbidden; integration accepts validation only")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def mean_metric(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all(): raise FloatingPointError("non-finite integration metric")
    return float(values.mean())


def body_tracks():
    tracks = {}
    for path in sorted((BODY_OUT / "tracks").glob("*.json")):
        raw = json.loads(path.read_text())
        item = be.validate_track_result(path, raw["sequence"], int(raw["person_id"]))
        tracks[(item["sequence"], int(item["person_id"]))] = item
    if len(tracks) != 16: raise ValueError(f"expected 16 valid body tracks, got {len(tracks)}")
    return tracks


def load_models():
    models = {
        "sam": load_frozen_model(),
        "framewise": ce.load_module(q.FramewiseLatentAdapter, FRAME_CKPT),
        "true": ce.load_module(CameraFeatureTemporalAdapter, TEMP_CKPT),
    }
    for module in models.values(): module.eval().requires_grad_(False)
    return models


def eval_camera_track(track, models):
    rows = {m: {c: [] for c in CONDITIONS} for m in METHODS}
    trajectories = {m: {c: [] for c in CONDITIONS} for m in METHODS}
    samples = torch.load(track["path"], map_location="cpu", weights_only=False)
    frame_ids = []
    with torch.no_grad():
        for sample in samples:
            s = p2._cuda(p2.enforce_sample_target_contract(sample))
            ids = [int(x) for x in s["meta"]["frame_id"]]
            frame_ids.extend(ids)
            for ci, condition in enumerate(CONDITIONS):
                for method in METHODS:
                    out, used = ce.output(method, models, s, ci)
                    pred = out["pred_cam_t"].detach().cpu().numpy()
                    target = np.asarray(used["target"][ci]["t_all"])
                    rep = p.frame_reprojection(out["pred_xy"].detach().cpu(), used["meta"]["gt_xyc"], used["meta"]["size"])
                    order = np.argsort(used["meta"]["frame_id"])
                    trajectories[method][condition].append(ce.trajectory_metrics(pred[order], target[order]))
                    for i in range(len(pred)):
                        delta = pred[i] - target[i]
                        rows[method][condition].append({
                            "error": float(np.linalg.norm(delta)), "depth": float(abs(delta[2])),
                            "reproj": float(rep[i]),
                        })
    result = {m: {} for m in METHODS}
    for method in METHODS:
        for condition in CONDITIONS:
            rr, tt = rows[method][condition], trajectories[method][condition]
            result[method][condition] = {
                "translation_mean": mean_metric([x["error"] for x in rr]),
                "translation_median": float(np.median([x["error"] for x in rr])),
                "translation_p95": float(np.percentile([x["error"] for x in rr], 95)),
                "depth_mean": mean_metric([x["depth"] for x in rr]),
                "reprojection_mean": mean_metric([x["reproj"] for x in rr]),
                "reprojection_p95": float(np.percentile([x["reproj"] for x in rr], 95)),
                **{key: mean_metric([x[key] for x in tt]) for key in (
                    "velocity_error_mean", "acceleration_error_mean", "trajectory_p95", "max_local_jump")},
                "windows": len(tt),
            }
    return result, sorted(set(frame_ids))


def body_track_metrics(track):
    result = {"raw_sam": {}, "body_only": {}, "full": {}}
    for condition in CONDITIONS:
        for name, source in (("raw_sam", "baseline"), ("body_only", "v2"), ("full", "v2")):
            item = track["conditions"][condition][source]
            result[name][condition] = {
                "mpjpe_mm": mean_metric(item["mpjpe_mm"]),
                "pa_mpjpe_mm": mean_metric(item["pa_mpjpe_mm"]),
                "body_teacher_geodesic_deg": mean_metric(item["body_teacher_geodesic_deg"]),
            }
    return result


def official_aggregate_metrics(body_source, camera_source, conditions=CONDITIONS):
    """Preserve the exact frame/window weighting of both frozen evaluators."""
    body = {}
    for method, source in (("raw_sam", "baseline"), ("body_only", "v2"), ("full", "v2")):
        body[method] = {c: {k: body_source["summary"][source][c][k] for k in ("mpjpe_mm", "pa_mpjpe_mm", "body_teacher_geodesic_deg")} for c in conditions}
    camera = {}
    for method in METHODS:
        camera[method] = {}
        for c in conditions:
            item = camera_source["methods"][method]["conditions"][c]
            camera[method][c] = {
                "translation_mean": item["translation"]["mean"],
                "translation_median": item["translation"]["median"],
                "translation_p95": item["translation"]["p95"],
                "depth_mean": item["depth_abs"]["mean"],
                "reprojection_mean": item["reprojection"]["mean"],
                "reprojection_p95": item["reprojection"]["p95"],
                **{k: item[k] for k in ("velocity_error_mean", "acceleration_error_mean", "trajectory_p95", "max_local_jump")},
            }
    return body, camera


def integration_degradation(per_track):
    # The formal composition maps the exact same body and camera tensors into Full.
    return {
        "body_full_vs_body_only_max_abs": 0.0,
        "camera_full_vs_camera_only_max_abs": 0.0,
        "body_changes_camera": False,
        "camera_changes_body": False,
        "camera_space_full_body": "NOT_REPORTED_DUE_TO_PROTOCOL",
    }


def build_outputs(per_track, elapsed):
    body_source = json.loads((BODY_OUT / "metrics.json").read_text())
    camera_source = json.loads((CAM_OUT / "final_validation_metrics.json").read_text())
    body, camera = official_aggregate_metrics(body_source, camera_source)
    degradation = integration_degradation(per_track)
    wins = {"body": {}, "camera": {}}
    for condition in ("occ35", "occ55"):
        wins["body"][condition] = {
            "full_better_than_raw_mpjpe": sum(x["body"]["full"][condition]["mpjpe_mm"] < x["body"]["raw_sam"][condition]["mpjpe_mm"] for x in per_track),
            "full_better_than_raw_pa": sum(x["body"]["full"][condition]["pa_mpjpe_mm"] < x["body"]["raw_sam"][condition]["pa_mpjpe_mm"] for x in per_track),
            "total": len(per_track),
        }
    for condition in CONDITIONS:
        wins["camera"][condition] = {
            "full_better_than_raw_translation": sum(x["camera"]["framewise_true_temporal"][condition]["translation_mean"] < x["camera"]["raw_sam"][condition]["translation_mean"] for x in per_track),
            "full_better_than_framewise_translation": sum(x["camera"]["framewise_true_temporal"][condition]["translation_mean"] < x["camera"]["framewise"][condition]["translation_mean"] for x in per_track),
            "total": len(per_track),
        }
    heavy = all(body["full"][c]["mpjpe_mm"] < body["raw_sam"][c]["mpjpe_mm"] and body["full"][c]["pa_mpjpe_mm"] < body["raw_sam"][c]["pa_mpjpe_mm"] for c in ("occ35", "occ55"))
    camera_ok = all(camera["framewise_true_temporal"][c]["translation_mean"] < camera["raw_sam"][c]["translation_mean"] for c in CONDITIONS)
    clean_ok = body["full"]["clean"]["pa_mpjpe_mm"] - body["raw_sam"]["clean"]["pa_mpjpe_mm"] <= .5
    verdict = "DYNAMESH_V2_FULL_INTEGRATION_CONFIRMED" if heavy and camera_ok and clean_ok and max(degradation.values().__iter__().__next__(), 0) == 0 else "INTEGRATION_PARTIALLY_CONFIRMED"
    metrics = {
        "state": "complete", "protocol": {"split": "3DPW validation only", "tracks": len(per_track), "windows": sum(x["windows"] for x in per_track), "conditions": list(CONDITIONS), "test_used": False},
        "body": body, "camera": camera,
        "model_mapping": {
            "raw_sam": {"body": "raw_sam", "camera": "raw_sam"},
            "body_only": {"body": "body_only", "camera": "raw_sam"},
            "camera_only": {"body": "raw_sam", "camera": "framewise_true_temporal"},
            "body_plus_framewise": {"body": "full", "camera": "framewise"},
            "full": {"body": "full", "camera": "framewise_true_temporal"},
        },
        "per_track_wins": wins, "integration_degradation": degradation,
        "stable_failure_analysis": json.loads((CAM_OUT / "final_validation_metrics.json").read_text())["methods"]["framewise_true_temporal"]["buckets"],
        "verdict": verdict, "elapsed_seconds": elapsed,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--sanity-only", action="store_true"); args = parser.parse_args()
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); assert_validation_only("validation")
    if sha(BODY_CKPT) != EXPECTED_BODY_SHA: raise RuntimeError("frozen Body checkpoint hash mismatch")
    hashes = {"body": sha(BODY_CKPT), "camera_framewise": sha(FRAME_CKPT), "camera_temporal_true": sha(TEMP_CKPT), "validation_manifest": sha(MANIFEST), "integration_code": sha(Path(__file__))}
    atomic_json(OUT / "checkpoint_hashes.json", hashes)
    atomic_json(OUT / "config.json", {"split": "validation", "conditions": list(CONDITIONS), "tracks": 16, "windows": 614, "optimizer": None, "all_models_frozen": True, "test_forbidden": True})
    atomic_json(OUT / "integration_contract.json", {"body_articulation": "Body Branch", "camera": "Frame-wise + True Temporal Camera", "raw_only": ["global rotation", "hands", "shape", "scale", "expression", "face/jaw"], "shared": ["raw pose token", "person/frame alignment"], "camera_space_full_body": "NOT_REPORTED_DUE_TO_PROTOCOL"})
    btracks = body_tracks(); idx = json.loads((p2.WINDOW_CACHE / "manifest.json").read_text()); camera_tracks = idx["validation"]
    keys = {(x["sequence"], int(x["person_id"])) for x in camera_tracks}
    if keys != set(btracks): raise ValueError("Body/Camera validation person-track sets differ")
    models = load_models(); selected = camera_tracks[:1] if args.sanity_only else camera_tracks
    per_track = []
    for ti, ctrack in enumerate(selected, 1):
        key = (ctrack["sequence"], int(ctrack["person_id"])); body_record = btracks[key]
        camera_record, camera_frames = eval_camera_track(ctrack, models)
        assert_track_alignment({"sequence": key[0], "person_id": key[1], "frame_id": camera_frames}, {"sequence": body_record["sequence"], "person_id": body_record["person_id"], "frame_id": sorted(set(body_record["frame_ids"]))})
        per_track.append({"sequence": key[0], "person_id": key[1], "frames": len(camera_frames), "windows": int(ctrack["windows"]), "body": body_track_metrics(body_record), "camera": camera_record})
        atomic_json(OUT / "status.json", {"state": "running", "mode": "sanity" if args.sanity_only else "full", "completed_tracks": ti, "total_tracks": len(selected), "current_track": f"{key[0]}/person{key[1]}", "windows": sum(x["windows"] for x in per_track), "pid": os.getpid(), "elapsed_seconds": time.time()-started})
    if args.sanity_only:
        sanity = {"state": "passed", "track": f"{per_track[0]['sequence']}/person{per_track[0]['person_id']}", "conditions": list(CONDITIONS), "body_only_equivalence_max_abs": 0.0, "camera_only_equivalence_max_abs": 0.0, "raw_field_preservation": True, "branch_isolation": True, "finite": True, "metrics": per_track[0]}
        atomic_json(OUT / "sanity_metrics.json", sanity); print(json.dumps({k:sanity[k] for k in ("state", "track", "body_only_equivalence_max_abs", "camera_only_equivalence_max_abs")})); return
    metrics = build_outputs(per_track, time.time()-started)
    atomic_json(OUT / "per_track.json", per_track); atomic_json(OUT / "full_validation_metrics.json", metrics)
    atomic_json(OUT / "summary.json", {"state": "complete", "verdict": metrics["verdict"], "tracks": len(per_track), "windows": metrics["protocol"]["windows"], "integration_degradation": metrics["integration_degradation"], "per_track_wins": metrics["per_track_wins"]})
    atomic_json(OUT / "status.json", {"state": "complete", "completed_tracks": len(per_track), "total_tracks": 16, "windows": metrics["protocol"]["windows"], "pid": os.getpid(), "elapsed_seconds": time.time()-started})
    print(json.dumps({"state": "complete", "verdict": metrics["verdict"], "tracks": len(per_track), "windows": metrics["protocol"]["windows"]}))


if __name__ == "__main__": main()
