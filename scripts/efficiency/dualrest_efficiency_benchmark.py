from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/root/sam-body4d-0.2.0")
OUT = ROOT / "outputs/dualrest_efficiency_benchmark"
DATA = Path("/root/data1/sam-body4d-master/dataset/3DPW")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "models/sam_3d_body")]

from ablate_dynamesh_v2_camera_phase111 import FramewiseLatentAdapter
from build_dynamesh_v2_cache_sanity import bbox_from_pose2d, load_pickle
from sam_3d_body import SAM3DBodyEstimator
from sam_3d_body.data.transforms import Compose
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
from sam_3d_body.utils import recursive_to
from train_dynamesh_v2_camera_phase1 import MHR_IDX, load_frozen_model, reliability_features
from train_dynamesh_v2_camera_phase112 import temporal_residual_forward
from ablate_dynamesh_v2_camera_phase16 import projection


BODY_CKPT = ROOT / "outputs/dynamesh_v2_phase2_training/best.pt"
FW_CKPT = ROOT / "outputs/dynamesh_v2_camera_phase2/framewise/best.pt"
TEMP_CKPT = ROOT / "outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt"
EXPECTED_HASHES = {
    "body": "f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329",
    "framewise_camera": "fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2",
    "temporal_camera": "e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865",
}
EXPECTED_COUNTS = {
    "sam_body4d": 1_285_412_468,
    "body_restoration": 2_345_472,
    "framewise_camera": 265_344,
    "temporal_camera": 733_633,
    "added_total": 3_344_449,
}
T = 16
ROUNDS = 3


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def log(message: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (OUT / "run.log").open("a") as f:
        f.write(line + "\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nparams(module):
    return sum(p.numel() for p in module.parameters())


def state_count(state):
    return sum(int(v.numel()) for v in state.values() if torch.is_tensor(v))


def gpu_snapshot():
    query = "index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
    row = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()[0]
    keys = query.split(",")
    values = [x.strip() for x in row.split(",")]
    return dict(zip(keys, values))


def compute_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    return [x.strip() for x in result.stdout.splitlines() if x.strip()]


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_times(values):
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    return {
        "iterations": int(len(a)), "mean_ms_per_window": mean,
        "median_ms_per_window": float(np.median(a)), "std_ms_per_window": float(a.std(ddof=0)),
        "p95_ms_per_window": percentile(a, 95), "mean_ms_per_frame": mean / T,
        "fps_equivalent": 1000.0 * T / mean,
    }


def benchmark_cuda(fn, warmup, measured):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        values = []
        for _ in range(measured):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    result = summarize_times(values)
    result.update({
        "baseline_allocated_bytes": baseline_alloc,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": peak_alloc - baseline_alloc,
        "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
    })
    return result


def aggregate_rounds(rounds):
    means = [x["mean_ms_per_window"] for x in rounds]
    result = {
        "rounds": rounds,
        "mean_ms_per_window": float(np.mean(means)),
        "median_ms_per_window": float(np.mean([x["median_ms_per_window"] for x in rounds])),
        "std_ms_per_window": float(np.mean([x["std_ms_per_window"] for x in rounds])),
        "p95_ms_per_window": float(np.mean([x["p95_ms_per_window"] for x in rounds])),
        "ms_per_frame": float(np.mean(means)) / T,
        "fps_equivalent": 1000.0 * T / float(np.mean(means)),
        "round_mean_variation_pct": 100.0 * (max(means) - min(means)) / float(np.mean(means)),
        "peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in rounds),
        "peak_reserved_bytes": max(x["peak_reserved_bytes"] for x in rounds),
        "incremental_peak_allocated_bytes": max(x["incremental_peak_allocated_bytes"] for x in rounds),
        "incremental_peak_reserved_bytes": max(x["incremental_peak_reserved_bytes"] for x in rounds),
    }
    return result


def prepare_validation_sample(estimator):
    sequence, person = "courtyard_basketball_01", 0
    ann = load_pickle(DATA / "sequenceFiles/validation" / f"{sequence}.pkl")
    valid = np.asarray(ann["campose_valid"][person], dtype=bool)
    selected_all = np.flatnonzero(valid)
    selected = selected_all[:T]
    frame_ids = np.asarray(ann["img_frame_ids"], dtype=np.int64)[selected]
    paths = [DATA / "imageFiles" / sequence / f"image_{i:05d}.jpg" for i in selected]
    images_bgr = [cv2.imread(str(p)) for p in paths]
    if any(x is None for x in images_bgr):
        raise FileNotFoundError("validation image missing")
    h, w = images_bgr[0].shape[:2]
    poses2d = np.asarray(ann["poses2d"][person])
    boxes = np.stack([bbox_from_pose2d(poses2d[i], w, h) for i in selected]).astype(np.float32)
    images = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in images_bgr]
    cam = torch.as_tensor(np.asarray(ann["cam_intrinsics"], dtype=np.float32), device="cuda").unsqueeze(0)

    batches = []
    for image, box in zip(images, boxes):
        b = prepare_batch(image, estimator.transform, box.reshape(1, 4), None, None)
        b["cam_int"] = cam.clone().to(b["img"])
        batches.append(b)
    merged = {}
    for b in batches:
        for key, value in b.items():
            if key == "bbox_format":
                continue
            if key == "img_ori":
                value = value[0]
            merged.setdefault(key, []).append(value)
    for key, values in merged.items():
        if key == "img_ori":
            continue
        merged[key] = torch.concat(values, dim=0)
    merged["bbox_format"] = batches[0]["bbox_format"]
    merged = recursive_to(merged, "cuda")

    center = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    size = (boxes[:, 2] - boxes[:, 0]) * 1.25
    meta = {
        "center": center.astype(np.float32), "size": size.astype(np.float32),
        "image_size": np.tile(np.asarray([[w, h]], np.float32), (T, 1)),
        "K": np.asarray(ann["cam_intrinsics"], dtype=np.float32),
    }
    sample = {
        "split": "validation", "sequence": sequence, "person_id": person,
        "local_indices": selected.astype(int).tolist(), "frame_ids": frame_ids.astype(int).tolist(),
        "image_paths": [str(p) for p in paths], "image_size_wh": [int(w), int(h)],
        "bbox_source": "3DPW poses2d via formal bbox_from_pose2d",
        "camera_intrinsics_source": "3DPW cam_intrinsics",
        "official_sampling_note": "16 contiguous valid/cache samples; 3DPW img_frame_ids are every second source frame",
    }
    return images, merged, meta, sample


def load_adapters(device="cuda"):
    body = BodyFeatureTemporalAdapter().to(device)
    fw = FramewiseLatentAdapter().to(device)
    temporal = CameraFeatureTemporalAdapter().to(device)
    body.load_state_dict(torch.load(BODY_CKPT, map_location=device, weights_only=False)["adapter_state_dict"])
    fw.load_state_dict(torch.load(FW_CKPT, map_location=device, weights_only=False)["adapter_state_dict"])
    temporal.load_state_dict(torch.load(TEMP_CKPT, map_location=device, weights_only=False)["adapter_state_dict"])
    for module in (body, fw, temporal):
        module.eval().requires_grad_(False)
    return body, fw, temporal


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    started = time.time()
    gpu = gpu_snapshot()
    foreign = compute_processes()
    if gpu["pci.bus_id"].lower() != "00000000:61:00.0" or foreign:
        atomic_json(OUT / "environment.json", {"gpu": gpu, "compute_processes": foreign, "status": "BLOCKED_BY_SAFETY_GATE"})
        raise RuntimeError(f"Safety Gate failed: gpu={gpu}, processes={foreign}")

    hashes = {"body": sha256(BODY_CKPT), "framewise_camera": sha256(FW_CKPT), "temporal_camera": sha256(TEMP_CKPT)}
    if hashes != EXPECTED_HASHES:
        atomic_json(OUT / "environment.json", {"gpu": gpu, "checkpoint_hashes": hashes, "status": "CHECKPOINT_MISMATCH"})
        raise RuntimeError("checkpoint hash mismatch")

    env = {
        "status": "SAFETY_GATE_PASS", "gpu": gpu, "compute_processes_before": foreign,
        "pytorch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()), "python": platform.python_version(),
        "precision": "FP32 (foundation internal backbone dtype follows frozen model config)",
        "inference_mode": "torch.inference_mode()", "model_mode": "eval()",
        "cuda_synchronization": True, "checkpoint_hashes": hashes,
    }
    atomic_json(OUT / "environment.json", env)
    log("Safety Gate and checkpoint identity PASS")

    # Load the exact frozen foundation once and establish the parameter-count Gate.
    sam = load_frozen_model()
    sam.eval().requires_grad_(False)
    cpu_body = BodyFeatureTemporalAdapter()
    cpu_fw = FramewiseLatentAdapter()
    cpu_temp = CameraFeatureTemporalAdapter()
    counts = {
        "sam_body4d": nparams(sam), "body_restoration": nparams(cpu_body),
        "framewise_camera": nparams(cpu_fw), "temporal_camera": nparams(cpu_temp),
    }
    counts["added_total"] = counts["body_restoration"] + counts["framewise_camera"] + counts["temporal_camera"]
    counts["sam_plus_dualrest"] = counts["sam_body4d"] + counts["added_total"]
    counts["added_ratio_pct"] = 100.0 * counts["added_total"] / counts["sam_body4d"]
    counts["foundation_trainable"] = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    counts["foundation_frozen"] = sum(p.numel() for p in sam.parameters() if not p.requires_grad)
    counts["dualrest_trainable_when_training_its_stages"] = counts["added_total"]
    counts["all_frozen_during_inference"] = counts["sam_plus_dualrest"]
    counts["expected"] = EXPECTED_COUNTS
    counts["gate_pass"] = all(counts[k] == v for k, v in EXPECTED_COUNTS.items())
    atomic_json(OUT / "parameter_counts.json", counts)
    del cpu_body, cpu_fw, cpu_temp
    if not counts["gate_pass"]:
        raise RuntimeError(f"parameter count mismatch: {counts}")
    log("Parameter-count Gate PASS")

    # Estimator is used only for its formal transforms. Preprocessing stays outside timed regions.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / "configs/body4d.yaml")
    estimator = SAM3DBodyEstimator(sam, sam.cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    images, batch, meta, sample = prepare_validation_sample(estimator)
    atomic_json(OUT / "benchmark_sample.json", sample)
    log(f"Prepared validation sample {sample['sequence']}/person{sample['person_id']} frames={sample['frame_ids']}")

    def foundation_forward():
        sam._initialize_batch(batch)
        return sam.run_inference_batch(
            images, batch, inference_type="full", transform_hand=estimator.transform_hand,
            thresh_wrist_angle=estimator.thresh_wrist_angle, idx_path={}, idx_dict={},
            mhr_shape_scale_dict={}, id_batch=[[1] for _ in range(T)], occ_dict={1: [1] * T},
        )

    # One untimed smoke verifies exact outputs and obtains cached restoration inputs.
    with torch.inference_mode():
        foundation_out = foundation_forward()
        torch.cuda.synchronize()
    mhr = foundation_out[0]["mhr"]
    tokens = mhr["pose_token"].detach()
    init_pose = mhr["mhr_init_pose"].detach()
    points = mhr["pred_keypoints_3d"][:, MHR_IDX].detach()
    if tuple(tokens.shape) != (T, 1024):
        raise RuntimeError(f"unexpected foundation pose_token shape {tuple(tokens.shape)}")
    del foundation_out, mhr
    gc.collect(); torch.cuda.empty_cache()

    # Baseline E2E is measured before loading any DualRest parameters onto the GPU.
    e2e_baseline_rounds = []
    for r in range(ROUNDS):
        result = benchmark_cuda(foundation_forward, warmup=5, measured=20)
        e2e_baseline_rounds.append(result)
        log(f"E2E SAM round {r+1}: {result['mean_ms_per_window']:.3f} ms/window")
    baseline = aggregate_rounds(e2e_baseline_rounds)

    body, fw, temporal = load_adapters("cuda")
    condition = torch.full((1, T), 0.15, device="cuda", dtype=tokens.dtype)
    init_camera = sam.init_camera.weight.expand(T, -1)

    with torch.inference_mode():
        fw_token_cached = fw(tokens[None])
        features_cached = reliability_features(fw_token_cached, meta, 0.0)

    def body_path(tok=tokens, init=init_pose):
        refined = body(tok[None], condition).squeeze(0)
        return sam.head_pose(refined, init)

    def framewise_path(tok=tokens, pts=points):
        fw_token = fw(tok[None])
        cam = sam.head_camera(fw_token.squeeze(0), init_camera)
        return projection(sam, pts, cam, meta)

    def temporal_path(pts=points):
        final_token = temporal_residual_forward(temporal, fw_token_cached, features_cached)
        cam = sam.head_camera(final_token.squeeze(0), init_camera)
        return projection(sam, pts, cam, meta)

    def full_restoration(tok=tokens, init=init_pose, pts=points):
        refined_body = body(tok[None], condition).squeeze(0)
        body_out = sam.head_pose(refined_body, init)
        fw_token = fw(tok[None])
        fw_cam = sam.head_camera(fw_token.squeeze(0), init_camera)
        fw_proj = projection(sam, pts, fw_cam, meta)
        features = reliability_features(fw_token, meta, 0.0)
        final_token = temporal_residual_forward(temporal, fw_token, features)
        final_cam = sam.head_camera(final_token.squeeze(0), init_camera)
        final_proj = projection(sam, pts, final_cam, meta)
        return body_out, fw_proj, final_proj

    restoration = {}
    for name, fn in [
        ("body_restoration_plus_frozen_head_pose", body_path),
        ("framewise_camera_plus_frozen_head_camera_and_projection", framewise_path),
        ("temporal_camera_plus_frozen_head_camera_and_projection", temporal_path),
        ("full_dualrest_restoration_path", full_restoration),
    ]:
        rounds = []
        for r in range(ROUNDS):
            result = benchmark_cuda(fn, warmup=20, measured=100)
            rounds.append(result)
            log(f"Restoration {name} round {r+1}: {result['mean_ms_per_window']:.3f} ms/window")
        restoration[name] = aggregate_rounds(rounds)
    atomic_json(OUT / "restoration_overhead.json", restoration)

    def dualrest_e2e():
        output = foundation_forward()
        mm = output[0]["mhr"]
        tok = mm["pose_token"]
        init = mm["mhr_init_pose"]
        pts = mm["pred_keypoints_3d"][:, MHR_IDX]
        refined_body = body(tok[None], condition).squeeze(0)
        body_out = sam.head_pose(refined_body, init)
        fw_token = fw(tok[None])
        fw_cam = sam.head_camera(fw_token.squeeze(0), init_camera)
        fw_proj = projection(sam, pts, fw_cam, meta)
        features = reliability_features(fw_token, meta, 0.0)
        final_token = temporal_residual_forward(temporal, fw_token, features)
        final_cam = sam.head_camera(final_token.squeeze(0), init_camera)
        final_proj = projection(sam, pts, final_cam, meta)
        return output, body_out, fw_proj, final_proj

    dualrest_rounds = []
    for r in range(ROUNDS):
        result = benchmark_cuda(dualrest_e2e, warmup=5, measured=20)
        dualrest_rounds.append(result)
        log(f"E2E DualRest round {r+1}: {result['mean_ms_per_window']:.3f} ms/window")
    dualrest = aggregate_rounds(dualrest_rounds)
    latency_overhead = 100.0 * (dualrest["mean_ms_per_window"] - baseline["mean_ms_per_window"]) / baseline["mean_ms_per_window"]
    fps_change = 100.0 * (dualrest["fps_equivalent"] - baseline["fps_equivalent"]) / baseline["fps_equivalent"]
    memory_delta = dualrest["peak_allocated_bytes"] - baseline["peak_allocated_bytes"]
    e2e = {
        "sam_body4d": baseline, "dualrest_hmr": dualrest,
        "latency_overhead_pct": latency_overhead, "fps_change_pct": fps_change,
        "peak_allocated_memory_increase_bytes": memory_delta,
        "peak_allocated_memory_increase_pct": 100.0 * memory_delta / baseline["peak_allocated_bytes"],
        "timed_scope": "GPU model forward only; image disk I/O and CPU batch preparation excluded; outputs remain on GPU",
        "foundation_path": "formal SAM-Body4D run_inference_batch(inference_type=full)",
        "dualrest_path": "foundation + Body adapter/head_pose + Framewise adapter/head_camera/projection + Temporal adapter/head_camera/projection",
    }
    atomic_json(OUT / "end_to_end_efficiency.json", e2e)

    installed = {}
    for package in ("fvcore", "thop", "ptflops"):
        try:
            __import__(package); installed[package] = True
        except Exception:
            installed[package] = False
    flops = {
        "status": "PARTIAL_OR_UNSUPPORTED", "reported_flops_or_macs": None,
        "installed_profilers": installed, "torch_profiler_available": hasattr(torch, "profiler"),
        "reason": "No profiler in this environment is established to fully cover Mamba custom/fused CUDA kernels and all SAM-Body4D operations; no manual estimate is substituted.",
    }
    atomic_json(OUT / "flops_status.json", flops)

    protocol = {
        "batch_size_windows": 1, "window_length": T, "restoration_warmup": 20,
        "restoration_measured": 100, "e2e_warmup": 5, "e2e_measured": 20,
        "independent_rounds": ROUNDS, "cuda_sync_each_iteration": True,
        "torch_inference_mode": True, "eval_mode": True, "precision": env["precision"],
        "sample": sample, "exclusions": ["disk I/O", "CPU input preparation", "MHR-to-SMPL conversion", "evaluator", "file saving", "rendering"],
        "memory": "Peak allocated/reserved measured after warmup; E2E SAM measured before adapters are resident, DualRest with adapters resident.",
    }
    atomic_json(OUT / "benchmark_protocol.json", protocol)
    summary = {
        "status": "EFFICIENCY_BENCHMARK_PASS",
        "environment": env, "parameter_counts": counts,
        "restoration_only": restoration, "end_to_end": e2e, "flops": flops,
        "stability_pass": all(x["round_mean_variation_pct"] <= 5.0 for x in list(restoration.values()) + [baseline, dualrest]),
        "elapsed_seconds": time.time() - started,
        "compute_processes_after": compute_processes(),
    }
    atomic_json(OUT / "summary.json", summary)
    log(f"Completed with status={summary['status']} stability_pass={summary['stability_pass']}")


if __name__ == "__main__":
    main()
