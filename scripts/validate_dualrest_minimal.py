#!/usr/bin/env python3
"""Run one real 16-frame DualRest-HMR validation sequence without training.

The script is intentionally standalone: repository paths are derived from this
file, while foundation checkpoints and datasets are explicit command-line
inputs.  Validation outputs must be outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "models" / "sam_3d_body")]

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402
from sam_3d_body.models.modules.camera_temporal import (  # noqa: E402
    CameraFeatureTemporalAdapter,
)
from sam_3d_body.models.modules.dualrest_inference import (  # noqa: E402
    FramewiseLatentAdapter,
    restore_camera_tokens,
)
from sam_3d_body.models.modules.dynamesh_v2_integration import (  # noqa: E402
    compose_parameters,
    rebuild_local_vertices,
)
from sam_3d_body.models.modules.dynamesh_v2_temporal import (  # noqa: E402
    BodyFeatureTemporalAdapter,
)


EXPECTED_CHECKPOINTS = {
    "body": "f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329",
    "framewise": "fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2",
    "temporal": "e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="courtyard_basketball_01")
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--occlusion-ratio", type=float, default=0.0)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--protected-source-root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bbox_from_pose2d(
    pose2d: np.ndarray, image_width: int, image_height: int, expansion: float = 1.2
) -> np.ndarray:
    pose2d = np.asarray(pose2d, dtype=np.float32)
    valid = np.isfinite(pose2d).all(axis=0) & (pose2d[2] > 0)
    if valid.sum() < 2:
        raise ValueError("fewer than two valid GT 2D joints")
    x, y = pose2d[0, valid], pose2d[1, valid]
    cx, cy = float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2)
    width = max(float(x.max() - x.min()) * expansion, 16.0)
    height = max(float(y.max() - y.min()) * expansion, 16.0)
    return np.asarray(
        [
            max(0.0, cx - width / 2),
            max(0.0, cy - height / 2),
            min(float(image_width), cx + width / 2),
            min(float(image_height), cy + height / 2),
        ],
        dtype=np.float32,
    )


def load_annotation(args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    path = args.dataset_root / "sequenceFiles" / args.split / f"{args.sequence}.pkl"
    with path.open("rb") as handle:
        annotation = pickle.load(handle, encoding="latin1")
    valid = np.asarray(annotation["campose_valid"][args.person_id], dtype=bool)
    selected = np.flatnonzero(valid)[: args.frames]
    if selected.size != args.frames:
        raise RuntimeError(f"sequence has only {selected.size} valid frames")
    return annotation, selected


def prepare_inputs(
    args: argparse.Namespace, annotation: dict, selected: np.ndarray
) -> tuple[list[str], np.ndarray, torch.Tensor, dict]:
    image_dir = args.dataset_root / "imageFiles" / args.sequence
    image_paths = [str(image_dir / f"image_{index:05d}.jpg") for index in selected]
    first = cv2.imread(image_paths[0])
    if first is None:
        raise FileNotFoundError(image_paths[0])
    height, width = first.shape[:2]
    poses2d = np.asarray(annotation["poses2d"][args.person_id])
    bboxes = np.stack(
        [bbox_from_pose2d(poses2d[index], width, height) for index in selected]
    )
    intrinsic = torch.as_tensor(
        np.asarray(annotation["cam_intrinsics"], dtype=np.float32), device="cuda"
    ).unsqueeze(0)
    center = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    size = (bboxes[:, 2] - bboxes[:, 0]) * 1.25
    meta = {
        "center": center.astype(np.float32),
        "size": size.astype(np.float32),
        "image_size": np.tile(
            np.asarray([[width, height]], dtype=np.float32), (len(selected), 1)
        ),
        "K": np.asarray(annotation["cam_intrinsics"], dtype=np.float32),
    }
    return image_paths, bboxes, intrinsic, meta


def load_adapter(name: str, module: torch.nn.Module, path: Path) -> torch.nn.Module:
    actual = sha256(path)
    if actual != EXPECTED_CHECKPOINTS[name]:
        raise RuntimeError(f"{name} checkpoint SHA-256 mismatch: {actual}")
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    module.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    return module.cuda().eval().requires_grad_(False)


def run_foundation(
    estimator: SAM3DBodyEstimator,
    image_paths: list[str],
    bboxes: np.ndarray,
    intrinsic: torch.Tensor,
    batch_size: int,
) -> list[dict]:
    outputs: list[dict] = []
    shape_scale_cache: dict = {}
    for start in range(0, len(image_paths), batch_size):
        stop = min(start + batch_size, len(image_paths))
        batch = estimator.process_frames(
            image_paths[start:stop],
            bboxes=[box.reshape(1, 4) for box in bboxes[start:stop]],
            masks=None,
            id_batch=[[1] for _ in range(stop - start)],
            idx_path={},
            idx_dict={},
            mhr_shape_scale_dict=shape_scale_cache,
            occ_dict={1: [1] * (stop - start)},
            cam_int=intrinsic,
            use_mask=False,
        )
        outputs.extend(frame[0] for frame in batch)
        print(f"foundation_frames={len(outputs)}/{len(image_paths)}", flush=True)
    if len(outputs) != len(image_paths):
        raise RuntimeError("foundation frame count mismatch")
    return outputs


def finite_summary(name: str, value: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(value).all())
    if not finite:
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
    }


def module_provenance(protected_source_root: Path | None) -> dict:
    copy_modules, protected_modules = {}, {}
    protected = (
        str(protected_source_root.resolve()) if protected_source_root is not None else None
    )
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        resolved = str(Path(filename).resolve())
        if resolved.startswith(str(REPO_ROOT)):
            copy_modules[name] = resolved
        if protected and resolved.startswith(protected):
            protected_modules[name] = resolved
    if protected_modules:
        raise RuntimeError(
            f"protected project source modules were imported: {protected_modules}"
        )
    return {"copy_modules": copy_modules, "protected_source_modules": {}}


def main() -> None:
    args = parse_args()
    started = time.time()
    if REPO_ROOT in args.output_dir.resolve().parents or args.output_dir.resolve() == REPO_ROOT:
        raise ValueError("output-dir must be outside the repository")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["DUALREST_CHECKPOINT_ROOT"] = str(
        args.foundation_checkpoint_root.resolve()
    )
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "body4d.yaml")
    annotation, selected = load_annotation(args)
    image_paths, bboxes, intrinsic, meta = prepare_inputs(args, annotation, selected)

    device = torch.device("cuda")
    model, model_cfg = load_sam_3d_body(
        str(cfg.sam_3d_body.ckpt_path),
        device=device,
        mhr_path=str(cfg.sam_3d_body.mhr_path),
    )
    model.eval().requires_grad_(False)
    estimator = SAM3DBodyEstimator(
        model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    adapters = {
        "body": load_adapter(
            "body",
            BodyFeatureTemporalAdapter(),
            REPO_ROOT / "checkpoints" / "dualrest" / "body" / "best.pt",
        ),
        "framewise": load_adapter(
            "framewise",
            FramewiseLatentAdapter(),
            REPO_ROOT
            / "checkpoints"
            / "dualrest"
            / "camera"
            / "framewise"
            / "best.pt",
        ),
        "temporal": load_adapter(
            "temporal",
            CameraFeatureTemporalAdapter(),
            REPO_ROOT
            / "checkpoints"
            / "dualrest"
            / "camera"
            / "temporal"
            / "best.pt",
        ),
    }
    outputs = run_foundation(
        estimator, image_paths, bboxes, intrinsic, args.batch_size
    )
    raw_token = torch.as_tensor(
        np.stack([item["pose_token"] for item in outputs]),
        device=device,
        dtype=torch.float32,
    )
    init_pose = torch.as_tensor(
        np.stack([item["mhr_init_pose"] for item in outputs]),
        device=device,
        dtype=torch.float32,
    )
    points3d = torch.as_tensor(
        np.stack([item["pred_keypoints_3d"] for item in outputs]),
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        body_token = adapters["body"](
            raw_token,
            torch.full(
                (len(raw_token),),
                float(args.occlusion_ratio),
                device=device,
            ),
        )
        raw_pose = model.head_pose(raw_token, init_pose)
        body_pose = model.head_pose(body_token, init_pose)
        framewise_token, temporal_token = restore_camera_tokens(
            adapters["framewise"],
            adapters["temporal"],
            raw_token,
            meta,
            args.occlusion_ratio,
        )
        camera_init = model.init_camera.weight.expand(len(raw_token), -1)
        raw_camera = model.head_camera(raw_token, camera_init)
        framewise_camera = model.head_camera(framewise_token.squeeze(0), camera_init)
        temporal_camera = model.head_camera(temporal_token.squeeze(0), camera_init)
        center = torch.as_tensor(meta["center"], device=device)
        size = torch.as_tensor(meta["size"], device=device)
        image_size = torch.as_tensor(meta["image_size"], device=device)
        camera_intrinsic = torch.as_tensor(
            np.broadcast_to(meta["K"], (len(raw_token), 3, 3)).copy(), device=device
        )
        raw_projection = model.head_camera.perspective_projection(
            points3d, raw_camera, center, size, image_size, camera_intrinsic
        )
        temporal_projection = model.head_camera.perspective_projection(
            points3d, temporal_camera, center, size, image_size, camera_intrinsic
        )
        raw_parameters = dict(raw_pose)
        raw_parameters.update(
            pred_cam=raw_camera, pred_cam_t=raw_projection["pred_cam_t"]
        )
        final_parameters = compose_parameters(
            raw_parameters,
            body_pose,
            {
                "pred_cam": temporal_camera,
                "pred_cam_t": temporal_projection["pred_cam_t"],
            },
        )
        final_vertices = rebuild_local_vertices(model.head_pose, final_parameters)

    tensors = {
        "raw_token": raw_token,
        "body_token": body_token,
        "framewise_token": framewise_token,
        "temporal_token": temporal_token,
        "raw_vertices": raw_pose["pred_vertices"],
        "body_vertices": body_pose["pred_vertices"],
        "final_vertices": final_vertices,
        "raw_camera": raw_camera,
        "framewise_camera": framewise_camera,
        "temporal_camera": temporal_camera,
        "raw_camera_translation": raw_projection["pred_cam_t"],
        "temporal_camera_translation": temporal_projection["pred_cam_t"],
    }
    summaries = {name: finite_summary(name, value) for name, value in tensors.items()}
    reference = None
    if args.reference_cache is not None:
        with np.load(args.reference_cache, allow_pickle=False) as cache:
            cached_ids = np.asarray(cache["frame_id"])
            expected_ids = np.asarray(annotation["img_frame_ids"])[selected]
            lookup = {int(value): index for index, value in enumerate(cached_ids)}
            indices = [lookup[int(value)] for value in expected_ids]
            cached = np.asarray(cache["pose_token"][0, indices], dtype=np.float32)
        difference = raw_token.detach().cpu().numpy() - cached
        reference = {
            "path": str(args.reference_cache),
            "frame_ids_match": True,
            "token_max_abs_difference": float(np.max(np.abs(difference))),
            "token_mean_abs_difference": float(np.mean(np.abs(difference))),
        }

    frame_ids = np.asarray(annotation["img_frame_ids"])[selected]
    np.savez_compressed(
        args.output_dir / "minimal_outputs.npz",
        frame_id=frame_ids,
        local_frame_index=selected,
        bboxes=bboxes,
        **{name: value.detach().cpu().numpy() for name, value in tensors.items()},
    )
    provenance = module_provenance(args.protected_source_root)
    report = {
        "state": "complete",
        "repo_root": str(REPO_ROOT),
        "cwd": str(Path.cwd()),
        "sequence": args.sequence,
        "person_id": args.person_id,
        "split": args.split,
        "frame_ids": frame_ids.astype(int).tolist(),
        "local_frame_index": selected.astype(int).tolist(),
        "input_images": image_paths,
        "occlusion_ratio": float(args.occlusion_ratio),
        "foundation_checkpoint_root": str(args.foundation_checkpoint_root),
        "checkpoints": {
            name: {
                "path": str(
                    REPO_ROOT
                    / "checkpoints"
                    / "dualrest"
                    / (
                        "body/best.pt"
                        if name == "body"
                        else f"camera/{name}/best.pt"
                    )
                ),
                "sha256": EXPECTED_CHECKPOINTS[name],
            }
            for name in EXPECTED_CHECKPOINTS
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "mamba_ssm": importlib.metadata.version("mamba-ssm"),
            "smplx": importlib.metadata.version("smplx"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        },
        "outputs": summaries,
        "reference_cache_equivalence": reference,
        "module_provenance": provenance,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "validation_report.json", report)
    print(json.dumps({"state": "complete", "output": str(args.output_dir), "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
