"""Three-phase frozen evaluation of the 16-frame Dyna-v2 micro-overfit."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/sam-body4d-0.2.0")
OUT = ROOT / "outputs/dynamesh_v2_phase1_micro_overfit"
CACHE = ROOT / "outputs/dynamesh_v2_phase1_cache_sanity/train/outdoors_slalom_00_person0_16f.npz"
DATA = Path("/root/data1/sam-body4d-master/dataset/3DPW")
SMPL_MODEL = Path("/root/DuoMo-main/data/body_models/smpl/SMPL_NEUTRAL.pkl")
REGRESSOR = Path("/root/DuoMo-main/data/body_models/J_regressor_h36m.npy")
RATIOS = (0.0, 0.15, 0.35, 0.55)


def _save(path, value):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def _load(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def infer():
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "models/sam_3d_body"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_dynamesh_v2_cache_sanity import (
        BASE_SEED, apply_bbox_rectangle_occlusion, bbox_from_pose2d, load_pickle, run_sequence,
    )
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
    from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body
    from train_dynamesh_v2_micro_overfit import decode_body_articulation_rotmats

    device = torch.device("cuda")
    cfg = OmegaConf.load(ROOT / "configs/body4d.yaml")
    model, model_cfg = load_sam_3d_body(cfg.sam_3d_body.ckpt_path, device=device, mhr_path=cfg.sam_3d_body.mhr_path)
    estimator = SAM3DBodyEstimator(model, model_cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    for parameter in estimator.model.parameters():
        parameter.requires_grad_(False)
    estimator.model.eval()
    head = estimator.model.head_pose

    checkpoint = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
    adapter = BodyFeatureTemporalAdapter().to(device)
    incompatible = adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    adapter.eval()

    cache = np.load(CACHE, allow_pickle=False)
    sequence, person_id = str(cache["sequence"]), int(cache["person_id"])
    annotation = load_pickle(DATA / "sequenceFiles/train" / f"{sequence}.pkl")
    selected = cache["local_frame_index"].astype(int)
    frame_ids = cache["frame_id"].astype(int)
    image_dir = DATA / "imageFiles" / sequence
    image_paths = [str(image_dir / f"image_{index:05d}.jpg") for index in selected]
    image = cv2.imread(image_paths[0]); height, width = image.shape[:2]
    poses2d = np.asarray(annotation["poses2d"][person_id])
    bboxes = np.stack([bbox_from_pose2d(poses2d[index], width, height) for index in selected])
    cam_int_np = np.asarray(annotation["cam_intrinsics"], dtype=np.float32)
    cam_int = torch.as_tensor(cam_int_np, device=device).unsqueeze(0)
    all_outputs = [run_sequence(estimator, image_paths, bboxes, cam_int)]
    with tempfile.TemporaryDirectory(prefix="dynav2_eval_") as temp:
        temp = Path(temp)
        for level, ratio in enumerate(RATIOS[1:]):
            paths = []
            for index, (path, bbox, frame_id) in enumerate(zip(image_paths, bboxes, frame_ids)):
                source = cv2.imread(path)
                occluded, _ = apply_bbox_rectangle_occlusion(source, bbox, ratio, seed=BASE_SEED + level * 100000 + int(frame_id), fill_value=0)
                target = temp / f"level{level}_{index:05d}.png"
                cv2.imwrite(str(target), occluded); paths.append(str(target))
            all_outputs.append(run_sequence(estimator, paths, bboxes, cam_int))

    cache_tokens = np.concatenate((cache["pose_token_clean"][None], cache["pose_token_occluded"]), axis=0)
    variants, diagnostics = {}, {}
    clean_teacher_rot = None
    for condition, (ratio, outputs) in enumerate(zip(RATIOS, all_outputs)):
        def stack(name):
            return torch.as_tensor(np.stack([item[name] for item in outputs]), device=device)
        raw = stack("pose_token")
        token_cache_max_diff = float(np.max(np.abs(raw.detach().cpu().numpy() - cache_tokens[condition])))
        init = stack("mhr_init_pose")
        with torch.no_grad():
            refined = adapter(raw, torch.full((len(outputs),), ratio, device=device))
            raw_decoded = head(raw, init)
            refined_decoded = head(refined, init)
            baseline_body = stack("body_pose_params")
            final_body = baseline_body + refined_decoded["body_pose"] - raw_decoded["body_pose"]
            baseline = stack("pred_vertices")
            rebuilt = head.mhr_forward(
                global_trans=torch.zeros_like(stack("global_rot")), global_rot=stack("global_rot"),
                body_pose_params=final_body, hand_pose_params=stack("hand_pose_params"),
                scale_params=stack("scale_params"), shape_params=stack("shape_params"),
                expr_params=stack("expr_params"), do_pcblend=True, return_keypoints=False,
                return_joint_coords=False, return_model_params=False, return_joint_rotations=False,
            )
            rebuilt[..., [1, 2]] *= -1
            refined_cont = compact_model_params_to_cont_body(final_body)
            rotations = decode_body_articulation_rotmats(refined_cont)
            baseline_rotations = decode_body_articulation_rotmats(
                compact_model_params_to_cont_body(baseline_body)
            )
            if condition == 0:
                clean_teacher_rot = baseline_rotations.detach()
            relative = rotations @ clean_teacher_rot.transpose(-1, -2)
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
            geodesic_deg = float(torch.rad2deg(torch.acos(cosine)).mean())
            baseline_relative = baseline_rotations @ clean_teacher_rot.transpose(-1, -2)
            baseline_cosine = ((baseline_relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
            baseline_geodesic_deg = float(torch.rad2deg(torch.acos(baseline_cosine)).mean())
        if not torch.isfinite(rebuilt).all():
            raise FloatingPointError("rebuilt vertices contain NaN/Inf")
        baseline_distance = np.linalg.norm(raw.detach().cpu().numpy() - cache_tokens[0], axis=-1).mean()
        trained_distance = np.linalg.norm(refined.detach().cpu().numpy() - cache_tokens[0], axis=-1).mean()
        label = "clean" if ratio == 0 else f"occlusion_{ratio:.2f}"
        variants[f"baseline_{label}"] = baseline.detach().cpu().numpy().astype(np.float32)
        variants[f"trained_{label}"] = rebuilt.detach().cpu().numpy().astype(np.float32)
        diagnostics[label] = {
            "token_cache_max_abs_diff": token_cache_max_diff,
            "baseline_token_l2": float(baseline_distance), "trained_token_l2": float(trained_distance),
            "token_recovery_percent": float(0 if baseline_distance == 0 else 100 * (baseline_distance - trained_distance) / baseline_distance),
            "baseline_body_teacher_geodesic_deg": baseline_geodesic_deg,
            "trained_body_teacher_geodesic_deg": geodesic_deg,
        }
    payload = {
        "sequence": sequence, "person_id": person_id, "frame_id": frame_ids,
        "variants": variants, "diagnostics": diagnostics,
        "parameter_sources": {"body": "trained_adapter", "global_hand_shape_scale_expression_camera": "raw_SAM_Body4D"},
    }
    _save(OUT / "evaluation_mhr.pkl", payload)
    (OUT / "feature_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


def convert():
    import torch
    sys.path.insert(0, "/root/MHR-main")
    sys.path.insert(0, "/root/MHR-main/tools/mhr_smpl_conversion")
    import smplx
    from conversion import Conversion
    from mhr.mhr import MHR

    source = _load(OUT / "evaluation_mhr.pkl")
    names = list(source["variants"])
    stacked = np.concatenate([source["variants"][name] for name in names], axis=0)
    converter = Conversion(mhr_model=MHR.from_files(lod=1, device=torch.device("cpu")), smpl_model=smplx.SMPL(model_path=str(SMPL_MODEL)).to("cpu"), method="pytorch")
    result = converter.convert_mhr2smpl(
        mhr_vertices=np.ascontiguousarray(stacked * np.float32(100.0)), single_identity=True,
        return_smpl_vertices=True, return_smpl_parameters=False, return_fitting_errors=True,
    )
    vertices = np.asarray(result.result_vertices, dtype=np.float32).reshape(len(names), 16, 6890, 3)
    errors = np.asarray(result.result_errors, dtype=np.float32).reshape(len(names), 16)
    if not np.isfinite(vertices).all():
        raise FloatingPointError("converted vertices contain NaN/Inf")
    _save(OUT / "evaluation_smpl.pkl", {"names": names, "vertices": vertices, "errors": errors, "meta": source})
    print(json.dumps({name: float(error.mean() * 1000) for name, error in zip(names, errors)}, indent=2))


def evaluate():
    sys.path.insert(0, str(ROOT))
    from eval_3dpw_standard import build_gt_records, evaluate_records

    data = _load(OUT / "evaluation_smpl.pkl")
    meta = data["meta"]
    keys = [(meta["sequence"], meta["person_id"], int(frame)) for frame in meta["frame_id"]]
    gt = build_gt_records(keys, gt_dir=DATA / "sequenceFiles/train", smpl_model_path=SMPL_MODEL, device="cpu")
    regressor = np.load(REGRESSOR)
    metrics = {}
    for name, vertices, errors in zip(data["names"], data["vertices"], data["errors"]):
        records = [{"sequence": seq, "person_id": pid, "frame_id": frame, "smpl_vertices": mesh} for (seq, pid, frame), mesh in zip(keys, vertices)]
        result = evaluate_records(records, gt, regressor)
        metrics[name] = {"mpjpe_mm": result["mpjpe_mm"], "pa_mpjpe_mm": result["pa_mpjpe_mm"], "converter_fitting_error_mm": float(errors.mean() * 1000), "finite": True}
    for ratio in RATIOS:
        label = "clean" if ratio == 0 else f"occlusion_{ratio:.2f}"
        for metric in ("mpjpe_mm", "pa_mpjpe_mm"):
            metrics[f"trained_{label}"][f"delta_vs_baseline_{metric}"] = metrics[f"trained_{label}"][metric] - metrics[f"baseline_{label}"][metric]
    (OUT / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("infer", "convert", "evaluate"))
    phase = parser.parse_args().phase
    {"infer": infer, "convert": convert, "evaluate": evaluate}[phase]()
