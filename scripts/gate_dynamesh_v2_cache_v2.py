"""Engineering gates before full DynaMesh4D-v2 cache generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path("/root/sam-body4d-0.2.0")
DATA = Path("/root/data1/sam-body4d-master/dataset/3DPW")
CACHE = ROOT / "outputs/dynamesh_v2_phase1_cache_sanity/train/outdoors_slalom_00_person0_16f.npz"
OUT = ROOT / "outputs/dynamesh_v2_phase2_gates"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models/sam_3d_body")); sys.path.insert(0, str(ROOT / "scripts"))

from build_dynamesh_v2_cache_sanity import BASE_SEED, apply_bbox_rectangle_occlusion, bbox_from_pose2d, load_pickle, run_sequence
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body
from train_dynamesh_v2_micro_overfit import decode_body_articulation_rotmats


def load_estimator():
    config = OmegaConf.load(ROOT / "configs/body4d.yaml")
    model, model_config = load_sam_3d_body(config.sam_3d_body.ckpt_path, device=torch.device("cuda"), mhr_path=config.sam_3d_body.mhr_path)
    estimator = SAM3DBodyEstimator(model, model_config, human_detector=None, human_segmentor=None, fov_estimator=None)
    estimator.model.eval()
    for parameter in estimator.model.parameters(): parameter.requires_grad_(False)
    return estimator


def recreate_one_frame_variants(estimator, cache):
    sequence, person = str(cache["sequence"]), int(cache["person_id"])
    annotation = load_pickle(DATA / "sequenceFiles/train" / f"{sequence}.pkl")
    local = int(cache["local_frame_index"][0]); frame_id = int(cache["frame_id"][0])
    path = DATA / "imageFiles" / sequence / f"image_{local:05d}.jpg"
    image = cv2.imread(str(path)); height, width = image.shape[:2]
    bbox = bbox_from_pose2d(np.asarray(annotation["poses2d"][person])[local], width, height)
    cam = torch.as_tensor(np.asarray(annotation["cam_intrinsics"], dtype=np.float32), device="cuda").unsqueeze(0)
    outputs = [run_sequence(estimator, [str(path)], bbox[None], cam)[0]]
    with tempfile.TemporaryDirectory(prefix="dynav2_init_gate_") as directory:
        for level, ratio in enumerate((0.15, 0.35, 0.55)):
            occluded, _ = apply_bbox_rectangle_occlusion(image, bbox, ratio, seed=BASE_SEED + level * 100000 + frame_id, fill_value=0)
            target = Path(directory) / f"occ{level}.png"; cv2.imwrite(str(target), occluded)
            outputs.append(run_sequence(estimator, [str(target)], bbox[None], cam)[0])
    return outputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE, allow_pickle=False)
    estimator = load_estimator(); head = estimator.model.head_pose
    outputs = recreate_one_frame_variants(estimator, cache)
    init = np.stack([item["mhr_init_pose"] for item in outputs]).astype(np.float32)
    tokens_live = np.stack([item["pose_token"] for item in outputs]).astype(np.float32)
    pairwise_init_max = float(max(np.abs(init[a] - init[b]).max() for a in range(4) for b in range(a + 1, 4)))
    live_pose_raw = []
    with torch.no_grad():
        for item in outputs:
            token = torch.as_tensor(item["pose_token"], device="cuda").unsqueeze(0)
            state = torch.as_tensor(item["mhr_init_pose"], device="cuda").unsqueeze(0)
            live_pose_raw.append(head(token, state)["pred_pose_raw"].squeeze(0).cpu().numpy())
    live_pose_raw_error = float(np.max(np.abs(np.stack(live_pose_raw) - np.stack([item["pred_pose_raw"] for item in outputs]))))

    tokens32 = np.concatenate((cache["pose_token_clean"][None], cache["pose_token_occluded"]), axis=0).astype(np.float32)
    tokens16 = tokens32.astype(np.float16).astype(np.float32)
    flat32 = torch.as_tensor(tokens32.reshape(-1, 1024), device="cuda")
    flat16 = torch.as_tensor(tokens16.reshape(-1, 1024), device="cuda")
    learned_init = estimator.model.init_pose.weight.detach().expand(flat32.shape[0], -1)
    with torch.no_grad():
        decoded32 = head(flat32, learned_init)["body_pose"]
        decoded16 = head(flat16, learned_init)["body_pose"]
        cont32 = compact_model_params_to_cont_body(decoded32)
        cont16 = compact_model_params_to_cont_body(decoded16)
        rot32 = decode_body_articulation_rotmats(cont32)
        rot16 = decode_body_articulation_rotmats(cont16)
        relative = rot16 @ rot32.transpose(-1, -2)
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1, 1)
        geodesic_deg = torch.rad2deg(torch.acos(cosine))
    token_delta = tokens16 - tokens32
    body_delta = decoded16.cpu().numpy() - decoded32.cpu().numpy()
    result = {
        "init_state": {
            "field": "mhr_init_pose", "shape_four_variants": list(init.shape),
            "per_frame_shape": list(init.shape[1:]),
            "source": "forward_decoder init_pose.weight because formal calls pass init_estimate=None",
            "pairwise_variant_max_abs": pairwise_init_max,
            "formal_redecode_pred_pose_raw_max_abs": live_pose_raw_error,
            "variant_specific_capture_required": True,
            "deduplication_allowed": bool(pairwise_init_max == 0.0),
        },
        "float16": {
            "num_tokens": int(np.prod(tokens32.shape[:-1])),
            "token_max_abs": float(np.abs(token_delta).max()),
            "token_mean_abs": float(np.abs(token_delta).mean()),
            "token_l2_mean": float(np.linalg.norm(token_delta, axis=-1).mean()),
            "body_param_max_abs": float(np.abs(body_delta).max()),
            "body_param_mean_abs": float(np.abs(body_delta).mean()),
            "body_rotation_geodesic_deg_mean": float(geodesic_deg.mean()),
            "body_rotation_geodesic_deg_max": float(geodesic_deg.max()),
        },
    }
    result["init_state"]["passed"] = bool(init.shape == (4, 519) and pairwise_init_max == 0 and live_pose_raw_error < 2e-5)
    result["float16"]["passed"] = bool(result["float16"]["token_l2_mean"] < .02 and result["float16"]["body_param_max_abs"] < 1e-3 and result["float16"]["body_rotation_geodesic_deg_mean"] < .05)
    result["passed"] = result["init_state"]["passed"] and result["float16"]["passed"]
    (OUT / "init_float16_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]: raise SystemExit(2)


if __name__ == "__main__": main()
