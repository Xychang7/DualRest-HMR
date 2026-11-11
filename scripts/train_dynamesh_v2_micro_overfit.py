"""Bounded 16-frame DynaMesh4D-v2 feature-adapter micro-overfit."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root/sam-body4d-0.2.0")
CACHE = ROOT / "outputs/dynamesh_v2_phase1_cache_sanity/train/outdoors_slalom_00_person0_16f.npz"
OUTPUT = ROOT / "outputs/dynamesh_v2_phase1_micro_overfit"
SEED = 20260830

# Mirrors the official MHR continuous-body decoder indices. Hand articulation
# (parameters 62..115), jaw (130..132), and translations (124..129) are excluded.
THREEDOF = (
    (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20),
    (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39),
    (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79),
    (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100),
    (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132),
)
ONEDOF = (
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49,
    50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77,
    78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108,
    110, 111, 116, 117, 118, 119, 120, 121, 122, 123,
)
HAND_PARAMS = frozenset(range(62, 116))
BODY_3DOF_POSITIONS = tuple(i for i, indexes in enumerate(THREEDOF) if not (set(indexes) & HAND_PARAMS) and indexes != (130, 131, 132))
BODY_1DOF_POSITIONS = tuple(i for i, index in enumerate(ONEDOF) if index not in HAND_PARAMS)


def balanced_condition_schedule(num_steps: int, seed: int = SEED) -> list[int]:
    if num_steps % 4:
        raise ValueError("num_steps must be divisible by four")
    rng = random.Random(seed)
    schedule = []
    for _ in range(num_steps // 4):
        block = [0, 1, 2, 3]
        rng.shuffle(block)
        schedule.extend(block)
    return schedule


def decode_body_articulation_rotmats(body_cont: torch.Tensor) -> torch.Tensor:
    """Decode only non-hand/non-jaw MHR articulation into legal SO(3) matrices."""
    from sam_3d_body.models.modules.mhr_utils import compact_cont_to_rotmat_body

    packed = compact_cont_to_rotmat_body(body_cont)
    rot3 = packed[..., : len(THREEDOF) * 9].reshape(*packed.shape[:-1], len(THREEDOF), 3, 3)
    start = len(THREEDOF) * 9
    rot2 = packed[..., start : start + len(ONEDOF) * 4].reshape(*packed.shape[:-1], len(ONEDOF), 2, 2)
    selected3 = rot3[..., BODY_3DOF_POSITIONS, :, :]
    selected2 = rot2[..., BODY_1DOF_POSITIONS, :, :]
    embedded = torch.zeros(*selected2.shape[:-2], 3, 3, device=body_cont.device, dtype=body_cont.dtype)
    embedded[..., :2, :2] = selected2
    embedded[..., 2, 2] = 1
    return torch.cat((selected3, embedded), dim=-3)


def body_rotation_geodesic_loss(refined: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if teacher.requires_grad:
        raise ValueError("teacher rotations must be detached")
    relative = refined @ teacher.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine).mean()


def assert_optimizer_owns_only(optimizer, module: torch.nn.Module) -> None:
    owned = {id(p) for p in module.parameters() if p.requires_grad}
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not owned or optimized != owned:
        raise AssertionError("optimizer parameters must exactly equal trainable Body adapter parameters")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def train(steps: int, learning_rate: float) -> None:
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "models/sam_3d_body"))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
    from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    cache = np.load(CACHE, allow_pickle=False)
    if str(cache["split"]) != "train" or str(cache["sequence"]) != "outdoors_slalom_00" or int(cache["person_id"]) != 0:
        raise RuntimeError("micro-overfit cache identity/split mismatch")
    if cache["pose_token_clean"].shape != (16, 1024) or cache["pose_token_occluded"].shape != (3, 16, 1024):
        raise RuntimeError("micro-overfit cache shape mismatch")

    config = OmegaConf.load(ROOT / "configs/body4d.yaml")
    model, model_config = load_sam_3d_body(config.sam_3d_body.ckpt_path, device=device, mhr_path=config.sam_3d_body.mhr_path)
    estimator = SAM3DBodyEstimator(model, model_config, human_detector=None, human_segmentor=None, fov_estimator=None)
    for parameter in estimator.model.parameters():
        parameter.requires_grad_(False)
    estimator.model.eval()
    head = estimator.model.head_pose

    adapter = BodyFeatureTemporalAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    assert_optimizer_owns_only(optimizer, adapter)
    if any(parameter.requires_grad for parameter in estimator.model.parameters()):
        raise AssertionError("SAM/MHR parameters are not fully frozen")

    clean = torch.as_tensor(cache["pose_token_clean"], device=device)
    occluded = torch.as_tensor(cache["pose_token_occluded"], device=device)
    inputs = torch.cat((clean.unsqueeze(0), occluded), dim=0)
    ratios = torch.tensor([0.0, *cache["occlusion_ratio"].tolist()], device=device)
    body_params = torch.cat(
        (
            torch.as_tensor(cache["baseline_body_pose"], device=device).unsqueeze(0),
            torch.as_tensor(cache["occluded_body_pose"], device=device),
        ), dim=0,
    )
    baseline_cont = compact_model_params_to_cont_body(body_params)
    with torch.no_grad():
        clean_teacher_cont = compact_model_params_to_cont_body(body_params[0]).detach()
        clean_teacher_rot = decode_body_articulation_rotmats(clean_teacher_cont).detach()
        raw_proj_cont = torch.stack([head.proj(tokens)[:, 6:266] for tokens in inputs]).detach()

    schedule = balanced_condition_schedule(steps)
    history = []
    best_loss = math.inf
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "seed": SEED, "steps": steps, "learning_rate": learning_rate,
        "optimizer": "AdamW", "weight_decay": 1e-4,
        "lambda_feature": 1.0, "lambda_body": 0.1, "lambda_anchor": 1.0,
        "sequence": str(cache["sequence"]), "person_id": int(cache["person_id"]),
        "frame_ids": cache["frame_id"].tolist(), "cache_schema": str(cache["schema_version"]),
        "condition_ratios": ratios.detach().cpu().tolist(),
        "adapter": {"input_dim": 1024, "d_model": 256, "n_layers": 2, "d_state": 16, "d_conv": 4, "expand": 2},
        "trainable_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
    }
    (OUTPUT / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

    for step, condition_index in enumerate(schedule, start=1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        input_tokens = inputs[condition_index]
        ratio = torch.full((16,), ratios[condition_index], device=device)
        refined = adapter(input_tokens, ratio)
        refined_clean = adapter(clean, torch.zeros(16, device=device))
        refined_projection = head.proj(refined)[:, 6:266]
        refined_cont = baseline_cont[condition_index] + (refined_projection - raw_proj_cont[condition_index])
        refined_rot = decode_body_articulation_rotmats(refined_cont)
        feature_loss = F.smooth_l1_loss(refined, clean)
        body_loss = body_rotation_geodesic_loss(refined_rot, clean_teacher_rot)
        anchor_loss = F.smooth_l1_loss(refined_clean, clean)
        total = feature_loss + 0.1 * body_loss + anchor_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        projection_grad = float(adapter.output_projection.weight.grad.norm())
        mamba_grad = float(sum((p.grad.norm() for layer in adapter.layers_fwd for p in layer.parameters() if p.grad is not None), torch.zeros((), device=device)))
        input_grad = float(sum((p.grad.norm() for p in adapter.input_projection.parameters() if p.grad is not None), torch.zeros((), device=device)))
        optimizer.step()
        item = {
            "step": step, "condition_index": condition_index, "occlusion_ratio": float(ratios[condition_index]),
            "total": float(total.detach()), "feature": float(feature_loss.detach()),
            "body": float(body_loss.detach()), "anchor": float(anchor_loss.detach()),
            "grad_output_projection": projection_grad, "grad_mamba_forward": mamba_grad,
            "grad_input_projection": input_grad, "learning_rate": learning_rate,
        }
        history.append(item)
        if item["total"] < best_loss:
            best_loss = item["total"]
            torch.save({"adapter_state_dict": adapter.state_dict(), "config": config_payload, "best_step": step, "best_loss": best_loss}, OUTPUT / "best.pt")
        if step == 1 or step % 40 == 0:
            print(json.dumps(item), flush=True)

    torch.save({"adapter_state_dict": adapter.state_dict(), "config": config_payload, "final_step": steps, "final_loss": history[-1]["total"]}, OUTPUT / "final.pt")
    (OUTPUT / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"best_loss": best_loss, "final": history[-1], "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.steps, args.learning_rate)
