"""Frozen Body/Camera composition contract for DynaMesh4D-v2 Phase-3."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import torch

BODY_KEYS = ("body_pose", "body_pose_params")
CAMERA_KEYS = ("pred_cam", "pred_cam_t")
RAW_ONLY_KEYS = (
    "global_rot", "hand", "hand_pose_params", "scale", "scale_params",
    "shape", "shape_params", "face", "expr_params", "jaw",
)


def _one_present(source: Mapping, keys: tuple[str, ...], label: str) -> str:
    found = [key for key in keys if key in source]
    if len(found) != 1:
        raise KeyError(f"{label} must provide exactly one of {keys}, got {found}")
    return found[0]


def compose_parameters(raw: Mapping, body_refined: Mapping | None, camera_refined: Mapping | None):
    """Compose branches without allowing cross-branch parameter leakage."""
    result = dict(raw)
    if body_refined is not None:
        raw_body = _one_present(raw, BODY_KEYS, "raw")
        refined_body = _one_present(body_refined, BODY_KEYS, "body branch")
        result[raw_body] = body_refined[refined_body]
    if camera_refined is not None:
        for key in CAMERA_KEYS:
            if key not in raw or key not in camera_refined:
                raise KeyError(f"raw and camera branch must both provide {key}")
            result[key] = camera_refined[key]
    for key in RAW_ONLY_KEYS:
        if key in raw and result[key] is not raw[key]:
            raise AssertionError(f"raw-only field was replaced: {key}")
    assert_finite(result)
    return result


def assert_track_alignment(left: Mapping, right: Mapping) -> None:
    if left["sequence"] != right["sequence"]:
        raise ValueError("sequence alignment mismatch")
    if int(left["person_id"]) != int(right["person_id"]):
        raise ValueError("person alignment mismatch")
    if list(left["frame_id"]) != list(right["frame_id"]):
        raise ValueError("frame alignment mismatch")


def assert_finite(value, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}" if path else str(key))
    elif torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {path}")
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite array at {path}")


@torch.no_grad()
def rebuild_local_vertices(head, parameters: Mapping):
    """Rebuild local MHR geometry; camera placement remains a separate branch."""
    vertices = head.mhr_forward(
        global_trans=torch.zeros_like(parameters["global_rot"]),
        global_rot=parameters["global_rot"],
        body_pose_params=parameters.get("body_pose_params", parameters.get("body_pose")),
        hand_pose_params=parameters.get("hand_pose_params", parameters.get("hand")),
        scale_params=parameters.get("scale_params", parameters.get("scale")),
        shape_params=parameters.get("shape_params", parameters.get("shape")),
        expr_params=parameters.get("expr_params", parameters.get("face")),
        do_pcblend=True,
        return_keypoints=False,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
    )
    vertices = vertices.clone()
    vertices[..., [1, 2]] *= -1
    assert_finite(vertices, "vertices")
    return vertices
