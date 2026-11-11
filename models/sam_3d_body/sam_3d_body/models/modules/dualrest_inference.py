"""Inference-only composition helpers for the released DualRest-HMR adapters.

The Frame-wise and temporal camera computations are an exact extraction of the
formal Phase-1.11/Phase-1.12 inference path.  Keeping them here avoids importing
training scripts (and their dataset/server path constants) at inference time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .camera_temporal import CameraReliabilityFeatures


class FramewiseLatentAdapter(nn.Module):
    """Official Phase-1.11 frame-wise camera latent calibrator."""

    feature_dim = 1024

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(self.feature_dim)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, self.feature_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pose_token: torch.Tensor) -> torch.Tensor:
        if pose_token.ndim not in (2, 3) or pose_token.shape[-1] != self.feature_dim:
            raise ValueError("pose_token must have shape (T,1024) or (B,T,1024)")
        if not torch.isfinite(pose_token).all():
            raise ValueError("pose_token contains NaN/Inf")
        return pose_token + self.fc2(F.gelu(self.fc1(self.norm(pose_token))))


def camera_reliability_features(
    tokens: torch.Tensor, meta: Mapping, occlusion_ratio: float
) -> CameraReliabilityFeatures:
    """Build the locked inference-only reliability cues used by Phase-1.12."""

    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError("tokens must have shape (T,1024) or (B,T,1024)")
    if not 0.0 <= float(occlusion_ratio) <= 1.0:
        raise ValueError("occlusion_ratio must be in [0,1]")
    tokens = tokens.float()
    disagreement = torch.zeros(
        tokens.shape[:2] + (1,), device=tokens.device, dtype=tokens.dtype
    )
    if tokens.shape[1] > 2:
        disagreement[:, 1:-1, 0] = torch.linalg.vector_norm(
            tokens[:, 1:-1] - 0.5 * (tokens[:, :-2] + tokens[:, 2:]), dim=-1
        ) / math.sqrt(tokens.shape[-1])
    scale = torch.quantile(disagreement.detach().flatten(), 0.75).clamp_min(1e-6)
    disagreement = (disagreement / scale).clamp(0, 4)

    center = torch.as_tensor(
        meta["center"], device=tokens.device, dtype=tokens.dtype
    )
    size = torch.as_tensor(meta["size"], device=tokens.device, dtype=tokens.dtype)
    if center.shape != (tokens.shape[1], 2) or size.shape != (tokens.shape[1],):
        raise ValueError("center/size metadata is not frame-aligned")
    center_delta = torch.zeros(
        (len(size), 2), device=tokens.device, dtype=tokens.dtype
    )
    center_delta[1:] = (center[1:] - center[:-1]) / size[1:, None].clamp_min(1e-6)
    scale_delta = torch.zeros(
        (len(size), 1), device=tokens.device, dtype=tokens.dtype
    )
    scale_delta[1:, 0] = torch.log(
        size[1:].clamp_min(1e-6) / size[:-1].clamp_min(1e-6)
    )
    visibility = torch.full(
        (tokens.shape[0], len(size), 1),
        1.0 - float(occlusion_ratio),
        device=tokens.device,
        dtype=tokens.dtype,
    )
    return CameraReliabilityFeatures(
        visibility=visibility,
        bbox_center_delta=center_delta.unsqueeze(0).expand(tokens.shape[0], -1, -1),
        bbox_scale_delta=scale_delta.unsqueeze(0).expand(tokens.shape[0], -1, -1),
        mask_area_delta=None,
        temporal_latent_disagreement=disagreement,
    )


def stable_output_layer_norm(adapter: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Evaluate LayerNorm statistics in float64 as in the formal Phase-1.12 path."""

    norm = adapter.output_norm
    weight = norm.weight.double() if norm.weight is not None else None
    bias = norm.bias.double() if norm.bias is not None else None
    return F.layer_norm(
        value.double(), norm.normalized_shape, weight, bias, norm.eps
    ).to(value.dtype)


def stable_mamba_stack(layers: nn.ModuleList, value: torch.Tensor) -> torch.Tensor:
    """Apply the locked per-layer normalization from Camera Phase-1.12."""

    for layer in layers:
        value = layer(value)
        if not torch.isfinite(value).all():
            raise FloatingPointError("camera temporal Mamba activation is non-finite")
        value = F.layer_norm(value.double(), (value.shape[-1],), eps=1e-5).to(
            value.dtype
        )
    return value


def camera_temporal_residual_forward(
    adapter: nn.Module,
    framewise_token: torch.Tensor,
    features: CameraReliabilityFeatures,
) -> torch.Tensor:
    """Official Phase-1.12 temporal residual inference (no training targets)."""

    tokens = (
        framewise_token
        if framewise_token.ndim == 3
        else framewise_token.unsqueeze(0)
    )
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError("framewise_token must be (T,1024) or (B,T,1024)")
    values, mask = features.packed()
    if values.ndim == 2:
        values, mask = values.unsqueeze(0), mask.unsqueeze(0)
    if values.shape[:2] != tokens.shape[:2]:
        raise ValueError("camera reliability features are not token-aligned")
    condition = torch.cat((values.to(tokens), mask.to(tokens)), dim=-1)
    latent = adapter.input_projection(adapter.input_norm(tokens))
    latent = latent + adapter.condition_embedding(condition)
    forward = stable_mamba_stack(adapter.layers_fwd, latent)
    backward = torch.flip(latent, (1,))
    backward = stable_mamba_stack(adapter.layers_bwd, backward)
    backward = torch.flip(backward, (1,))
    delta = adapter.output_projection(
        stable_output_layer_norm(adapter, forward + backward)
    )
    refined = tokens + delta
    if not torch.isfinite(refined).all():
        raise FloatingPointError("camera temporal output is non-finite")
    return refined


def restore_camera_tokens(
    framewise_adapter: FramewiseLatentAdapter,
    temporal_adapter: nn.Module,
    raw_pose_token: torch.Tensor,
    meta: Mapping,
    occlusion_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen Frame-wise then temporal Camera Restoration stages."""

    tokens = raw_pose_token if raw_pose_token.ndim == 3 else raw_pose_token.unsqueeze(0)
    framewise = framewise_adapter(tokens)
    features = camera_reliability_features(framewise, meta, occlusion_ratio)
    temporal = camera_temporal_residual_forward(
        temporal_adapter, framewise, features
    )
    return framewise, temporal
