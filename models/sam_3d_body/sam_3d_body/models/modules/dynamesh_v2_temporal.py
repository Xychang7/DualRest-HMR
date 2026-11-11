"""Temporal feature adapters for DynaMesh4D-v2."""

from __future__ import annotations

import torch
from mamba_ssm import Mamba
from torch import nn


class _IdentityFeatureTemporalAdapter(nn.Module):
    feature_dim = 1024

    def forward(self, pose_tokens: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        if pose_tokens.ndim != 2 or pose_tokens.shape[1] != self.feature_dim:
            raise ValueError(
                f"pose_tokens must have shape (T, {self.feature_dim}), got {tuple(pose_tokens.shape)}"
            )
        if visibility.ndim != 1 or visibility.shape[0] != pose_tokens.shape[0]:
            raise ValueError(
                "visibility must have shape (T,) aligned with pose_tokens, "
                f"got {tuple(visibility.shape)} for T={pose_tokens.shape[0]}"
            )
        return pose_tokens


class BodyFeatureTemporalAdapter(nn.Module):
    """Occlusion-conditioned residual Bi-Mamba over complete person sequences."""

    feature_dim = 1024

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.input_projection = nn.Linear(self.feature_dim, d_model)
        self.condition_embedding = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.layers_fwd = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.layers_bwd = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, self.feature_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _validate(self, pose_tokens: torch.Tensor, occlusion_ratio: torch.Tensor):
        unbatched = pose_tokens.ndim == 2
        if unbatched:
            if pose_tokens.shape[-1] != self.feature_dim or occlusion_ratio.ndim != 1:
                raise ValueError("unbatched inputs must have shapes (T,1024) and (T,)")
            pose_tokens = pose_tokens.unsqueeze(0)
            occlusion_ratio = occlusion_ratio.unsqueeze(0)
        elif pose_tokens.ndim != 3 or pose_tokens.shape[-1] != self.feature_dim or occlusion_ratio.ndim != 2:
            raise ValueError("batched inputs must have shapes (B,T,1024) and (B,T)")
        if tuple(occlusion_ratio.shape) != tuple(pose_tokens.shape[:2]):
            raise ValueError("visibility/occlusion_ratio must align exactly with the token B,T dimensions")
        if not torch.isfinite(pose_tokens).all() or not torch.isfinite(occlusion_ratio).all():
            raise ValueError("tokens and occlusion_ratio must be finite")
        if torch.any((occlusion_ratio < 0) | (occlusion_ratio > 1)):
            raise ValueError("occlusion_ratio must be in [0,1]")
        return pose_tokens, occlusion_ratio.to(dtype=pose_tokens.dtype), unbatched

    def forward(self, pose_tokens: torch.Tensor, occlusion_ratio: torch.Tensor) -> torch.Tensor:
        tokens, ratio, unbatched = self._validate(pose_tokens, occlusion_ratio)
        # mamba_ssm 2.3.2 uses CUDA-only fused kernels.  Preserve the Phase-0
        # CPU identity contract without invoking those kernels while the
        # zero-initialized residual head guarantees an exact zero correction.
        if tokens.device.type != "cuda":
            if torch.count_nonzero(self.output_projection.weight) == 0 and torch.count_nonzero(self.output_projection.bias) == 0:
                return pose_tokens
            raise RuntimeError("A trained BodyFeatureTemporalAdapter requires CUDA")
        latent = self.input_projection(self.input_norm(tokens))
        latent = latent + self.condition_embedding(ratio.unsqueeze(-1))
        forward = latent
        for layer in self.layers_fwd:
            forward = layer(forward)
        backward = torch.flip(latent, dims=(1,))
        for layer in self.layers_bwd:
            backward = layer(backward)
        backward = torch.flip(backward, dims=(1,))
        delta = self.output_projection(self.output_norm(forward + backward))
        refined = tokens + delta
        return refined.squeeze(0) if unbatched else refined


class CameraFeatureTemporalAdapter(_IdentityFeatureTemporalAdapter):
    """Phase-1 camera adapter remains an exact identity and has no parameters."""
