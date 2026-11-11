"""Independent reliability-aware Camera temporal adapter (Phase-0)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from mamba_ssm import Mamba
from torch import nn


@dataclass(frozen=True)
class CameraReliabilityFeatures:
    visibility: Optional[torch.Tensor]
    bbox_center_delta: Optional[torch.Tensor]
    bbox_scale_delta: Optional[torch.Tensor]
    mask_area_delta: Optional[torch.Tensor]
    temporal_latent_disagreement: Optional[torch.Tensor]

    # Inference-only cues.  Reprojection is deliberately excluded because the
    # current production pipeline has no independent observed 2D keypoints.
    widths=(1,2,1,1,1)

    def unbatched(self):
        fields={name:(None if value is None else value.squeeze(0)) for name,value in self.__dict__.items()}
        return replace(self,**fields)

    def packed(self):
        present=[x for x in self.__dict__.values() if x is not None]
        if not present: raise ValueError("at least one reliability feature must be present")
        reference=present[0]; unbatched=reference.ndim==2
        expected_prefix=reference.shape[:-1]
        values=[]; masks=[]
        for (name,value),width in zip(self.__dict__.items(),self.widths):
            if value is None:
                values.append(torch.zeros(*expected_prefix,width,device=reference.device,dtype=reference.dtype)); masks.append(torch.zeros_like(values[-1]))
            else:
                if value.shape[:-1]!=expected_prefix or value.shape[-1]!=width: raise ValueError(f"{name} shape/alignment mismatch")
                if not torch.isfinite(value).all(): raise ValueError(f"{name} contains NaN/Inf")
                values.append(value.to(dtype=reference.dtype)); masks.append(torch.ones_like(value,dtype=reference.dtype))
        return torch.cat(values,-1),torch.cat(masks,-1)


def apply_reliability_gate(
    raw_token: torch.Tensor,
    delta_token: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Apply the locked reliability convention.

    reliability=1 means raw SAM camera is reliable and suppresses correction;
    reliability=0 means it is unreliable and permits the full correction.
    """
    if raw_token.shape != delta_token.shape:
        raise ValueError("raw_token and delta_token must have identical shape")
    if reliability.shape != raw_token.shape[:-1] + (1,):
        raise ValueError("reliability must have shape (..., 1) aligned to tokens")
    if not torch.all((reliability >= 0) & (reliability <= 1)):
        raise ValueError("reliability must be in [0, 1]")
    return raw_token + (1.0 - reliability) * delta_token


class CameraFeatureTemporalAdapter(nn.Module):
    feature_dim=1024
    reliability_dim=6

    def __init__(self,d_model=128,n_layers=2,d_state=16,d_conv=4,expand=2):
        super().__init__(); self.input_norm=nn.LayerNorm(self.feature_dim); self.input_projection=nn.Linear(self.feature_dim,d_model); self.condition_embedding=nn.Linear(self.reliability_dim*2,d_model)
        self.layers_fwd=nn.ModuleList([Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand) for _ in range(n_layers)])
        self.layers_bwd=nn.ModuleList([Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand) for _ in range(n_layers)])
        self.output_norm=nn.LayerNorm(d_model); self.output_projection=nn.Linear(d_model,self.feature_dim); self.reliability_head=nn.Sequential(nn.Linear(self.reliability_dim*2,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.output_projection.weight); nn.init.zeros_(self.output_projection.bias)

    def forward(self,raw_pose_token:torch.Tensor,features:CameraReliabilityFeatures):
        unbatched=raw_pose_token.ndim==2
        if unbatched: tokens=raw_pose_token.unsqueeze(0)
        elif raw_pose_token.ndim==3: tokens=raw_pose_token
        else: raise ValueError("raw_pose_token must be (T,1024) or (B,T,1024)")
        if tokens.shape[-1]!=self.feature_dim or not torch.isfinite(tokens).all(): raise ValueError("invalid raw_pose_token")
        values,mask=features.packed()
        if unbatched: values=values.unsqueeze(0); mask=mask.unsqueeze(0)
        if values.shape[:2]!=tokens.shape[:2]: raise ValueError("token/reliability frame alignment mismatch")
        condition=torch.cat((values.to(tokens),mask.to(tokens)),-1); reliability=torch.sigmoid(self.reliability_head(condition))
        # Phase-0 CPU path avoids CUDA-only fused Mamba kernels while proving exact identity.
        if tokens.device.type!="cuda" and torch.count_nonzero(self.output_projection.weight)==0 and torch.count_nonzero(self.output_projection.bias)==0:
            refined=tokens
        else:
            latent=self.input_projection(self.input_norm(tokens))+self.condition_embedding(condition); forward=latent
            for layer in self.layers_fwd: forward=layer(forward)
            backward=torch.flip(latent,(1,))
            for layer in self.layers_bwd: backward=layer(backward)
            backward=torch.flip(backward,(1,)); delta=self.output_projection(self.output_norm(forward+backward)); refined=apply_reliability_gate(tokens,delta,reliability)
        return (refined.squeeze(0),reliability.squeeze(0)) if unbatched else (refined,reliability)
