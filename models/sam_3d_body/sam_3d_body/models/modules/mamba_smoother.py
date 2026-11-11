from functools import lru_cache

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMambaSmoother(nn.Module):
    def __init__(
        self,
        input_dim=266,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers_fwd = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.layers_bwd = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        residual = x
        x_embed = self.embedding(x)

        x_fwd = x_embed
        for layer in self.layers_fwd:
            x_fwd = layer(x_fwd)

        x_bwd = torch.flip(x_embed, dims=[1])
        for layer in self.layers_bwd:
            x_bwd = layer(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])

        correction = self.output_head(self.norm(x_fwd + x_bwd))
        return residual + correction


@lru_cache(maxsize=None)
def load_bimamba_smoother(checkpoint_path, device):
    model = BiMambaSmoother(
        input_dim=266,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "BiMamba checkpoint mismatch: "
            f"missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )
    model.eval()
    return model
