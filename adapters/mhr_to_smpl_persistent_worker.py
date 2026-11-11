"""Official Meta MHR->SMPL worker that preserves fitted SMPL parameters."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["/root/MHR-main", "/root/MHR-main/tools/mhr_smpl_conversion"]
import smplx
from conversion import Conversion
from mhr.mhr import MHR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smpl-model", required=True)
    args = parser.parse_args()
    vertices = np.load(args.input, mmap_mode="r")
    if vertices.ndim != 3 or vertices.shape[1:] != (18439, 3) or not np.isfinite(vertices).all():
        raise ValueError(f"invalid MHR vertices {vertices.shape}")
    converter = Conversion(
        mhr_model=MHR.from_files(lod=1, device=torch.device("cpu")),
        smpl_model=smplx.SMPL(model_path=args.smpl_model).to("cpu"), method="pytorch"
    )
    result = converter.convert_mhr2smpl(
        mhr_vertices=np.ascontiguousarray(vertices, dtype=np.float32) * np.float32(100.0),
        single_identity=True, return_smpl_vertices=True,
        return_smpl_parameters=True, return_fitting_errors=True,
    )
    output = {
        "vertices": np.asarray(result.result_vertices, dtype=np.float32),
        "errors": np.asarray(result.result_errors, dtype=np.float32).reshape(-1),
    }
    for key, value in result.result_parameters.items():
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        output[f"param__{key}"] = np.asarray(array, dtype=np.float32)
    if output["vertices"].shape != (len(vertices), 6890, 3): raise ValueError("invalid SMPL topology")
    if any(not np.isfinite(value).all() for value in output.values()): raise ValueError("non-finite converter output")
    path = Path(args.output); temp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temp, **output)
    with temp.open("rb") as handle: os.fsync(handle.fileno())
    os.replace(temp, path)


if __name__ == "__main__": main()
