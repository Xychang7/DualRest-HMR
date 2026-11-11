from pathlib import Path
import argparse
import pickle
import sys
import time

import numpy as np


MHR_ROOT = "/root/MHR-main"
MHR_TOOLS = "/root/MHR-main/tools/mhr_smpl_conversion"
sys.path.insert(0, MHR_ROOT)
sys.path.insert(0, MHR_TOOLS)

import smplx
import torch
from mhr.mhr import MHR
from conversion import Conversion


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(item) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smplx-model-path", required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        payload = pickle.load(handle)
    mhr_vertices = np.asarray(payload["mhr_vertices"], dtype=np.float32)
    if mhr_vertices.ndim != 3 or mhr_vertices.shape[1:] != (18439, 3):
        raise ValueError(f"expected (N, 18439, 3), got {mhr_vertices.shape}")

    started = time.perf_counter()
    device = torch.device("cpu")
    mhr_model = MHR.from_files(lod=1, device=device)
    smplx_model = smplx.SMPLX(model_path=args.smplx_model_path).to(device)
    if tuple(smplx_model.v_template.shape) != (10475, 3):
        raise RuntimeError(
            f"target is not standard SMPL-X topology: {smplx_model.v_template.shape}"
        )
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch",
    )
    converted = converter.convert_mhr2smpl(
        mhr_vertices=mhr_vertices,
        single_identity=bool(payload.get("single_identity", True)),
        return_smpl_vertices=True,
        return_smpl_parameters=True,
        return_fitting_errors=True,
    )
    smplx_vertices = np.asarray(_to_numpy(converted.result_vertices), dtype=np.float32)
    if smplx_vertices.shape != (mhr_vertices.shape[0], 10475, 3):
        raise RuntimeError(f"unexpected SMPL-X output shape: {smplx_vertices.shape}")
    if not np.isfinite(smplx_vertices).all():
        raise RuntimeError("SMPL-X output contains NaN or Inf")
    result = {
        "smplx_vertices": np.ascontiguousarray(smplx_vertices),
        "smplx_parameters": _to_numpy(converted.result_parameters),
        "fitting_errors": _to_numpy(converted.result_errors),
        "elapsed_seconds": time.perf_counter() - started,
        "topology": "SMPL-X 10475 standard vertex ordering",
    }
    with open(args.output, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
