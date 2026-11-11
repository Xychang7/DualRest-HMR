from pathlib import Path
import os
import pickle
import subprocess
import tempfile

import numpy as np


MHR_CLEAN_ROOT = Path("/root/miniconda3/envs/mhr_clean")
MHR_CLEAN_PYTHON = MHR_CLEAN_ROOT / "bin/python"
DEFAULT_WORKER = Path(__file__).with_name("mhr_worker_smplx.py")
DEFAULT_SMPLX_MODEL = Path(
    "/root/sam-body4d-0.2.0/models/SMPLX/SMPLX_NEUTRAL.npz"
)


class MHRtoSMPLXAdapter:
    def __init__(
        self,
        smplx_model_path=DEFAULT_SMPLX_MODEL,
        worker_script=DEFAULT_WORKER,
        worker_python=MHR_CLEAN_PYTHON,
    ):
        self.smplx_model_path = Path(smplx_model_path)
        self.worker_script = Path(worker_script)
        self.worker_python = Path(worker_python)
        for path in (
            self.smplx_model_path,
            self.worker_script,
            self.worker_python,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

    @staticmethod
    def _prepare_vertices(mhr_vertices):
        vertices = np.asarray(mhr_vertices, dtype=np.float32)
        if vertices.ndim != 3 or vertices.shape[1:] != (18439, 3):
            raise ValueError(
                f"mhr_vertices must have shape (N, 18439, 3), got {vertices.shape}"
            )
        if not np.isfinite(vertices).all():
            raise ValueError("mhr_vertices contains NaN or Inf")
        return np.ascontiguousarray(vertices)

    def _worker_environment(self):
        environment = os.environ.copy()
        torch_lib = MHR_CLEAN_ROOT / "lib/python3.12/site-packages/torch/lib"
        env_lib = MHR_CLEAN_ROOT / "lib"
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(torch_lib), str(env_lib), existing]
        ).rstrip(os.pathsep)
        return environment

    def convert(self, mhr_vertices, *, single_identity=True):
        vertices_m = self._prepare_vertices(mhr_vertices)
        # DynaMesh4D vertices are in meters, while Meta MHR conversion expects
        # centimeter-scale input before its internal 0.01 conversion.
        vertices_cm = np.ascontiguousarray(vertices_m * np.float32(100.0))
        payload = {
            "mhr_vertices": vertices_cm,
            "single_identity": bool(single_identity),
        }
        with tempfile.TemporaryDirectory(prefix="mhr2smplx_") as temp_dir:
            input_path = Path(temp_dir) / "input.pkl"
            output_path = Path(temp_dir) / "output.pkl"
            with input_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            command = [
                str(self.worker_python),
                str(self.worker_script),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--smplx-model-path",
                str(self.smplx_model_path),
            ]
            completed = subprocess.run(
                command,
                cwd="/root/MHR-main/tools/mhr_smpl_conversion",
                env=self._worker_environment(),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "SMPL-X worker failed "
                    f"(exit={completed.returncode})\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError("SMPL-X worker did not create an output file")
            with output_path.open("rb") as handle:
                result = pickle.load(handle)

        vertices = np.asarray(result["smplx_vertices"])
        if vertices.shape != (payload["mhr_vertices"].shape[0], 10475, 3):
            raise RuntimeError(f"unexpected SMPL-X shape: {vertices.shape}")
        if vertices.dtype != np.float32 or not np.isfinite(vertices).all():
            raise RuntimeError("SMPL-X vertices must be finite float32")
        return result
