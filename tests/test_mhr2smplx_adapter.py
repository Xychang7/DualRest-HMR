from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

import numpy as np


ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT))


class MHRtoSMPLXAdapterTests(unittest.TestCase):
    def test_subprocess_round_trip_uses_numpy_payload(self):
        from adapters.mhr2smplx_adapter import MHRtoSMPLXAdapter

        with tempfile.TemporaryDirectory() as temp_dir:
            worker = Path(temp_dir) / "fake_worker.py"
            worker.write_text(textwrap.dedent("""
                import argparse
                import pickle
                import numpy as np

                parser = argparse.ArgumentParser()
                parser.add_argument("--input", required=True)
                parser.add_argument("--output", required=True)
                parser.add_argument("--smplx-model-path", required=True)
                args = parser.parse_args()
                with open(args.input, "rb") as handle:
                    payload = pickle.load(handle)
                vertices = payload["mhr_vertices"]
                assert isinstance(vertices, np.ndarray)
                assert vertices.dtype == np.float32
                assert np.all(vertices == 100.0), "worker input must be centimeters"
                result = {
                    "smplx_vertices": np.zeros((vertices.shape[0], 10475, 3), dtype=np.float32),
                    "fitting_errors": np.full((vertices.shape[0],), 0.125, dtype=np.float32),
                    "elapsed_seconds": 0.0,
                }
                with open(args.output, "wb") as handle:
                    pickle.dump(result, handle)
            """))
            adapter = MHRtoSMPLXAdapter(
                worker_script=worker,
                smplx_model_path=ROOT / "models/SMPLX/SMPLX_NEUTRAL.npz",
            )
            result = adapter.convert(np.ones((1, 18439, 3), dtype=np.float64))

        self.assertEqual(result["smplx_vertices"].shape, (1, 10475, 3))
        self.assertEqual(result["smplx_vertices"].dtype, np.float32)
        self.assertEqual(result["fitting_errors"].tolist(), [0.125])

    def test_rejects_non_mhr_vertex_shape_before_subprocess(self):
        from adapters.mhr2smplx_adapter import MHRtoSMPLXAdapter

        adapter = MHRtoSMPLXAdapter(
            smplx_model_path=ROOT / "models/SMPLX/SMPLX_NEUTRAL.npz"
        )
        with self.assertRaisesRegex(ValueError, r"\(N, 18439, 3\)"):
            adapter.convert(np.zeros((1, 10475, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
