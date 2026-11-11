import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.dynamesh_v2_unified_eval_core import (
    H36M_TO_J14,
    acceleration_error,
    atomic_json,
    build_temporal_coverage,
    canonical_hash,
    compute_geometry_metrics,
    validate_resume_state,
)


class UnifiedEvalTest(unittest.TestCase):
    def test_overlap_coverage_keeps_tail_and_is_deterministic(self):
        a = build_temporal_coverage(np.arange(19), window_length=16, stride=8)
        b = build_temporal_coverage(np.arange(19), window_length=16, stride=8)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a["owner"].keys()), list(range(19)))
        self.assertEqual(a["windows"], [list(range(16)), list(range(3, 19))])
        self.assertEqual(a["owner"][18], [1, 15])


    def test_short_segment_uses_edge_padding_without_crossing_gap(self):
        coverage = build_temporal_coverage(np.array([0, 1, 2, 8, 9]), 4, 2)
        self.assertEqual(sorted(coverage["owner"]), [0, 1, 2, 3, 4])
        self.assertTrue(all(set(w).issubset({0, 1, 2}) or set(w).issubset({3, 4}) for w in coverage["windows"]))


    def test_acceleration_uses_only_true_contiguous_triplets(self):
        joints = np.zeros((6, 14, 3), np.float64)
        joints[:, :, 0] = np.array([0, 1, 4, 10, 20, 35])[:, None]
        result = acceleration_error(joints, np.zeros_like(joints), np.array([0, 1, 2, 7, 8, 9]))
        self.assertEqual(result["triplet_centers"].tolist(), [1, 4])
        self.assertTrue(np.allclose(result["per_triplet"], [2000.0, 5000.0]))


    def test_geometry_metrics_pelvis_aligns_mpvpe_and_mpjpe(self):
        rng = np.random.default_rng(7)
        gt = rng.normal(size=(2, 6890, 3))
        pred = gt + np.array([3.0, -2.0, 8.0])
        reg = np.zeros((17, 6890)); reg[:, :17] = np.eye(17)
        out = compute_geometry_metrics(pred, gt, reg)
        self.assertLess(max(out["mpjpe_mm"]), 1e-8)
        self.assertLess(max(out["pa_mpjpe_mm"]), 1e-8)
        self.assertLess(max(out["mpvpe_mm"]), 1e-8)
        self.assertEqual(len(H36M_TO_J14), 14)


    def test_atomic_json_and_hash_resume_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit.json"
            payload = {"schema": "unit-v1", "completion": True, "protocol_hash": "p", "metric": 1.0}
            payload["checksum_sha256"] = canonical_hash(payload)
            atomic_json(path, payload)
            self.assertFalse((Path(directory) / "unit.json.tmp").exists())
            validate_resume_state(path, protocol_hash="p", expected_schema="unit-v1")
            bad = json.loads(path.read_text()); bad["protocol_hash"] = "wrong"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "protocol hash"):
                validate_resume_state(path, protocol_hash="p", expected_schema="unit-v1")


if __name__ == "__main__":
    unittest.main()
