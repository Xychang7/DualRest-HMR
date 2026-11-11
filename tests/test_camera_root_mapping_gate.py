import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT / 'scripts'))

from validate_dynamesh_v2_camera_root_mapping import evaluate_root_mapping


class RootMappingNumericalGateTests(unittest.TestCase):
    def test_reports_physical_root_error_and_mapping_variation_without_magic_scale(self):
        mhr_root = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=np.float32)
        pred_cam_t = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        regressor = np.array([[1.0, 0.0]], dtype=np.float32)
        converted = np.array([
            [[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]],
            [[0.21, 0.2, 0.3], [9.0, 9.0, 9.0]],
        ], dtype=np.float32)
        gt_world = np.array([
            [[1.1, 2.2, 3.3], [0.0, 0.0, 0.0]],
            [[1.2, 2.2, 3.3], [0.0, 0.0, 0.0]],
        ], dtype=np.float32)
        camera = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        report = evaluate_root_mapping(mhr_root, pred_cam_t, converted, gt_world, regressor, 0, camera)
        np.testing.assert_allclose(report['predicted_camera_root'], gt_world[:, 0], atol=1e-6)
        self.assertAlmostEqual(report['error']['max'], 0.0, places=6)
        self.assertIn('systematic_offset', report)
        self.assertGreater(report['local_mapping']['std_norm'], 0.0)

    def test_rejects_non_metre_units(self):
        with self.assertRaises(ValueError):
            evaluate_root_mapping(
                np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 2, 3)),
                np.zeros((1, 2, 3)), np.array([[1.0, 0.0]]), 0,
                np.eye(4)[None], unit='centimeters')


if __name__ == '__main__':
    unittest.main(verbosity=2)
