import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT / 'scripts'))

from run_dynamesh_v2_camera_root_gate_50f import relation_statistics


class CameraRootGateStatisticsTests(unittest.TestCase):
    def test_relation_statistics_preserve_signed_offset_and_percentiles(self):
        source = np.zeros((3, 3), dtype=np.float64)
        target = np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        stats = relation_statistics(source, target)
        self.assertAlmostEqual(stats['mean'], 2.0)
        self.assertAlmostEqual(stats['median'], 2.0)
        self.assertAlmostEqual(stats['max'], 3.0)
        np.testing.assert_allclose(stats['xyz_mean_signed_offset'], [-2., 0., 0.])
        np.testing.assert_allclose(stats['xyz_std'], [np.std([1., 2., 3.]), 0., 0.])

    def test_relation_statistics_reject_misalignment_and_nonfinite(self):
        with self.assertRaises(ValueError):
            relation_statistics(np.zeros((2, 3)), np.zeros((3, 3)))
        bad = np.zeros((2, 3)); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            relation_statistics(bad, np.zeros((2, 3)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
