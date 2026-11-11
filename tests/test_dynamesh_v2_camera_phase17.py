import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path[:0] = [str(ROOT), str(ROOT/'models/sam_3d_body'), str(ROOT/'scripts')]

from audit_dynamesh_v2_camera_phase17 import (
    first_difference,
    relative_relation_statistics,
    route_verdict,
    temporal_capacity_verdict,
)


class Phase17Contracts(unittest.TestCase):
    def test_first_difference_preserves_alignment(self):
        values = np.array([[0., 0., 0.], [1., 2., 3.], [3., 5., 9.]])
        np.testing.assert_allclose(first_difference(values), [[1., 2., 3.], [2., 3., 6.]])

    def test_relative_statistics_cancel_only_constant_offset(self):
        target = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
        source = target + np.array([0., .08, 0.])
        report = relative_relation_statistics(source, target)
        self.assertGreater(report['absolute']['mean'], .079)
        self.assertLess(report['first_difference']['mean'], 1e-12)
        self.assertAlmostEqual(report['first_difference']['direction_cosine_mean'], 1., places=7)

    def test_temporal_and_route_verdicts(self):
        self.assertEqual(temporal_capacity_verdict(.30, .20), 'TEMPORAL_MAMBA_CAPACITY_CONFIRMED')
        self.assertEqual(temporal_capacity_verdict(.05, .01), 'TEMPORAL_MAPPING_CAPACITY_LIMITED')
        self.assertEqual(route_verdict('TEMPORAL_MAMBA_CAPACITY_CONFIRMED', 'RELATIVE_CAMERA_SUPERVISION_USABLE_AS_AUXILIARY'),
                         'ROUTE_KEEP_LATENT_MAMBA_UPDATE_SUPERVISION')
        self.assertEqual(route_verdict('TEMPORAL_MAPPING_CAPACITY_LIMITED', 'RELATIVE_CAMERA_SUPERVISION_NOT_RELIABLE'),
                         'ROUTE_REDESIGN_CAMERA_MAPPING')
        self.assertEqual(route_verdict('TEMPORAL_MAPPING_CAPACITY_LIMITED', 'RELATIVE_CAMERA_SUPERVISION_NOT_RELIABLE',
                                       camera_observable_supervision=False), 'ROUTE_DROP_CAMERA_BRANCH')


if __name__ == '__main__':
    unittest.main(verbosity=2)
