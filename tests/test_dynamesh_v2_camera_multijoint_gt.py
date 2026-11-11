import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path('/root/sam-body4d-0.2.0/scripts/validate_dynamesh_v2_camera_multijoint_gt.py')
spec = importlib.util.spec_from_file_location('phase18', SCRIPT)
phase18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase18)


class Phase18Tests(unittest.TestCase):
    def test_strict_3d_mapping_is_semantically_exact_and_excludes_approximate_roots(self):
        phase18.validate_joint_mapping()
        names = [row['name'] for row in phase18.JOINT_MAPPING]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 12)
        self.assertFalse({'pelvis', 'neck', 'head', 'hip_midpoint'}.intersection(names))
        self.assertEqual([row['smpl_native_index'] for row in phase18.JOINT_MAPPING], [16,17,18,19,20,21,1,2,4,5,7,8])

    def test_weighted_translation_recovers_exact_target_and_respects_mask(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 3)); truth = np.array([0.2, -0.4, 3.5]); y = x + truth
        weights = np.ones(12); weights[3] = 0; y[3] = 999
        result = phase18.translation_only_fit(x, y, weights)
        np.testing.assert_allclose(result['translation'], truth, atol=1e-12)
        self.assertEqual(result['valid_joint_count'], 11)
        self.assertLess(result['mean_residual'], 1e-12)

    def test_regional_fit_exposes_pose_dependent_mismatch(self):
        x = np.zeros((12, 3)); y = np.zeros((12, 3))
        y[:6] += np.array([0, 0, 2.0]); y[6:] += np.array([0, 0, 3.0])
        result = phase18.fit_joint_subsets(x, y, np.ones(12))
        self.assertAlmostEqual(result['upper_lower_disagreement'], 1.0)
        np.testing.assert_allclose(result['all']['translation'], [0, 0, 2.5])

    def test_translation_only_does_not_hide_rotation_or_scale_error(self):
        x = np.array([[1,0,0], [0,1,0], [-1,0,0], [0,-1,0]], dtype=float)
        y = 2.0 * x[:, [1,0,2]] + np.array([0.1, 0.2, 4.0])
        self.assertGreater(phase18.translation_only_fit(x, y, np.ones(4))['mean_residual'], 0.5)

    def test_invalid_or_empty_confidence_is_rejected(self):
        x = np.zeros((12, 3))
        with self.assertRaisesRegex(ValueError, 'valid joint'):
            phase18.translation_only_fit(x, x, np.zeros(12))
        with self.assertRaisesRegex(ValueError, 'shape'):
            phase18.translation_only_fit(x[:3], x[:3], np.ones(12))

    def test_missing_regional_support_does_not_invent_correspondence(self):
        x = np.zeros((12,3)); y = x + np.array([1,2,3]); weights = np.ones(12); weights[:6] = 0
        result = phase18.fit_joint_subsets(x,y,weights)
        self.assertEqual(result['upper']['valid_joint_count'], 0)
        self.assertTrue(np.isnan(result['upper_lower_disagreement']))
        np.testing.assert_allclose(result['all']['translation'], [1,2,3])

    def test_scalar_diagnostics_are_json_serializable(self):
        result = phase18.translation_only_fit(np.zeros((12,3)), np.ones((12,3)), np.ones(12))
        json.dumps({'valid_joint_count': result['valid_joint_count'], 'mean_residual': result['mean_residual']}, allow_nan=False)

    def test_reliability_verdict_accepts_broad_failure_but_not_unconditional_target(self):
        clean={'fit_residual':{'p95':.06},'upper_lower_disagreement':{'p95':.07,'median':.035},'correlations':{'fit_vs_pose_magnitude':.2}}
        failure={520:{'camera_error':5.4,'target_tx':0,'target_ty':0,'target_tz':3.7},522:{'camera_error':5.8,'target_tx':0,'target_ty':0,'target_tz':3.75},524:{'camera_error':1.3,'target_tx':0,'target_ty':0,'target_tz':3.78}}
        verdict, isolated, broad, _ = phase18.choose_verdict(clean,np.full(100,.04),failure,.7)
        self.assertFalse(isolated); self.assertTrue(broad)
        self.assertEqual(verdict,'MULTI_JOINT_CAMERA_GT_USABLE_WITH_RELIABILITY')


if __name__ == '__main__':
    unittest.main()
