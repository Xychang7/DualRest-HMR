import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path('/root/sam-body4d-0.2.0')
SCRIPT = ROOT / 'scripts/validate_dynamesh_v2_camera_reprojection_gate.py'


def load_module():
    spec = importlib.util.spec_from_file_location('camera_reprojection_gate', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CameraReprojectionGateTests(unittest.TestCase):
    def test_strict_correspondence_has_unique_indices_and_exact_limb_semantics(self):
        m = load_module()
        m.validate_strict_correspondence(m.STRICT_COMMON_JOINTS)
        names = [row['common_joint'] for row in m.STRICT_COMMON_JOINTS]
        self.assertEqual(names, [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        ])
        self.assertEqual(len({row['mhr70_index'] for row in m.STRICT_COMMON_JOINTS}), 12)
        self.assertEqual(len({row['openpose18_index'] for row in m.STRICT_COMMON_JOINTS}), 12)

    def test_confidence_mask_rejects_zero_and_nonfinite_observations(self):
        m = load_module()
        gt = np.array([[10, 20, 1], [30, 40, 0], [np.nan, 50, 1]], dtype=float)
        self.assertEqual(m.confidence_mask(gt).tolist(), [True, False, False])

    def test_projection_shape_finite_and_full_image_pixel_coordinates(self):
        m = load_module()
        points = np.array([[[1.0, 2.0, 4.0], [0.0, 0.0, 2.0]]])
        cam_t = np.zeros((1, 3))
        K = np.array([[[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]])
        out = m.project_mhr_to_pixels(points, cam_t, K)
        self.assertEqual(out.shape, (1, 2, 2))
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out[0], [[75.0, 110.0], [50.0, 60.0]])

    def test_projection_rejects_nonpositive_depth(self):
        m = load_module()
        with self.assertRaisesRegex(ValueError, 'positive camera depth'):
            m.project_mhr_to_pixels(np.zeros((1, 1, 3)), np.zeros((1, 3)), np.eye(3)[None])

    def test_normalized_error_is_pixel_error_over_bbox_width(self):
        m = load_module()
        np.testing.assert_allclose(m.normalized_reprojection_error(np.array([10.0, 20.0]), 200.0), [0.05, 0.1])

    def test_gt_2d_is_training_only_and_inference_contract_rejects_it(self):
        m = load_module()
        gt = np.ones((12, 3))
        m.assert_reprojection_contract('training', gt)
        with self.assertRaisesRegex(ValueError, 'forbids GT 2D'):
            m.assert_reprojection_contract('inference', gt)
        m.assert_reprojection_contract('inference', None)

    def test_frame_metrics_apply_confidence_mask_and_are_finite(self):
        m = load_module()
        pred = np.array([[0, 0], [3, 4], [20, 20]], dtype=float)
        gt = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        result = m.compute_frame_metrics(pred, gt, bbox_size=100.0, mode='training')
        self.assertEqual(result['valid_joint_count'], 2)
        self.assertEqual(result['mean_pixel_error'], 2.5)
        self.assertEqual(result['median_pixel_error'], 2.5)
        self.assertEqual(result['max_pixel_error'], 5.0)
        self.assertEqual(result['mean_normalized_error'], 0.025)

    def test_background_body_evaluator_files_match_pre_gate_hashes(self):
        expected = {
            ROOT / 'scripts/evaluate_dynamesh_v2_phase2.py': '4060bdff357d820a338284ed07e1d597a0138e2aa9ad0fd24c7262dbd6866432',
            ROOT / 'scripts/evaluate_dynamesh_v2_phase2_resume.py': '253fecf8c541c9f2d2258c25b7ff4e08d078ee9f9d72dc1a7bb7f66d0d7015a0',
            ROOT / 'adapters/mhr_to_smpl_eval_worker.py': '29b4d82d12dc6ef55fc08eb94c148cf4f84d13ae66ea09c17b78629ff67d27a2',
        }
        for path, digest in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
