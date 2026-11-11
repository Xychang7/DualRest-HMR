import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/sam_3d_body'))
sys.path.insert(0, str(ROOT / 'scripts'))

from mine_dynamesh_v2_camera_failures import (
    is_camera_specific_candidate,
    mine_records,
    validate_mining_contract,
)
from train_dynamesh_v2_camera_phase1 import (
    assert_aligned_camera_fields,
    assert_camera_optimizer_ownership,
    assert_inference_reliability_contract,
    assert_projection_equivalence,
    deltas,
    masked_normalized_reprojection_loss,
    soft_reliability_target,
    teacher_stability_weight,
    weighted_smooth_l1,
)


def record(frame, cam, reproj, body=0.05, bbox=0.01):
    return {
        'sequence': 'train_seq', 'person_id': 0, 'frame_id': frame,
        'condition': 'clean', 'pred_cam': cam,
        'reprojection_error': reproj,
        'body_temporal_innovation': body,
        'visibility': -1,
        'bbox_center_delta': [bbox, bbox],
        'bbox_scale_delta': bbox,
        'mask_area_delta': None,
    }


class MiningContractTests(unittest.TestCase):
    def test_camera_specific_gate_rejects_bbox_driven_candidate(self):
        candidate = {
            'body_temporal_innovation': .05, 'bbox_dynamics': .30,
            'reprojection_spike': .20, 'camera_temporal_innovation': .90,
        }
        self.assertFalse(is_camera_specific_candidate(candidate))
        candidate['bbox_dynamics'] = .10
        self.assertTrue(is_camera_specific_candidate(candidate))

    def test_only_clean_train_is_allowed(self):
        self.assertEqual(validate_mining_contract('train', 'clean'), ('train', 'clean'))
        for split, condition in [('validation', 'clean'), ('test', 'clean'), ('train', 'occ15')]:
            with self.assertRaises(ValueError):
                validate_mining_contract(split, condition)

    def test_normal_abnormal_normal_camera_failure_ranks_first(self):
        records = [
            record(0, [1., 0., 0.], .05),
            record(1, [1., 0., 0.], .05),
            record(2, [1., 2., 0.], .60, body=.01, bbox=.001),
            record(3, [1., 0., 0.], .04),
            record(4, [1., .1, 0.], .06),
        ]
        candidates = mine_records(records, split='train', condition='clean')
        self.assertEqual(candidates[0]['frame_id'], 2)
        self.assertGreater(candidates[0]['reprojection_spike'], 0)
        self.assertNotIn('raw_common_root_error', candidates[0])

    def test_duplicate_and_common_root_dependency_are_rejected(self):
        duplicate = [record(0, [1., 0., 0.], .1), record(0, [1., 0., 0.], .2)]
        with self.assertRaises(ValueError):
            mine_records(duplicate, split='train', condition='clean')
        bad = [record(i, [1., 0., 0.], .1) for i in range(3)]
        bad[1]['raw_common_root_error'] = 1.0
        with self.assertRaises(ValueError):
            mine_records(bad, split='train', condition='clean')


class AlignmentAndEquivalenceTests(unittest.TestCase):
    def test_frame_delta_uses_matching_destination_normalizer(self):
        values = np.array([[0., 0.], [2., 4.], [5., 10.]])
        normalizer = np.array([[1.], [2.], [5.]])
        result = deltas(values, normalizer)
        np.testing.assert_allclose(result, [[0., 0.], [1., 2.], [.6, 1.2]])

    def test_token_init_metadata_alignment(self):
        assert_aligned_camera_fields(
            pose_token=np.zeros((4, 3, 1024), np.float16),
            init_state=np.zeros((4, 3, 519), np.float32),
            frame_id=np.array([3, 4, 5]), person_id=2,
            conditions=('clean', 'occ15', 'occ35', 'occ55'),
        )
        with self.assertRaises(ValueError):
            assert_aligned_camera_fields(
                np.zeros((4, 3, 1024)), np.zeros((1, 3, 519)),
                np.array([3, 4, 5]), 2, ('clean', 'occ15', 'occ35', 'occ55'))

    def test_projection_equivalence_has_explicit_tolerance(self):
        expected = np.array([[10., 20.], [30., 40.]])
        assert_projection_equivalence(expected, expected + 1e-6, atol=1e-5)
        with self.assertRaises(ValueError):
            assert_projection_equivalence(expected, expected + 1e-2, atol=1e-5)


class LossContractTests(unittest.TestCase):
    def test_reliability_target_and_teacher_weight_decrease_with_error(self):
        errors = torch.tensor([0.01, 0.10, 0.50])
        target = soft_reliability_target(errors, scale=0.10)
        weights = teacher_stability_weight(target, torch.tensor([1., 1., 1.]))
        self.assertTrue(torch.all(target[:-1] > target[1:]))
        self.assertTrue(torch.all(weights[:-1] > weights[1:]))
        self.assertTrue(torch.all((target >= 0) & (target <= 1)))

    def test_gt_reprojection_is_forbidden_as_inference_feature(self):
        assert_inference_reliability_contract({'visibility': torch.ones(2, 1)})
        for forbidden in ('gt_2d', 'gt_reprojection_error', 'reprojection_residual'):
            with self.assertRaises(ValueError):
                assert_inference_reliability_contract({forbidden: torch.ones(2, 1)})

    def test_masked_reprojection_and_weighted_teacher_are_finite_differentiable(self):
        pred = torch.tensor([[[10., 10.], [20., 20.]]], requires_grad=True)
        gt = torch.tensor([[[12., 10., 1.], [999., 999., 0.]]])
        reproj = masked_normalized_reprojection_loss(pred, gt, torch.tensor([100.]))
        teacher = weighted_smooth_l1(torch.ones(1, 2, 3, requires_grad=True), torch.zeros(1, 2, 3), torch.tensor([[1., 0.]]))
        total = reproj + teacher
        self.assertTrue(torch.isfinite(total)); total.backward()
        self.assertAlmostEqual(float(reproj), 0.02, places=4)

    def test_optimizer_must_own_only_camera_adapter(self):
        camera = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(camera.parameters())
        assert_camera_optimizer_ownership(optimizer, camera)
        foreign = torch.nn.Parameter(torch.ones(1))
        optimizer.add_param_group({'params': [foreign]})
        with self.assertRaises(AssertionError):
            assert_camera_optimizer_ownership(optimizer, camera)


if __name__ == '__main__':
    unittest.main(verbosity=2)

