import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from train_dynamesh_v2_camera_phase110 import (
    assert_disjoint_tracks,
    assert_inference_contract,
    balanced_schedule,
    camera_need,
    category_anchor,
    classify_quality,
    derive_quality_thresholds,
    reliability_target,
    locked_gradient_probe,
    training_checkpoint,
    stable_output_layer_norm,
    stable_mamba_stack,
)


def rows():
    out = []
    for i in range(100):
        out.append({
            'split': 'train', 'condition': 'clean',
            'sequence': f's{i // 20}', 'person_id': i % 2, 'frame_id': i,
            'camera_error': float(i + 1),
            'fit_residual': 0.01 + i * 0.0001,
            'upper_lower_disagreement': 0.02 + i * 0.0001,
            'reprojection_error': float(i + 1) / 100.0,
            'body_teacher_geodesic_deg': 0.0,
        })
    return out


class Phase110ContractTests(unittest.TestCase):
    def test_thresholds_are_train_only_and_deterministic(self):
        a = derive_quality_thresholds(rows())
        b = derive_quality_thresholds(list(reversed(rows())))
        self.assertEqual(a, b)
        bad = rows() + [{**rows()[0], 'split': 'validation'}]
        with self.assertRaises(ValueError):
            derive_quality_thresholds(bad)

    def test_quality_classification_is_objective(self):
        t = derive_quality_thresholds(rows())
        self.assertEqual(classify_quality(rows()[0], t), 'stable')
        self.assertEqual(classify_quality(rows()[49], t), 'medium')
        failure = {
            **rows()[0], 'camera_error': t['camera_p75'],
            'reprojection_error': t['reproj_p75'],
            'fit_residual': t['fit_p75'],
            'upper_lower_disagreement': t['ul_p75'],
            'body_teacher_geodesic_deg': t['body_p75'],
        }
        self.assertEqual(classify_quality(failure, t), 'failure')

    def test_track_split_is_atomic(self):
        assert_disjoint_tracks(
            [{'sequence': 'a', 'person_id': 0}],
            [{'sequence': 'b', 'person_id': 0}],
        )
        with self.assertRaises(ValueError):
            assert_disjoint_tracks(
                [{'sequence': 'a', 'person_id': 0}],
                [{'sequence': 'a', 'person_id': 0}],
            )

    def test_need_and_reliability_have_opposite_direction(self):
        errors = np.array([1.0, 5.0, 10.0])
        need = camera_need(errors, low=2.0, high=8.0)
        rel = reliability_target(need)
        np.testing.assert_allclose(need, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(rel, [1.0, 0.5, 0.0])

    def test_target_reliability_is_not_camera_need(self):
        need = camera_need(np.array([1.0, 9.0]), low=2.0, high=8.0)
        w_target = np.array([0.2, 0.9])
        self.assertFalse(np.allclose(w_target, need))
        np.testing.assert_allclose(reliability_target(need), [1.0, 0.0])

    def test_anchor_strength_stable_medium_failure(self):
        w = np.ones(3)
        need = np.array([0.0, 0.5, 1.0])
        anchor = category_anchor(np.array(['stable', 'medium', 'failure']), w, need)
        self.assertGreater(anchor[0], anchor[1])
        self.assertGreater(anchor[1], anchor[2])
        self.assertEqual(anchor[2], 0.0)

    def test_balanced_schedule(self):
        schedule = balanced_schedule(12, ('stable', 'medium', 'failure'), 4, seed=7)
        cats = [x[0] for x in schedule]
        conds = [x[1] for x in schedule]
        self.assertEqual({c: cats.count(c) for c in set(cats)}, {'stable': 4, 'medium': 4, 'failure': 4})
        self.assertLessEqual(max(conds.count(i) for i in range(4)) - min(conds.count(i) for i in range(4)), 1)

    def test_inference_contract_rejects_gt(self):
        assert_inference_contract({'visibility', 'bbox_center_delta', 'bbox_scale_delta', 'temporal_latent_disagreement'})
        with self.assertRaises(ValueError):
            assert_inference_contract({'visibility', 'camera_error'})

    def test_checkpoint_contains_strict_resume_state(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        payload = training_checkpoint(model, optimizer, 17, {'stable': 3}, {'best': 1.0})
        required = {
            'adapter_state_dict', 'optimizer_state_dict', 'current_step',
            'rng_state', 'sampler_state', 'metadata',
        }
        self.assertTrue(required.issubset(payload))
        self.assertEqual(payload['current_step'], 17)
        self.assertEqual(payload['sampler_state']['counters']['stable'], 3)

    def test_locked_gradient_probe_is_reused_exactly(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'probe.json'
            expected = {'lambdas': {'cam3d': 1.0, 'reliability': 0.8217990987546042}}
            path.write_text(json.dumps(expected))
            self.assertEqual(locked_gradient_probe(path), expected)

    def test_output_layer_norm_survives_large_finite_mamba_state(self):
        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output_norm = torch.nn.LayerNorm(4)
        holder = Holder()
        x = torch.tensor([[[4.5e29, -4.1e29, 3.8e29, -3.7e29]]], dtype=torch.float32)
        out = stable_output_layer_norm(holder, x)
        self.assertTrue(torch.isfinite(out).all())

    def test_parameter_free_interlayer_norm_prevents_mamba_cascade(self):
        class Amplify(torch.nn.Module):
            def forward(self, x): return x * 1e20
        x = torch.ones(1, 16, 4)
        out = stable_mamba_stack([Amplify(), Amplify()], x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(float(out.abs().max()), 10.0)


if __name__ == '__main__':
    unittest.main()
