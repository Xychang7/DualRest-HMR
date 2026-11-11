import sys
import unittest
from pathlib import Path

import torch

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path[:0] = [str(ROOT), str(ROOT/'models/sam_3d_body'), str(ROOT/'scripts')]

from ablate_dynamesh_v2_camera_phase16 import (
    apply_gate_mode,
    depth_scale_diagnostic,
    derive_balanced_lambdas,
    predicted_reliability_weight,
    select_verdict,
)


class Phase16Contracts(unittest.TestCase):
    def test_balanced_lambdas_use_measured_raw_gradients(self):
        raw = {'teacher': .0001842242782, 'reproj': .01638062298, 'reliability': .00861416198, 'anchor': .0001783647604}
        lambdas, weighted = derive_balanced_lambdas(raw)
        self.assertEqual(lambdas['reproj'], 1.0)
        core = [weighted[name] for name in ('teacher', 'reproj', 'anchor')]
        self.assertLessEqual(max(core)/min(core), 1.51)
        self.assertGreater(weighted['reliability'], 0.)
        self.assertLess(weighted['reliability'], weighted['reproj'])

    def test_predicted_reliability_strengthens_teacher_without_gt_leakage(self):
        reliability = torch.tensor([.1, .5, .9], requires_grad=True)
        weight = predicted_reliability_weight(reliability, torch.ones(3))
        self.assertTrue(torch.all(weight[:-1] < weight[1:]))
        self.assertFalse(weight.requires_grad)

    def test_fixed_open_gate_applies_full_delta(self):
        raw = torch.zeros(1, 2, 3)
        delta = torch.ones_like(raw)
        reliability = torch.tensor([[[.8], [.2]]])
        learned = apply_gate_mode(raw, delta, reliability, 'learned')
        opened = apply_gate_mode(raw, delta, reliability, 'fixed_open')
        torch.testing.assert_close(learned[..., 0], torch.tensor([[.2, .8]]))
        torch.testing.assert_close(opened, torch.ones_like(raw))

    def test_depth_diagnostic_flags_oracle_like_depth_collapse(self):
        raw_cam = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        new_cam = torch.tensor([[2., 0., 0.], [2., 0., 0.]])
        raw_t = torch.tensor([[0., 0., 9.5], [0., 0., 9.4]])
        new_t = torch.tensor([[0., 0., 4.0], [0., 0., 4.1]])
        report = depth_scale_diagnostic(raw_cam, new_cam, raw_t, new_t, 0)
        self.assertTrue(report['depth_scale_degeneracy'])
        self.assertLess(report['failure_z_change_percent'], -50.)

    def test_verdict_distinguishes_gate_bottleneck(self):
        result = select_verdict(
            exp_a={'failure_gain': .12, 'window_gain': .03, 'safe': True, 'preserved': True},
            exp_b={'failure_gain': .12, 'window_gain': .03, 'safe': True, 'preserved': True},
            exp_c={'failure_gain': .35, 'window_gain': .12, 'safe': True, 'preserved': True},
        )
        self.assertEqual(result, 'RELIABILITY_GATING_BOTTLENECK')


if __name__ == '__main__':
    unittest.main(verbosity=2)
