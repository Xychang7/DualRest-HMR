import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

SCRIPT=Path('/root/sam-body4d-0.2.0/scripts/train_dynamesh_v2_camera_phase19.py')
spec=importlib.util.spec_from_file_location('phase19',SCRIPT)
phase19=importlib.util.module_from_spec(spec); spec.loader.exec_module(phase19)


class Phase19Tests(unittest.TestCase):
    def test_tau_is_derived_only_from_clean_train_distribution(self):
        rows=[{'split':'train','condition':'clean','fit_residual':x,'upper_lower_disagreement':x*2,'camera_error':10*x} for x in (.01,.02,.03)]
        rows.append({'split':'train','condition':'occ55','fit_residual':9.,'upper_lower_disagreement':9.,'camera_error':9.})
        tau=phase19.derive_train_taus(rows)
        self.assertAlmostEqual(tau['tau_fit'],.02); self.assertAlmostEqual(tau['tau_ul'],.04)
        self.assertAlmostEqual(tau['camera_close_threshold'],.15)
        with self.assertRaisesRegex(ValueError,'train'):
            phase19.derive_train_taus([{'split':'validation','condition':'clean','fit_residual':.1,'upper_lower_disagreement':.1}])

    def test_tau_ignores_unavailable_regional_diagnostics_without_imputation(self):
        rows=[{'split':'train','condition':'clean','fit_residual':.02,'upper_lower_disagreement':np.nan,'camera_error':1.},
              {'split':'train','condition':'clean','fit_residual':.04,'upper_lower_disagreement':.03,'camera_error':2.}]
        tau=phase19.derive_train_taus(rows)
        self.assertAlmostEqual(tau['tau_fit'],.03); self.assertAlmostEqual(tau['tau_ul'],.03)
        self.assertEqual(tau['finite_ul_count'],1)

    def test_target_reliability_decreases_with_either_residual(self):
        good=phase19.target_reliability(np.array([.01]),np.array([.01]),.04,.05)[0]
        bad_fit=phase19.target_reliability(np.array([.1]),np.array([.01]),.04,.05)[0]
        bad_ul=phase19.target_reliability(np.array([.01]),np.array([.1]),.04,.05)[0]
        self.assertGreater(good,bad_fit); self.assertGreater(good,bad_ul)
        self.assertTrue(0<=bad_fit<=1 and 0<=bad_ul<=1)

    def test_inference_reliability_forbids_gt_and_target_fields(self):
        phase19.assert_inference_reliability_fields({'visibility','bbox_center_delta','bbox_scale_delta','temporal_latent_disagreement'})
        for leaked in ('gt_2d','t_all','fit_residual','upper_lower_disagreement'):
            with self.assertRaisesRegex(ValueError,'leak'):
                phase19.assert_inference_reliability_fields({'visibility',leaked})

    def test_cam3d_loss_is_weighted_and_differentiable(self):
        pred=torch.tensor([[1.,2.,3.],[2.,3.,4.]],requires_grad=True)
        target=torch.zeros_like(pred); weight=torch.tensor([1.,0.])
        loss=phase19.weighted_cam3d_loss(pred,target,weight); loss.backward()
        self.assertGreater(float(pred.grad[0].abs().sum()),0)
        self.assertEqual(float(pred.grad[1].abs().sum()),0)

    def test_train_only_candidate_guard_and_common_root_prohibition(self):
        phase19.assert_train_only('train')
        with self.assertRaisesRegex(ValueError,'train'):
            phase19.assert_train_only('test')
        phase19.assert_loss_contract(('cam3d','reproj','reliability','anchor'))
        with self.assertRaisesRegex(ValueError,'common-root'):
            phase19.assert_loss_contract(('cam3d','common_root'))

    def test_output_is_independent_of_prior_phases(self):
        self.assertEqual(phase19.OUT.name,'dynamesh_v2_camera_phase19_multijoint_micro')
        self.assertNotIn(phase19.OUT,phase19.PRIOR_OUTPUTS)

    def test_aligned_track_slice_preserves_scalar_metadata(self):
        track={'sequence':np.asarray('seq'),'frame_id':np.arange(5),'gt_pose':np.zeros((5,72))}
        sliced=phase19.slice_aligned_track(track,slice(1,4))
        self.assertEqual(sliced['sequence'].item(),'seq')
        np.testing.assert_array_equal(sliced['frame_id'],[1,2,3])
        self.assertEqual(sliced['gt_pose'].shape,(3,72))

    def test_zero_gradient_anchor_gets_small_fixed_lambda_and_does_not_break_probe(self):
        result=phase19.balance_gradient_lambdas({'cam3d':2.,'reproj':.5,'reliability':1.,'anchor':0.})
        self.assertEqual(result['lambdas']['anchor'],.01)
        self.assertLessEqual(result['core_max_min_ratio'],4.01)
        self.assertLess(max(result['lambdas'].values()),100.)


if __name__=='__main__': unittest.main()
