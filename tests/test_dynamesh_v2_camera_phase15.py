import sys, unittest
from pathlib import Path
import torch

ROOT=Path('/root/sam-body4d-0.2.0')
sys.path[:0]=[str(ROOT),str(ROOT/'models/sam_3d_body'),str(ROOT/'scripts')]

from diagnose_dynamesh_v2_camera_phase15 import (
    clone_independent_condition_variables,
    gradient_norm,
    weighted_gradient_report,
)


class OracleContractTests(unittest.TestCase):
    def test_condition_variables_are_initialized_from_self_and_do_not_share_storage(self):
        raw=torch.arange(4*2*3,dtype=torch.float32).reshape(4,2,3)
        variables=clone_independent_condition_variables(raw)
        self.assertEqual(len(variables),4)
        for index,value in enumerate(variables):
            self.assertTrue(torch.equal(value.detach(),raw[index]))
            self.assertNotEqual(value.data_ptr(),raw[index].data_ptr())
        variables[0].data.add_(100)
        self.assertTrue(torch.equal(variables[1].detach(),raw[1]))

    def test_gradient_norm_and_lambda_weighting(self):
        parameter=torch.nn.Parameter(torch.tensor([3.,4.]))
        loss=(parameter**2).sum()
        raw=gradient_norm(loss,[parameter])
        self.assertAlmostEqual(raw,10.0,places=6)
        report=weighted_gradient_report('teacher',loss,[parameter],1000.)
        self.assertAlmostEqual(report['raw_gradient_norm'],10.0,places=6)
        self.assertAlmostEqual(report['weighted_gradient_norm'],10000.0,places=4)

    def test_unused_loss_has_zero_gradient_norm(self):
        parameter=torch.nn.Parameter(torch.ones(2))
        independent=torch.tensor(2.,requires_grad=True)
        self.assertEqual(gradient_norm(independent,[parameter]),0.0)


if __name__=='__main__': unittest.main(verbosity=2)

