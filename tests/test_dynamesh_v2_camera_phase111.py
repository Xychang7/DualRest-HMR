import sys, unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/root/sam-body4d-0.2.0'); sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from ablate_dynamesh_v2_camera_phase111 import (
 GlobalAffineCalibration, FramewiseLatentAdapter, deterministic_permutation,
 permute_time_aligned, assert_matching_supervision, reject_non_train_split,
 project_points_with_translation, trajectory_errors,
)

class Phase111Tests(unittest.TestCase):
 def test_global_calibration_has_only_six_or_two_parameters_and_no_token_input(self):
  xyz=GlobalAffineCalibration('xyz'); depth=GlobalAffineCalibration('depth')
  self.assertEqual(sum(p.numel() for p in xyz.parameters()),6)
  self.assertEqual(sum(p.numel() for p in depth.parameters()),2)
  x=torch.randn(4,3); self.assertEqual(xyz(x).shape,x.shape); self.assertTrue(torch.equal(depth(x)[:,:2],x[:,:2]))
  with self.assertRaises(TypeError): xyz(x,torch.randn(4,1024))
 def test_framewise_adapter_does_not_mix_frames(self):
  m=FramewiseLatentAdapter(); x=torch.randn(1,4,1024); y=m(x)
  x2=x.clone(); x2[:,1]+=10; y2=m(x2)
  self.assertTrue(torch.equal(y[:,0],y2[:,0])); self.assertTrue(torch.equal(y[:,2],y2[:,2]))
 def test_framewise_output_starts_as_identity(self):
  m=FramewiseLatentAdapter(); x=torch.randn(2,3,1024); self.assertTrue(torch.equal(m(x),x))
 def test_deterministic_permutation(self):
  a=deterministic_permutation(16,7,'x'); b=deterministic_permutation(16,7,'x')
  self.assertTrue(np.array_equal(a,b)); self.assertFalse(np.array_equal(a,np.arange(16)))
 def test_permutation_keeps_all_fields_aligned(self):
  p=np.array([2,0,1]); d={'token':torch.tensor([10,20,30]),'target':np.array([1,2,3]),'constant':'x'}
  q=permute_time_aligned(d,p,{'token','target'}); self.assertEqual(q['token'].tolist(),[30,10,20]); self.assertEqual(q['target'].tolist(),[3,1,2])
 def test_supervision_contract_detects_mismatch(self):
  a={'t_all':np.zeros((2,3)),'weight':np.ones(2)}; assert_matching_supervision(a,a)
  with self.assertRaises(AssertionError): assert_matching_supervision(a,{'t_all':np.ones((2,3)),'weight':np.ones(2)})
 def test_validation_and_test_are_rejected(self):
  reject_non_train_split('train')
  reject_non_train_split('3dpw_train_only')
  for split in ('validation','test'):
   with self.assertRaises(ValueError): reject_non_train_split(split)
 def test_projection_is_finite(self):
  pts=torch.zeros(2,3,3); t=torch.tensor([[0.,0.,2.],[0.,0.,3.]]); K=torch.eye(3); K[0,0]=K[1,1]=1000
  out=project_points_with_translation(pts,t,K); self.assertEqual(out.shape,(2,3,2)); self.assertTrue(torch.isfinite(out).all())
 def test_trajectory_errors_zero_for_identical(self):
  x=np.arange(15,dtype=float).reshape(5,3); m=trajectory_errors(x,x)
  self.assertEqual(m['velocity_mean'],0); self.assertEqual(m['acceleration_mean'],0)

if __name__=='__main__': unittest.main()
