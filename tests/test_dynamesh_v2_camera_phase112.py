import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/sam-body4d-0.2.0');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from train_dynamesh_v2_camera_phase112 import (pair_weights,velocity_loss,acceleration_loss,
 assert_temporal_optimizer_ownership, temporal_residual_forward, reject_split,
 assert_inference_fields, trajectory_diagnostics)
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter,CameraReliabilityFeatures

class Tests(unittest.TestCase):
 def test_pair_weights_use_adjacent_minimum(self):
  w=torch.tensor([.9,.2,.7,.4]);self.assertTrue(torch.equal(pair_weights(w),torch.tensor([.2,.2,.4])))
 def test_velocity_indexing(self):
  x=torch.tensor([[0.,0,0],[1,0,0],[3,0,0]]);self.assertEqual(float(velocity_loss(x,x,torch.ones(3))),0.)
  y=x.clone();y[2,0]=4;self.assertGreater(float(velocity_loss(y,x,torch.ones(3))),0.)
 def test_acceleration_indexing(self):
  x=torch.tensor([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]]);self.assertEqual(float(acceleration_loss(x,x,torch.ones(4))),0.)
 def test_optimizer_excludes_frozen_framewise(self):
  f=torch.nn.Linear(2,2);t=torch.nn.Linear(2,2);f.requires_grad_(False);o=torch.optim.AdamW(t.parameters())
  assert_temporal_optimizer_ownership(o,t,f)
  o2=torch.optim.AdamW(list(t.parameters())+list(f.parameters()))
  with self.assertRaises(AssertionError):assert_temporal_optimizer_ownership(o2,t,f)
 def test_zero_init_temporal_is_identity(self):
  a=CameraFeatureTemporalAdapter();x=torch.randn(1,4,1024)
  feat=CameraReliabilityFeatures(visibility=torch.ones(1,4,1),bbox_center_delta=torch.zeros(1,4,2),bbox_scale_delta=torch.zeros(1,4,1),mask_area_delta=None,temporal_latent_disagreement=torch.zeros(1,4,1))
  y=temporal_residual_forward(a,x,feat);self.assertTrue(torch.equal(x,y))
 def test_split_guard(self):
  reject_split('3dpw_train_only')
  for x in ('validation','test'):
   with self.assertRaises(ValueError):reject_split(x)
 def test_no_inference_gt_leakage(self):
  assert_inference_fields({'visibility','bbox_center_delta','bbox_scale_delta','temporal_latent_disagreement'})
  with self.assertRaises(ValueError):assert_inference_fields({'visibility','t_all'})
 def test_trajectory_diagnostics_identical_is_zero(self):
  x=np.arange(18,dtype=float).reshape(6,3);r=trajectory_diagnostics(x,x)
  self.assertEqual(r['velocity_mean'],0);self.assertEqual(r['acceleration_mean'],0);self.assertEqual(r['max_local_jump_error'],0)
if __name__=='__main__':unittest.main()
