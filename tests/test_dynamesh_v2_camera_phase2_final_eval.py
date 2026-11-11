import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path('/root/sam-body4d-0.2.0');sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_dynamesh_v2_camera_phase2 as e

class MetricsTest(unittest.TestCase):
 def test_error_summary(self):
  x=e.scalar_stats(np.array([1.,2.,3.,4.]));self.assertEqual(x['mean'],2.5);self.assertEqual(x['median'],2.5);self.assertAlmostEqual(x['p95'],3.85)
 def test_trajectory_zero_for_exact_target(self):
  x=np.arange(18,dtype=float).reshape(6,3);m=e.trajectory_metrics(x,x);self.assertEqual(m['velocity_error_mean'],0.);self.assertEqual(m['acceleration_error_mean'],0.)
 def test_train_threshold_bucket_contract(self):
  t={'camera_p25':2,'camera_p40':3,'camera_p60':5,'camera_p75':6,'fit_p75':1,'ul_p75':1,'reproj_p50':2,'reproj_p75':3,'body_p75':0}
  self.assertEqual(e.quality_bucket({'camera_error':1,'fit_residual':.1,'upper_lower_disagreement':.1,'reprojection_error':1},t),'stable')
  self.assertEqual(e.quality_bucket({'camera_error':7,'fit_residual':.1,'upper_lower_disagreement':.1,'reprojection_error':4},t),'failure')
  self.assertEqual(e.quality_bucket({'camera_error':4,'fit_residual':.1,'upper_lower_disagreement':.1,'reprojection_error':2},t),'medium')
 def test_test_split_rejected(self):
  with self.assertRaises(ValueError):e.assert_validation_only('test')
if __name__=='__main__':unittest.main()
