import sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT=Path('/root/sam-body4d-0.2.0')
sys.path.insert(0,str(ROOT/'scripts'))
import train_dynamesh_v2_camera_phase2 as phase2
import train_dynamesh_v2_camera_phase1 as phase1

def fixture(path, split):
    np.savez(path, split=np.asarray(split), person_id=np.asarray(0),
             condition_names=np.asarray(['clean','occ15','occ35','occ55']),
             frame_id=np.asarray([2]), pose_token=np.zeros((4,1,1024),np.float16),
             mhr_init_pose=np.zeros((4,1,519),np.float32))

class SplitContractTest(unittest.TestCase):
    def test_step40_resume_cursor_skips_exactly_first_40_windows(self):
        self.assertEqual(phase2.resume_cursor(current_step=40,current_epoch=1,windows_per_epoch=1370),(1,40))
        self.assertEqual(phase2.resume_cursor(current_step=1370,current_epoch=1,windows_per_epoch=1370),(2,0))

    def test_resume_hash_mismatch_is_rejected(self):
        checkpoint={'config_hash':'a','manifest_hash':'b'}
        phase2.assert_resume_identity(checkpoint,'a','b')
        with self.assertRaisesRegex(ValueError,'config hash mismatch'):
            phase2.assert_resume_identity(checkpoint,'x','b')

    def test_undefined_target_diagnostic_means_zero_reliability_not_sample_drop(self):
        target={'weight':np.asarray([.4,np.nan,.8]),'fit':np.asarray([.1,.2,.3]),'ul':np.asarray([.1,np.nan,.2])}
        fixed=phase2.enforce_target_reliability_contract(target)
        np.testing.assert_array_equal(fixed['weight'],np.asarray([.4,0.,.8]))
        self.assertEqual(len(fixed['weight']),3)
        self.assertTrue(np.isnan(fixed['ul'][1]))

    def test_annotation_split_follows_phase2_purpose(self):
        self.assertEqual(phase2.annotation_split_for_purpose('optimization'),'train')
        self.assertEqual(phase2.annotation_split_for_purpose('validation'),'validation')
        with self.assertRaisesRegex(ValueError,'unknown Phase-2 loader purpose'):
            phase2.annotation_split_for_purpose('test')

    def test_phase1_guard_remains_train_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v.npz'; fixture(p,'validation')
            with self.assertRaisesRegex(ValueError,'Phase-1 accepts train cache only'):
                phase1.load_track(p)

    def test_optimization_accepts_train_only(self):
        with tempfile.TemporaryDirectory() as d:
            t=Path(d)/'t.npz'; v=Path(d)/'v.npz'; fixture(t,'train');fixture(v,'validation')
            self.assertEqual(str(phase2.load_phase2_track(t,'optimization')['split'].item()),'train')
            with self.assertRaisesRegex(ValueError,'optimization requires train'):
                phase2.load_phase2_track(v,'optimization')

    def test_validation_accepts_validation_only(self):
        with tempfile.TemporaryDirectory() as d:
            t=Path(d)/'t.npz'; v=Path(d)/'v.npz'; fixture(t,'train');fixture(v,'validation')
            self.assertEqual(str(phase2.load_phase2_track(v,'validation')['split'].item()),'validation')
            with self.assertRaisesRegex(ValueError,'validation requires validation'):
                phase2.load_phase2_track(t,'validation')

    def test_test_is_always_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz'; fixture(p,'test')
            for purpose in ('optimization','validation'):
                with self.assertRaisesRegex(ValueError,'test is forbidden'):
                    phase2.load_phase2_track(p,purpose)

if __name__=='__main__': unittest.main()
