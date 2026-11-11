from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))


class DynaMeshV2CacheTests(unittest.TestCase):
    def test_manifest_splits_are_disjoint_and_test_is_rejected(self):
        from sam_3d_body.data.dynamesh_v2_cache import build_split_manifests
        manifests = build_split_manifests(
            ROOT / "tests/fixtures/3dpw_sequence_files", allow_missing=True
        )
        manifests = {
            "train": {"sequences": ["train_a", "train_b"]},
            "validation": {"sequences": ["val_a"]},
            "test": {"sequences": ["test_a"]},
        }
        train = set(manifests["train"]["sequences"])
        val = set(manifests["validation"]["sequences"])
        test = set(manifests["test"]["sequences"])
        self.assertFalse(train & val or train & test or val & test)

        from sam_3d_body.data.dynamesh_v2_cache import validate_cache_split
        validate_cache_split("train")
        validate_cache_split("validation")
        with self.assertRaisesRegex(ValueError, "test"):
            validate_cache_split("test")

    def test_schema_requires_strict_person_frame_metadata_and_shapes(self):
        from sam_3d_body.data.dynamesh_v2_cache import validate_cache_payload
        payload = {
            "schema_version": "dynamesh_v2_feature_cache_v1",
            "split": "train",
            "sequence": "train_a",
            "person_id": 0,
            "frame_id": np.array([0, 2]),
            "pose_token_clean": np.zeros((2, 1024), np.float32),
            "visibility": np.array([-1, -1], np.int8),
            "valid": np.array([True, True]),
            "baseline_body_pose": np.zeros((2, 133), np.float32),
            "baseline_global_rot": np.zeros((2, 3), np.float32),
            "gt_body_pose": np.zeros((2, 69), np.float32),
            "gt_global_orient": np.zeros((2, 3), np.float32),
            "gt_betas": np.zeros((2, 10), np.float32),
            "gt_transl": np.zeros((2, 3), np.float32),
            "gt_joints3d": np.zeros((2, 24, 3), np.float32),
            "bbox": np.zeros((2, 4), np.float32),
            "camera_intrinsics": np.zeros((2, 3, 3), np.float32),
        }
        validate_cache_payload(payload)
        bad = dict(payload)
        bad["frame_id"] = np.array([0, 0])
        with self.assertRaisesRegex(ValueError, "frame_id"):
            validate_cache_payload(bad)
        bad = dict(payload)
        bad["pose_token_clean"] = np.zeros((2, 266), np.float32)
        with self.assertRaisesRegex(ValueError, "pose_token_clean"):
            validate_cache_payload(bad)

    def test_human_bbox_occlusion_is_seeded_and_ratio_ordered(self):
        from sam_3d_body.data.dynamesh_v2_cache import apply_bbox_rectangle_occlusion
        image = np.full((120, 160, 3), 200, np.uint8)
        bbox = np.array([40, 20, 120, 100], np.float32)
        light, light_meta = apply_bbox_rectangle_occlusion(image, bbox, 0.15, seed=7)
        again, again_meta = apply_bbox_rectangle_occlusion(image, bbox, 0.15, seed=7)
        heavy, heavy_meta = apply_bbox_rectangle_occlusion(image, bbox, 0.55, seed=7)
        self.assertTrue(np.array_equal(light, again))
        self.assertEqual(light_meta, again_meta)
        self.assertLess(light_meta["actual_ratio"], heavy_meta["actual_ratio"])
        self.assertGreater(np.count_nonzero(light != image), 0)
        self.assertGreater(np.count_nonzero(heavy != image), np.count_nonzero(light != image))
        x1, y1, x2, y2 = light_meta["rectangle_xyxy"]
        self.assertGreaterEqual(x1, bbox[0]); self.assertGreaterEqual(y1, bbox[1])
        self.assertLessEqual(x2, bbox[2]); self.assertLessEqual(y2, bbox[3])


if __name__ == "__main__":
    unittest.main()
