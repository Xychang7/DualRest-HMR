from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np

ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))


def make_payload():
    from sam_3d_body.data.dynamesh_v2_cache_v2 import make_alignment_hashes
    rng = np.random.default_rng(3)
    payload = {
        "schema_version": np.asarray("dynamesh_v2_feature_cache_v2"),
        "split": np.asarray("train"), "sequence": np.asarray("seq"),
        "person_id": np.asarray(2, dtype=np.int64),
        "condition_names": np.asarray(["clean", "occ15", "occ35", "occ55"]),
        "occlusion_ratio": np.asarray([0, .15, .35, .55], dtype=np.float32),
        "frame_id": np.asarray([10, 12], dtype=np.int64),
        "local_frame_index": np.asarray([5, 6], dtype=np.int64),
        "valid": np.ones(2, dtype=bool), "visibility": np.full(2, -1, dtype=np.int8),
        "pose_token": rng.normal(size=(4, 2, 1024)).astype(np.float16),
        "mhr_init_pose": rng.normal(size=(4, 2, 519)).astype(np.float32),
        "body_cont": rng.normal(size=(4, 2, 260)).astype(np.float32),
        "gt_pose": rng.normal(size=(2, 72)).astype(np.float32),
        "gt_transl": rng.normal(size=(2, 3)).astype(np.float32),
        "gt_betas": rng.normal(size=(2, 10)).astype(np.float32),
        "camera_extrinsics": np.repeat(np.eye(4, dtype=np.float32)[None], 2, 0),
    }
    payload["alignment_sha256"] = make_alignment_hashes(payload)
    return payload


class CacheV2Tests(unittest.TestCase):
    def test_valid_payload_and_atomic_checksum(self):
        from sam_3d_body.data.dynamesh_v2_cache_v2 import atomic_save_track, validate_track_payload
        payload = make_payload(); validate_track_payload(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.npz"
            digest = atomic_save_track(path, payload)
            self.assertTrue(path.exists()); self.assertEqual(len(digest), 64)
            self.assertFalse((Path(directory) / "track.npz.tmp").exists())

    def test_test_split_and_duplicate_or_nonmonotonic_frames_are_rejected(self):
        from sam_3d_body.data.dynamesh_v2_cache_v2 import validate_track_payload
        for field, value in (("split", np.asarray("test")), ("frame_id", np.asarray([10, 10])), ("local_frame_index", np.asarray([6, 5]))):
            payload = make_payload(); payload[field] = value
            with self.assertRaises(ValueError): validate_track_payload(payload)

    def test_token_init_frame_person_condition_alignment_tampering_is_rejected(self):
        from sam_3d_body.data.dynamesh_v2_cache_v2 import validate_track_payload
        mutations = []
        token = make_payload(); token["pose_token"] = token["pose_token"][:, ::-1].copy(); mutations.append(token)
        init = make_payload(); init["mhr_init_pose"] = init["mhr_init_pose"][::-1].copy(); mutations.append(init)
        frame = make_payload(); frame["frame_id"] = np.asarray([12, 14]); mutations.append(frame)
        person = make_payload(); person["person_id"] = np.asarray(3); mutations.append(person)
        condition = make_payload(); condition["condition_names"] = condition["condition_names"][::-1].copy(); mutations.append(condition)
        for payload in mutations:
            with self.assertRaisesRegex(ValueError, "alignment"):
                validate_track_payload(payload)

    def test_float16_gate_reports_token_and_decode_errors(self):
        from sam_3d_body.data.dynamesh_v2_cache_v2 import float16_error_metrics
        tokens = np.linspace(-2, 2, 2 * 1024, dtype=np.float32).reshape(2, 1024)
        matrix = np.ones((1024, 260), dtype=np.float32) / 1024
        result = float16_error_metrics(tokens, lambda x: x @ matrix)
        self.assertGreater(result["token_max_abs"], 0)
        self.assertLess(result["decoded_max_abs"], 1e-3)

    def test_variant_batch_layout_restores_condition_frame_alignment(self):
        from sam_3d_body.data.dynamesh_v2_cache_v2 import make_variant_batch, restore_variant_batch
        paths = [[f"c{condition}_f{frame}" for frame in range(3)] for condition in range(4)]
        flat, mapping = make_variant_batch(paths)
        self.assertEqual(flat[:4], ["c0_f0", "c1_f0", "c2_f0", "c3_f0"])
        restored = restore_variant_batch(list(range(12)), mapping, conditions=4, frames=3)
        self.assertEqual(restored.shape, (4, 3))
        self.assertEqual(restored[2, 1], mapping.index((2, 1)))


if __name__ == "__main__": unittest.main()
