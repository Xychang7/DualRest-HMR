import unittest

import numpy as np

from eval_3dpw_standard import (
    H36M_TO_J14,
    evaluate_records,
    regress_h36m_j14,
    strict_match_records,
    validate_records,
)


def record(sequence="seq", person_id=0, frame_id=0, vertices=None):
    if vertices is None:
        vertices = np.zeros((6890, 3), dtype=np.float32)
    return {"sequence": sequence, "person_id": person_id, "frame_id": frame_id, "smpl_vertices": vertices}


def selector_regressor():
    regressor = np.zeros((17, 6890), dtype=np.float64)
    regressor[np.arange(17), np.arange(17)] = 1.0
    return regressor


class StandardEvaluatorTests(unittest.TestCase):
    def test_strict_matching_uses_sequence_person_and_frame_not_input_order(self):
        pred, gt, keys = strict_match_records(
            [record("s", 1, 4), record("s", 0, 3)],
            [record("s", 0, 3), record("s", 1, 4)],
        )
        self.assertEqual(keys, [("s", 0, 3), ("s", 1, 4)])
        self.assertEqual([r["person_id"] for r in pred], [0, 1])
        self.assertEqual([r["person_id"] for r in gt], [0, 1])

    def test_missing_required_metadata_is_rejected(self):
        for missing in ("sequence", "person_id", "frame_id", "smpl_vertices"):
            with self.subTest(missing=missing):
                item = record()
                del item[missing]
                with self.assertRaisesRegex(ValueError, missing):
                    validate_records([item], source="prediction")

    def test_multiple_people_cannot_be_cross_matched(self):
        pred, gt, keys = strict_match_records(
            [record("s", 1, 9), record("s", 0, 9)],
            [record("s", 0, 9), record("s", 1, 9)],
        )
        self.assertEqual([r["person_id"] for r in pred], [0, 1])
        self.assertEqual([r["person_id"] for r in gt], [0, 1])
        self.assertEqual(keys, [("s", 0, 9), ("s", 1, 9)])

    def test_duplicate_or_missing_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records([record("s", 0, 1), record("s", 0, 1)], source="prediction")
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            strict_match_records([record("s", 0, 1)], [record("s", 0, 2)])

    def test_h36m_regression_then_j14_uses_shared_pelvis_root(self):
        vertices = np.zeros((1, 6890, 3), dtype=np.float64)
        vertices[0, :17, 0] = np.arange(17)
        joints = regress_h36m_j14(vertices, selector_regressor())
        expected = np.asarray(H36M_TO_J14, dtype=np.float64)[:, None] * np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(joints[0], expected)

    def test_mpjpe_is_pelvis_translation_aligned_but_not_scale_aligned(self):
        regressor = selector_regressor()
        gt = np.zeros((6890, 3), dtype=np.float64)
        gt[:17, 0] = np.arange(17)
        translated = gt + np.array([9.0, -3.0, 2.0])
        zero = evaluate_records([record(vertices=translated)], [record(vertices=gt)], regressor)
        self.assertAlmostEqual(zero["mpjpe_mm"], 0.0, places=10)
        scaled = gt * 2.0
        nonzero = evaluate_records([record(vertices=scaled)], [record(vertices=gt)], regressor)
        self.assertGreater(nonzero["mpjpe_mm"], 0.0)

    def test_pa_mpjpe_removes_similarity_transform(self):
        regressor = selector_regressor()
        gt = np.zeros((6890, 3), dtype=np.float64)
        theta = np.deg2rad(37.0)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        points = np.stack([np.arange(17), np.arange(17) ** 2, np.arange(17) % 3], axis=1).astype(float)
        gt[:17] = points
        pred = gt.copy()
        pred[:17] = 2.5 * points @ rotation.T + np.array([4.0, -7.0, 3.0])
        result = evaluate_records([record(vertices=pred)], [record(vertices=gt)], regressor)
        self.assertAlmostEqual(result["pa_mpjpe_mm"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
