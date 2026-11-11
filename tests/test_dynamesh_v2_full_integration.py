from pathlib import Path
import sys

import torch
import unittest

ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))
sys.path.insert(0, str(ROOT / "models/sam_3d_body/sam_3d_body/models/modules"))
sys.path.insert(0, str(ROOT / "scripts"))


def sample_parameters():
    return {
        "global_rot": torch.full((2, 3), 1.0),
        "body_pose": torch.full((2, 8), 2.0),
        "hand": torch.full((2, 4), 3.0),
        "scale": torch.full((2, 2), 4.0),
        "shape": torch.full((2, 5), 5.0),
        "face": torch.full((2, 6), 6.0),
        "jaw": torch.full((2, 3), 7.0),
        "pred_cam": torch.full((2, 3), 8.0),
        "pred_cam_t": torch.full((2, 3), 9.0),
    }


class FullIntegrationContractTests(unittest.TestCase):
 def test_composition_takes_only_body_articulation_and_camera_from_branches(self):
    from dynamesh_v2_integration import compose_parameters

    raw = sample_parameters()
    body = {"body_pose": torch.full((2, 8), 20.0)}
    camera = {
        "pred_cam": torch.full((2, 3), 80.0),
        "pred_cam_t": torch.full((2, 3), 90.0),
    }
    result = compose_parameters(raw, body, camera)
    self.assertTrue(torch.equal(result["body_pose"], body["body_pose"]))
    self.assertTrue(torch.equal(result["pred_cam"], camera["pred_cam"]))
    self.assertTrue(torch.equal(result["pred_cam_t"], camera["pred_cam_t"]))
    for key in ("global_rot", "hand", "scale", "shape", "face", "jaw"):
        self.assertTrue(torch.equal(result[key], raw[key]))


 def test_branch_isolation_and_disable_equivalence(self):
    from dynamesh_v2_integration import compose_parameters

    raw = sample_parameters()
    body_a = {"body_pose": torch.zeros_like(raw["body_pose"])}
    body_b = {"body_pose": torch.ones_like(raw["body_pose"])}
    camera_a = {"pred_cam": raw["pred_cam"] + 1, "pred_cam_t": raw["pred_cam_t"] + 1}
    camera_b = {"pred_cam": raw["pred_cam"] + 2, "pred_cam_t": raw["pred_cam_t"] + 2}
    self.assertTrue(torch.equal(compose_parameters(raw, body_a, camera_a)["pred_cam_t"], compose_parameters(raw, body_b, camera_a)["pred_cam_t"]))
    self.assertTrue(torch.equal(compose_parameters(raw, body_a, camera_a)["body_pose"], compose_parameters(raw, body_a, camera_b)["body_pose"]))
    self.assertTrue(torch.equal(compose_parameters(raw, None, camera_a)["body_pose"], raw["body_pose"]))
    self.assertTrue(torch.equal(compose_parameters(raw, body_a, None)["pred_cam_t"], raw["pred_cam_t"]))


 def test_track_alignment_is_exact_and_rejects_shift_or_person_crossing(self):
    from dynamesh_v2_integration import assert_track_alignment

    a = {"sequence": "s", "person_id": 1, "frame_id": [0, 2, 4]}
    assert_track_alignment(a, dict(a))
    with self.assertRaisesRegex(ValueError, "frame"):
        assert_track_alignment(a, {**a, "frame_id": [2, 4, 6]})
    with self.assertRaisesRegex(ValueError, "person"):
        assert_track_alignment(a, {**a, "person_id": 2})


 def test_full_finite_contract_checks_nested_tensors(self):
    from dynamesh_v2_integration import assert_finite

    assert_finite({"body": torch.ones(2), "camera": {"t": torch.zeros(3)}})
    with self.assertRaisesRegex(FloatingPointError, "camera.t"):
        assert_finite({"camera": {"t": torch.tensor([float("nan")])}})


 def test_split_contract_permanently_rejects_test(self):
    from evaluate_dynamesh_v2_full_integration import assert_validation_only

    assert_validation_only("validation")
    with self.assertRaisesRegex(ValueError, "test"):
        assert_validation_only("test")

 def test_official_aggregate_contract_uses_frozen_evaluator_totals(self):
    from evaluate_dynamesh_v2_full_integration import official_aggregate_metrics

    body_source = {"summary": {"baseline": {"clean": {"mpjpe_mm": 1, "pa_mpjpe_mm": 2, "body_teacher_geodesic_deg": 3}}, "v2": {"clean": {"mpjpe_mm": 4, "pa_mpjpe_mm": 5, "body_teacher_geodesic_deg": 6}}}}
    cam_item = {"translation": {"mean": 7, "median": 8, "p95": 9}, "depth_abs": {"mean": 10}, "reprojection": {"mean": 11, "p95": 12}, "velocity_error_mean": 13, "acceleration_error_mean": 14, "trajectory_p95": 15, "max_local_jump": 16}
    camera_source = {"methods": {name: {"conditions": {"clean": cam_item}} for name in ("raw_sam", "framewise", "framewise_true_temporal")}}
    body, camera = official_aggregate_metrics(body_source, camera_source, conditions=("clean",))
    self.assertEqual(body["full"]["clean"]["mpjpe_mm"], 4)
    self.assertEqual(camera["framewise_true_temporal"]["clean"]["translation_mean"], 7)


if __name__ == "__main__":
    unittest.main()
