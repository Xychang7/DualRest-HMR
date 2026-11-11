from pathlib import Path
import sys
import unittest

import torch

ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))


class FeatureTemporalAdapterTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("mamba_ssm CUDA test requires a GPU")
        cls.device = torch.device("cuda")

    def test_batched_and_unbatched_contracts_are_exact_identity_at_init(self):
        from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter

        model = BodyFeatureTemporalAdapter().to(self.device)
        batched = torch.randn(1, 5, 1024, device=self.device)
        condition = torch.tensor([[0.0, 0.15, 0.35, 0.55, 0.0]], device=self.device)
        self.assertTrue(torch.equal(model(batched, condition), batched))
        self.assertTrue(torch.equal(model(batched[0], condition[0]), batched[0]))
        self.assertTrue(torch.equal(model.output_projection.weight, torch.zeros_like(model.output_projection.weight)))
        self.assertTrue(torch.equal(model.output_projection.bias, torch.zeros_like(model.output_projection.bias)))

    def test_condition_validation_and_independent_directions(self):
        from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter

        model = BodyFeatureTemporalAdapter().to(self.device)
        x = torch.randn(1, 3, 1024, device=self.device)
        with self.assertRaises(ValueError):
            model(x, torch.zeros(3, device=self.device))
        with self.assertRaises(ValueError):
            model(x, torch.tensor([[0.0, 1.1, 0.2]], device=self.device))
        with self.assertRaises(ValueError):
            model(x, torch.tensor([[0.0, float("nan"), 0.2]], device=self.device))
        self.assertEqual(len(model.layers_fwd), 2)
        self.assertEqual(len(model.layers_bwd), 2)
        forward_ids = {id(parameter) for layer in model.layers_fwd for parameter in layer.parameters()}
        backward_ids = {id(parameter) for layer in model.layers_bwd for parameter in layer.parameters()}
        self.assertFalse(forward_ids & backward_ids)

    def test_gradients_start_at_projection_then_reach_mamba(self):
        from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter

        model = BodyFeatureTemporalAdapter().to(self.device)
        x = torch.randn(1, 4, 1024, device=self.device)
        target = x + 0.1 * torch.randn_like(x)
        condition = torch.full((1, 4), 0.35, device=self.device)
        torch.nn.functional.smooth_l1_loss(model(x, condition), target).backward()
        self.assertGreater(float(model.output_projection.weight.grad.norm()), 0.0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.output_projection.weight.normal_(std=1e-4)
        torch.nn.functional.smooth_l1_loss(model(x, condition), target).backward()
        mamba_grad = sum(
            float(parameter.grad.norm())
            for layer in model.layers_fwd
            for parameter in layer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(mamba_grad, 0.0)

    def test_camera_adapter_remains_identity(self):
        from sam_3d_body.models.modules.dynamesh_v2_temporal import CameraFeatureTemporalAdapter

        model = CameraFeatureTemporalAdapter().to(self.device)
        x = torch.randn(4, 1024, device=self.device)
        visibility = torch.ones(4, device=self.device)
        self.assertTrue(torch.equal(model(x, visibility), x))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_balanced_schedule_and_optimizer_ownership(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from train_dynamesh_v2_micro_overfit import (
            assert_optimizer_owns_only,
            balanced_condition_schedule,
        )
        schedule = balanced_condition_schedule(20, seed=7)
        self.assertEqual([schedule.count(index) for index in range(4)], [5, 5, 5, 5])
        for start in range(0, 20, 4):
            self.assertEqual(set(schedule[start : start + 4]), {0, 1, 2, 3})
        module = torch.nn.Linear(3, 2)
        assert_optimizer_owns_only(torch.optim.AdamW(module.parameters()), module)
        other = torch.nn.Linear(3, 2)
        with self.assertRaises(AssertionError):
            assert_optimizer_owns_only(torch.optim.AdamW(other.parameters()), module)

    def test_body_rotation_decoder_excludes_hand_jaw_and_teacher_is_detached(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from train_dynamesh_v2_micro_overfit import (
            body_rotation_geodesic_loss,
            decode_body_articulation_rotmats,
        )
        from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body

        params = torch.zeros(2, 133, device=self.device)
        continuous = compact_model_params_to_cont_body(params)
        rotations = decode_body_articulation_rotmats(continuous)
        self.assertEqual(rotations.shape[-2:], (3, 3))
        self.assertGreater(rotations.shape[-3], 0)
        self.assertLess(rotations.shape[-3], 127)
        identity = torch.eye(3, device=self.device).expand_as(rotations).detach()
        loss = body_rotation_geodesic_loss(rotations, identity)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            body_rotation_geodesic_loss(rotations, identity.requires_grad_())


if __name__ == "__main__":
    unittest.main()
