import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/sam_3d_body'))
sys.path.insert(0, str(ROOT / 'scripts'))

from sam_3d_body.models.modules.camera_root_geometry import (
    compare_common_roots, get_3dpw_gt_camera_root,
    get_mhr_local_root, get_pred_camera_root,
)
from sam_3d_body.models.modules.camera_temporal import (
    CameraFeatureTemporalAdapter, CameraReliabilityFeatures,
    apply_reliability_gate,
)
from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
from mine_dynamesh_v2_camera_failures import validate_training_split


def reliability(batch=2, frames=5):
    return CameraReliabilityFeatures(
        visibility=torch.ones(batch, frames, 1),
        bbox_center_delta=torch.zeros(batch, frames, 2),
        bbox_scale_delta=torch.zeros(batch, frames, 1),
        mask_area_delta=None,
        temporal_latent_disagreement=None,
    )


class CameraAdapterTests(unittest.TestCase):
    def test_batched_and_unbatched_shape_and_zero_identity(self):
        model = CameraFeatureTemporalAdapter()
        x = torch.randn(2, 5, 1024)
        y, r = model(x, reliability())
        self.assertEqual(y.shape, x.shape); self.assertEqual(r.shape, (2, 5, 1))
        self.assertTrue(torch.equal(x, y)); self.assertTrue(torch.all((r >= 0) & (r <= 1)))
        xu = torch.randn(5, 1024)
        yu, ru = model(xu, reliability(1, 5).unbatched())
        self.assertTrue(torch.equal(xu, yu)); self.assertEqual(ru.shape, (5, 1))

    def test_forward_backward_and_body_camera_parameters_are_independent(self):
        camera = CameraFeatureTemporalAdapter(); body = BodyFeatureTemporalAdapter()
        self.assertTrue(all(id(f) != id(b) for f, b in zip(camera.layers_fwd, camera.layers_bwd)))
        self.assertTrue({id(p) for p in camera.parameters()}.isdisjoint({id(p) for p in body.parameters()}))

    def test_missing_reliability_fields_have_explicit_mask(self):
        values, mask = reliability().packed()
        self.assertEqual(values.shape, (2, 5, 6)); self.assertEqual(mask.shape, values.shape)
        self.assertTrue(torch.all(mask[..., 4] == 0)); self.assertTrue(torch.all(mask[..., 5] == 0))
        self.assertTrue(torch.all(mask[..., :4] == 1))

    def test_reliability_one_suppresses_and_zero_allows_correction(self):
        raw = torch.zeros(1, 2, 3)
        delta = torch.full_like(raw, 4.0)
        reliability = torch.tensor([[[1.0], [0.0]]])
        refined = apply_reliability_gate(raw, delta, reliability)
        self.assertTrue(torch.equal(refined[:, 0], raw[:, 0]))
        self.assertTrue(torch.equal(refined[:, 1], raw[:, 1] + delta[:, 1]))

    def test_frozen_perspective_head_identity(self):
        from sam_3d_body.models.heads.camera_head import PerspectiveHead
        head = PerspectiveHead(input_dim=1024, img_size=(256, 256))
        raw = torch.randn(1, 4, 1024); init = torch.zeros(4, 3)
        adapter = CameraFeatureTemporalAdapter(); refined, _ = adapter(raw, reliability(1, 4))
        base_cam = head(raw.squeeze(0), init); new_cam = head(refined.squeeze(0), init)
        self.assertTrue(torch.equal(base_cam, new_cam))
        points = torch.randn(4, 3, 3); center = torch.full((4, 2), 128.); size = torch.full((4,), 200.); image = torch.full((4, 2), 256.); K = torch.eye(3).repeat(4, 1, 1); K[:, 0, 0] = K[:, 1, 1] = 500.; K[:, :2, 2] = 128.
        base_t = head.perspective_projection(points, base_cam, center, size, image, K)['pred_cam_t']
        new_t = head.perspective_projection(points, new_cam, center, size, image, K)['pred_cam_t']
        self.assertTrue(torch.equal(base_t, new_t))
        root = points[:, 0]; self.assertTrue(torch.equal(get_pred_camera_root(root, base_t), get_pred_camera_root(root, new_t)))


class RootGeometryTests(unittest.TestCase):
    def test_homogeneous_world_to_camera_root(self):
        vertices = np.array([[[1., 2., 3.], [3., 2., 1.]]], dtype=np.float32)
        regressor = np.array([[0.5, 0.5]], dtype=np.float32)
        camera = np.eye(4, dtype=np.float32)[None]; camera[0, :3, 3] = [10., -2., 1.]
        root = get_3dpw_gt_camera_root(vertices, regressor, 0, camera)
        np.testing.assert_allclose(root, [[12., 0., 3.]])
        local = get_mhr_local_root(np.array([[[2., 0., 1.], [9., 9., 9.]]]), root_index=0)
        pred = get_pred_camera_root(local, np.array([[10., 0., 2.]]))
        report = compare_common_roots(pred, np.array([[12., 0., 3.]]))
        self.assertEqual(report['mean'], 0.0); self.assertEqual(report['per_frame'].shape, (1,))

    def test_rejects_implicit_unit_or_magic_offset(self):
        with self.assertRaises(ValueError):
            get_mhr_local_root(np.zeros((1, 2, 3)), 0, unit='centimeters')


class MiningGuardTests(unittest.TestCase):
    def test_only_train_split_allowed(self):
        self.assertEqual(validate_training_split('train'), 'train')
        for split in ('validation', 'val', 'test'):
            with self.assertRaises(ValueError): validate_training_split(split)


if __name__ == '__main__': unittest.main(verbosity=2)
