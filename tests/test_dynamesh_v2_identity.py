from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch

ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/sam_3d_body"))


class FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls = []
        self.mhr_calls = []

    def forward(self, token, init_pose):
        self.forward_calls.append((token.clone(), init_pose.clone()))
        return {"body_pose": token[:, :6] + init_pose[:, :6]}

    def mhr_forward(self, **kwargs):
        self.mhr_calls.append(kwargs)
        body = kwargs["body_pose_params"]
        return body[:, :2].unsqueeze(-1).expand(-1, -1, 3).clone()


class DynaMeshV2IdentityTests(unittest.TestCase):
    def test_identity_adapters_preserve_tokens_and_validate_visibility(self):
        from sam_3d_body.models.modules.dynamesh_v2_temporal import (
            BodyFeatureTemporalAdapter,
            CameraFeatureTemporalAdapter,
        )
        tokens = torch.randn(7, 1024)
        visibility = torch.tensor([1, 1, 0, 1, 0, 1, 1], dtype=torch.bool)
        self.assertTrue(torch.equal(BodyFeatureTemporalAdapter()(tokens, visibility), tokens))
        self.assertTrue(torch.equal(CameraFeatureTemporalAdapter()(tokens, visibility), tokens))
        with self.assertRaisesRegex(ValueError, "visibility"):
            BodyFeatureTemporalAdapter()(tokens, visibility[:-1])

    def test_feature_sequence_grouping_is_person_and_global_frame_ordered(self):
        from scripts.offline_app import collect_dynamesh_v2_sequences
        records = []
        for frame_id in (8, 2, 5):
            outputs, ids = [], []
            for person_id in (2, 1):
                outputs.append({
                    "pose_token": np.full(1024, frame_id + person_id / 10, np.float32),
                    "visibility": int((frame_id + person_id) % 2),
                    "person_id": person_id,
                    "global_frame_id": frame_id,
                })
                ids.append(person_id)
            records.append((frame_id, outputs, ids))
        sequences = collect_dynamesh_v2_sequences(records)
        self.assertEqual(sorted(sequences), [1, 2])
        self.assertEqual(sequences[1]["frame_ids"].tolist(), [2, 5, 8])
        self.assertEqual(sequences[1]["pose_tokens"].shape, (3, 1024))
        self.assertEqual(sequences[2]["visibility"].tolist(), [0, 1, 0])

    def test_identity_feature_refinement_is_parameter_and_vertex_equivalent(self):
        from models.sam_3d_body.notebook.utils import refine_oid_outputs_with_feature_adapters
        head = FakeHead()
        estimator = SimpleNamespace(model=SimpleNamespace(head_pose=head, head_camera=object()))
        outputs, originals = [], []
        for frame_id in range(4):
            token = np.arange(1024, dtype=np.float32) + frame_id
            body = np.arange(6, dtype=np.float32) + frame_id * 10
            vertex = np.stack([body[:2], -body[:2], -body[:2]], axis=-1)
            record = {
                "pose_token": token,
                "mhr_init_pose": np.zeros(519, np.float32),
                "visibility": frame_id != 2,
                "person_id": 3,
                "global_frame_id": frame_id,
                "global_rot": np.full(3, 20 + frame_id, np.float32),
                "body_pose_params": body,
                "hand_pose_params": np.full(108, 30 + frame_id, np.float32),
                "scale_params": np.full(28, 40 + frame_id, np.float32),
                "shape_params": np.full(45, 50 + frame_id, np.float32),
                "expr_params": np.full(72, 60 + frame_id, np.float32),
                "pred_vertices": vertex,
            }
            outputs.append([record])
            originals.append({k: np.array(v, copy=True) for k, v in record.items() if isinstance(v, np.ndarray)})
        trace = {}
        refine_oid_outputs_with_feature_adapters(estimator, outputs, trace=trace)
        self.assertEqual(len(head.forward_calls), 2)
        self.assertEqual(len(head.mhr_calls), 1)
        self.assertTrue(torch.equal(head.forward_calls[0][0], head.forward_calls[1][0]))
        call = head.mhr_calls[0]
        for name, key in (("global_rot", "global_rot"), ("hand_pose_params", "hand_pose_params"), ("scale_params", "scale_params"), ("shape_params", "shape_params"), ("expr_params", "expr_params")):
            self.assertTrue(np.array_equal(call[name].cpu().numpy(), np.stack([x[key] for x in originals])))
        for output, original in zip(outputs, originals):
            self.assertTrue(np.array_equal(output[0]["body_pose_params"], original["body_pose_params"]))
            self.assertTrue(np.array_equal(output[0]["pred_vertices"], original["pred_vertices"]))
        self.assertEqual(trace["pose_tokens_shape"], (4, 1024))
        self.assertEqual(trace["visibility_shape"], (4,))
        self.assertTrue(trace["body_token_identity"])
        self.assertEqual(trace["camera_source"], "raw_pose_token")
        self.assertEqual(trace["max_parameter_abs_diff"], 0.0)
        self.assertEqual(trace["max_vertex_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
