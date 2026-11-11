from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path("/root/sam-body4d-0.2.0")
sys.path.insert(0, str(ROOT))


class RecordingSmoother(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return x + 0.25


class RecordingHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.calls = []

    def mhr_forward(self, **kwargs):
        self.calls.append(kwargs)
        batch = kwargs["global_rot"].shape[0]
        return torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 2, 3)


def _frame(frame_index):
    raw = np.arange(266, dtype=np.float32) + frame_index * 1000
    return [{
        "pred_pose_raw": raw,
        "hand_pose_params": np.full(12, frame_index + 10, dtype=np.float32),
        "scale_params": np.full(28, frame_index + 20, dtype=np.float32),
        "shape_params": np.full(45, frame_index + 30, dtype=np.float32),
        "expr_params": np.full(72, frame_index + 40, dtype=np.float32),
        "pred_vertices": np.zeros((2, 3), dtype=np.float32),
    }]


class DynaMesh4DInferenceTests(unittest.TestCase):
    def test_process_image_with_mask_does_not_run_batch_local_mamba(self):
        source = (
            ROOT / "models/sam_3d_body/notebook/utils.py"
        ).read_text()
        start = source.index("def process_image_with_mask(")
        end = source.index("def process_image_with_bbox(", start)
        self.assertNotIn(
            "refine_oid_outputs_with_mamba(", source[start:end]
        )

    def test_25_batches_are_grouped_by_person_and_refined_once(self):
        from scripts.offline_app import refine_complete_video_sequences

        frame_records = []
        for batch_index in range(25):
            for local_index in range(4):
                frame_index = batch_index * 4 + local_index
                outputs = []
                person_ids = []
                for person_id in (1, 2):
                    output = _frame(frame_index)[0]
                    output["person_marker"] = person_id
                    outputs.append(output)
                    person_ids.append(person_id)
                frame_records.append((frame_index, outputs, person_ids))

        calls = []

        def fake_refiner(estimator, oid_outputs, trace=None):
            calls.append(oid_outputs)
            trace.update({
                "sequence_length": len(oid_outputs),
                "mamba_input_shape": (1, len(oid_outputs), 266),
                "mamba_output_shape": (1, len(oid_outputs), 266),
            })
            for frame_outputs in oid_outputs:
                frame_outputs[0]["pred_vertices"][:] = 99
            return oid_outputs

        traces = refine_complete_video_sequences(
            estimator=object(),
            frame_records=frame_records,
            refiner=fake_refiner,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual([len(sequence) for sequence in calls], [100, 100])
        for person_index, sequence in enumerate(calls, start=1):
            self.assertEqual(
                [int(frame[0]["pred_pose_raw"][0] // 1000) for frame in sequence],
                list(range(100)),
            )
            self.assertTrue(all(
                frame[0]["person_marker"] == person_index for frame in sequence
            ))
        self.assertEqual(traces[1]["mamba_input_shape"], (1, 100, 266))
        self.assertEqual(traces[1]["mamba_output_shape"], (1, 100, 266))
        self.assertEqual(traces[2]["sequence_length"], 100)
        self.assertTrue(all(
            np.all(output["pred_vertices"] == 99)
            for _, outputs, _ in frame_records
            for output in outputs
        ))

    def test_run_inference_batch_preserves_pred_pose_raw(self):
        source = (
            ROOT
            / "models/sam_3d_body/sam_3d_body/models/meta_arch/sam3d_body.py"
        ).read_text()
        start = source.index("    def run_inference_batch(")
        end = source.index("    def run_keypoint_prompt(", start)
        function_source = source[start:end]
        self.assertNotIn("pred_pose_raw is not valid anymore", function_source)

    def test_full_sequence_refinement_uses_mamba_pose_and_kalman_hand(self):
        from sam_3d_body.models.modules import geometry_utils, mhr_utils
        from models.sam_3d_body.notebook import utils as notebook_utils
        import roma

        oid_outputs = [_frame(i) for i in range(4)]
        original_hands = [frame[0]["hand_pose_params"].copy() for frame in oid_outputs]
        smoother = RecordingSmoother()
        head = RecordingHead()
        estimator = SimpleNamespace(model=SimpleNamespace(head_pose=head))
        trace = {}

        notebook_utils.refine_oid_outputs_with_mamba(
            estimator, oid_outputs, smoother=smoother, trace=trace
        )

        expected_raw = torch.from_numpy(
            np.stack([_frame(i)[0]["pred_pose_raw"] for i in range(4)])
        )
        expected_smoothed = expected_raw + 0.25
        self.assertEqual(tuple(smoother.inputs[0].shape), (1, 4, 266))
        self.assertTrue(torch.equal(smoother.inputs[0].squeeze(0), expected_raw))
        self.assertEqual(len(head.calls), 1)

        call = head.calls[0]
        self.assertEqual(call["global_rot"].dtype, torch.float32)
        self.assertEqual(call["body_pose_params"].dtype, torch.float32)
        self.assertEqual(call["hand_pose_params"].dtype, torch.float32)
        expected_global = roma.rotmat_to_euler(
            "ZYX", geometry_utils.rot6d_to_rotmat(expected_smoothed[:, :6])
        )
        expected_body = mhr_utils.compact_cont_to_model_params_body(
            expected_smoothed[:, 6:]
        )
        expected_body[:, mhr_utils.mhr_param_hand_mask] = 0
        expected_body[:, -3:] = 0
        self.assertTrue(torch.allclose(call["global_rot"], expected_global))
        self.assertTrue(torch.allclose(call["body_pose_params"], expected_body))
        self.assertTrue(torch.equal(
            call["hand_pose_params"], torch.from_numpy(np.stack(original_hands))
        ))
        self.assertEqual(trace["raw_stack_shape"], (4, 266))
        self.assertEqual(trace["mamba_input_shape"], (1, 4, 266))
        self.assertEqual(trace["mamba_output_shape"], (1, 4, 266))
        self.assertEqual(trace["global_rot_source"], "mamba[:6]")
        self.assertEqual(trace["body_pose_source"], "mamba[6:]")
        self.assertEqual(trace["hand_source"], "kalman")
        self.assertTrue(trace["mhr_forward_called"])
        self.assertTrue(trace["pred_vertices_overwritten"])
        expected_verts = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        expected_verts[..., [1, 2]] *= -1
        for index, frame in enumerate(oid_outputs):
            self.assertTrue(np.array_equal(
                frame[0]["pred_vertices"], expected_verts[index].numpy()
            ))

    def test_smoother_loader_caches_checkpoint(self):
        from sam_3d_body.models.modules import mamba_smoother

        class TinySmoother(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        loads = []

        def fake_load(path, map_location, **kwargs):
            loads.append((path, str(map_location)))
            return {"weight": torch.ones(1)}

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            mamba_smoother, "BiMambaSmoother", TinySmoother
        ), mock.patch.object(torch, "load", fake_load):
            mamba_smoother.load_bimamba_smoother.cache_clear()
            checkpoint = str(Path(temp_dir) / "weights.pth")
            first = mamba_smoother.load_bimamba_smoother(checkpoint, "cpu")
            second = mamba_smoother.load_bimamba_smoother(checkpoint, "cpu")
        self.assertIs(first, second)
        self.assertEqual(len(loads), 1)
        self.assertFalse(first.training)


if __name__ == "__main__":
    unittest.main()
