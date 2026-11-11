import inspect

import numpy as np
import torch

import evaluate_dynamesh_v2_unified_3dpw as unified
import run_three_dpw_oc_full as full


def test_evaluate_sequence_requires_explicit_prepare_track_callback():
    signature = inspect.signature(full.evaluate_sequence)
    assert "prepare_track_fn" in signature.parameters


def test_condition_tensor_records_actual_runtime_values():
    assert hasattr(unified, "build_condition_tensor")
    reference = torch.zeros((2, 16, 1024), dtype=torch.float32)
    condition, audit = unified.build_condition_tensor(reference, 0.15)
    assert tuple(condition.shape) == (2, 16)
    assert condition.dtype == reference.dtype
    assert condition.device == reference.device
    assert np.isclose(audit["requested_alpha"], 0.15)
    assert np.isclose(audit["actual_min"], 0.15)
    assert np.isclose(audit["actual_max"], 0.15)
    assert np.isclose(audit["actual_mean"], 0.15)
    assert audit["shape"] == [2, 16]
    assert audit["dtype"] == "torch.float32"
    assert audit["device"] == "cpu"


def test_condition_tensor_rejects_out_of_range_alpha():
    assert hasattr(unified, "build_condition_tensor")
    reference = torch.zeros((1, 16, 1024), dtype=torch.float32)
    try:
        unified.build_condition_tensor(reference, 1.01)
    except ValueError as exc:
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("out-of-range alpha was accepted")
