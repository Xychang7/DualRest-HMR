import inspect, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path('/root/sam-body4d-0.2.0')
sys.path.insert(0,str(ROOT/'scripts'))
import train_dynamesh_v2_camera_phase2 as m

def test_split_and_counts():
    x=m.audit_feature_manifest(ROOT/'data/dynamesh_v2/cache_v2/manifest.json')
    assert (x['train_tracks'],x['validation_tracks'])==(34,16)
    assert (x['train_frames'],x['validation_frames'])==(22735,10413)
    assert x['test_cache_files']==0

def test_test_split_hard_reject():
    with pytest.raises(ValueError): m.reject_split('test')

def test_windows_never_cross_discontinuity():
    assert m.contiguous_windows(np.array([0,1,2,4,5,6]),3)==[(0,3),(3,6)]

def test_optimizer_ownership_and_stage_b_freeze():
    import torch
    a=torch.nn.Linear(2,2); b=torch.nn.Linear(2,2).requires_grad_(False)
    o=torch.optim.AdamW(a.parameters()); m.assert_optimizer_ownership(o,a,[b])
    b.weight.requires_grad_(True)
    with pytest.raises(AssertionError): m.assert_optimizer_ownership(o,a,[b])

def test_resume_contract():
    assert {'optimizer_state_dict','current_step','rng_state','sampler_state','config_hash','manifest_hash'}<=m.RESUME_KEYS

def test_stage_checkpoints_are_isolated():
    assert len({m.stage_dir(x) for x in ('framewise','temporal_true','temporal_shuffled')})==3

def test_stable_norm_and_no_forbidden_runtime():
    source=inspect.getsource(m)
    assert 'stable_mamba_stack' in source and 'temporal_residual_forward' in source
    assert 'run_inference' not in source and 'mhr_to_smpl' not in source

def test_shuffled_is_deterministic():
    assert np.array_equal(m.temporal_permutation(16,7,'x'),m.temporal_permutation(16,7,'x'))

def test_frozen_train_statistics():
    cfg=m.protocol_config()
    assert cfg['tau_source']=='3DPW train clean only'
    assert cfg['validation_thresholds_source']=='frozen train thresholds'
    assert cfg['body_sha256']==m.EXPECTED_BODY_SHA
