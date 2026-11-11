from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from camera_evaluator_v1 import sequence_metrics, validate_identity_frame_mapping
from validate_camera_root_correspondence import (
    apply_surface_mapping,
    normalize_joint_regressor,
    regress_common_pelvis,
    world_to_camera,
    camera_to_world,
)

OUT = ROOT / 'outputs/dynamesh_v2_camera_phase2'
FEATURE_MANIFEST = ROOT / 'data/dynamesh_v2/cache_v2/manifest.json'
WINDOW_MANIFEST = OUT / 'camera_window_cache/manifest.json'
MAPPING = Path('/root/data1/MHR-main/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz')
REGRESSOR = Path('/root/DuoMo-main/data/body_models/J_regressor_h36m.npy')
SPLIT_ROOT = Path('/root/data1/sam-body4d-master/dataset/3DPW/sequenceFiles')
LEAKAGE_OUT = ROOT / 'CAMERA_DATA_LEAKAGE_AUDIT.json'
TESTS_OUT = ROOT / 'CAMERA_EVALUATOR_V1_TESTS.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def official_sequences(split):
    return sorted(p.stem for p in (SPLIT_ROOT / split).glob('*.pkl') if not p.name.startswith('._'))


def checkpoint_meta(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'path': str(path), 'sha256': sha256(path), 'manifest_hash': ckpt['manifest_hash'],
        'config_hash': ckpt['config_hash'], 'best_validation_state': ckpt['best_validation_state'],
        'epoch': int(ckpt['current_epoch']), 'step': int(ckpt['current_step']),
    }


def build_leakage_audit():
    feature = json.loads(FEATURE_MANIFEST.read_text())
    windows = json.loads(WINDOW_MANIFEST.read_text())
    official = {split: official_sequences(split) for split in ('train', 'validation', 'test')}
    cache_sequences = {
        split: sorted({e['sequence'] for e in windows[split]}) for split in ('train', 'validation')
    }
    feature_sequences = {
        split: sorted({e['sequence'] for e in feature['entries'] if e['split'] == split})
        for split in ('train', 'validation')
    }
    test = set(official['test'])
    intersections = {
        'optimization_train_vs_test': sorted(set(cache_sequences['train']) & test),
        'checkpoint_selection_validation_vs_test': sorted(set(cache_sequences['validation']) & test),
        'all_camera_cache_vs_test': sorted((set(cache_sequences['train']) | set(cache_sequences['validation'])) & test),
    }
    manifest_hash = sha256(FEATURE_MANIFEST)
    checkpoints = {
        'framewise': checkpoint_meta(OUT / 'framewise/best.pt'),
        'temporal_true': checkpoint_meta(OUT / 'temporal_true/best.pt'),
    }
    hashes_match = all(v['manifest_hash'] == manifest_hash for v in checkpoints.values())
    no_test = not any(intersections.values()) and feature.get('test_cache_files') == 0 and hashes_match
    audit = {
        'schema': 'dynamesh-camera-data-leakage-audit-v1',
        'verdict': 'NO_TEST_LEAKAGE' if no_test else 'TEST_LEAKAGE_RISK',
        'dataset': '3DPW',
        'training_contract': {
            'optimization_split': 'train',
            'checkpoint_selection_split': 'validation',
            'test_split': 'forbidden by loader guard',
            'framewise': 'gradients from train windows; validation used for best epoch and early stopping',
            'temporal_true': 'gradients from train windows with frozen framewise adapter; validation used for best epoch and early stopping',
        },
        'source_hashes': {
            'feature_manifest': manifest_hash,
            'window_manifest': sha256(WINDOW_MANIFEST),
            'training_script': sha256(ROOT / 'scripts/train_dynamesh_v2_camera_phase2.py'),
        },
        'checkpoints': checkpoints,
        'checkpoint_manifest_hash_match': hashes_match,
        'official_splits': official,
        'feature_cache_sequences': feature_sequences,
        'camera_window_cache_sequences': cache_sequences,
        'counts': {
            'train_sequences': len(cache_sequences['train']),
            'validation_sequences': len(cache_sequences['validation']),
            'train_tracks': len(windows['train']),
            'validation_tracks': len(windows['validation']),
            'train_windows': int(sum(e['windows'] for e in windows['train'])),
            'validation_windows': int(sum(e['windows'] for e in windows['validation'])),
            'feature_manifest_test_cache_files': int(feature.get('test_cache_files', -1)),
        },
        'test_intersections': intersections,
        'evidence': {
            'loader_guard': 'train_dynamesh_v2_camera_phase2.py:39-49 rejects test and binds optimization=train, validation=validation',
            'framewise_loop': 'train lines 175-187; validation lines 162-168',
            'temporal_loop': 'train lines 201-210; validation lines 194-200',
        },
    }
    LEAKAGE_OUT.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    return audit


def add_test(results, name, fn):
    try:
        observed = fn()
        results.append({'name': name, 'status': 'PASS', 'observed': observed})
    except Exception as error:
        results.append({'name': name, 'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'})


def build_tests():
    results = []
    mapping = np.load(MAPPING)
    triangle_ids = np.asarray(mapping['triangle_ids'], np.int64)
    bary = np.asarray(mapping['baryc_coords'], np.float64)
    reg_raw = np.asarray(np.load(REGRESSOR), np.float64)
    reg = normalize_joint_regressor(reg_raw)
    gate = np.load(OUT.parent / 'dynamesh_v2_camera_root_gate_50f/gate_input_50f.npz')
    vertices = np.asarray(gate['mhr_local_vertices'][:2], np.float64)
    from train_dynamesh_v2_camera_phase19 import load_frozen_model
    frozen_sam = load_frozen_model()
    faces = np.asarray(frozen_sam.head_pose.faces.detach().cpu(), np.int64)
    del frozen_sam

    def bary_test():
        error = float(np.max(np.abs(bary.sum(1) - 1)))
        assert error < 1e-12
        return {'rows': len(bary), 'max_abs_row_sum_error': error}

    def equivariance_test():
        mapped = apply_surface_mapping(vertices, faces, triangle_ids, bary)
        shift = np.array([.31, -.17, .42])
        moved = apply_surface_mapping(vertices + shift, faces, triangle_ids, bary)
        error = float(np.max(np.abs(moved - (mapped + shift))))
        assert error < 1e-12
        return {'max_abs_m': error}

    def gt_roundtrip_test():
        theta = .4; r = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        t = np.array([.2,-.3,1.7]); points = vertices[:, :20]
        restored = camera_to_world(world_to_camera(points, r, t), r, t)
        error = float(np.max(np.abs(restored - points))); assert error < 1e-12
        return {'max_abs_m': error, 'direction': 'world-camera-world'}

    def pred_roundtrip_test():
        theta = -.2; r = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        t = np.array([-.8,.1,2.]); points = vertices[:, :20]
        restored = world_to_camera(camera_to_world(points, r, t), r, t)
        error = float(np.max(np.abs(restored - points))); assert error < 1e-12
        return {'max_abs_m': error, 'direction': 'camera-world-camera'}

    def pelvis_test():
        mapped = apply_surface_mapping(vertices, faces, triangle_ids, bary)
        pelvis = regress_common_pelvis(mapped, reg)
        shift = np.array([.2,.4,-.1])
        moved = regress_common_pelvis(mapped + shift, reg)
        error = float(np.max(np.abs(moved - (pelvis + shift)))); assert error < 1e-12
        return {'max_abs_m': error, 'normalized_pelvis_row_sum': float(reg[0].sum())}

    def identity_test():
        keys = [('sequence', 0, i, i * 2) for i in range(8)]
        assert validate_identity_frame_mapping(keys, list(keys))
        rejected = False
        try: validate_identity_frame_mapping(keys + [keys[-1]], keys)
        except ValueError: rejected = True
        assert rejected
        return {'keys': len(keys), 'duplicate_rejected': rejected}

    def base_track():
        t=np.arange(6)[:,None,None]; x=np.zeros((6,14,3)); x[...,0:1]=t*.01
        x += np.arange(14)[None,:,None]*np.array([[[.001,.002,-.001]]])
        return x

    def perfect_test():
        x=base_track(); m=sequence_metrics(x,x,np.arange(6),fps=30.)
        maximum=max(m[k] for k in ('global_pelvis_error_mm','w_mpjpe_mm','wa_mpjpe_mm','person_root_ate_mm','rte_mm_per_frame','world_velocity_error_m_per_s','world_acceleration_error_m_per_s2','trajectory_jitter_m_per_s3'))
        assert maximum < 1e-9 and abs(m['motion_preservation_score']-1)<1e-12
        return {'max_error_metric': maximum, 'motion_preservation_score': m['motion_preservation_score']}

    def translation_test():
        x=base_track(); delta=np.array([.3,-.4,0.]); m=sequence_metrics(x+delta,x,np.arange(6),fps=30.)
        assert abs(m['global_pelvis_error_mm']-500)<1e-9 and m['wa_mpjpe_mm']<1e-9 and m['person_root_ate_mm']<1e-9
        return {k:m[k] for k in ('global_pelvis_error_mm','wa_mpjpe_mm','person_root_ate_mm','rte_mm_per_frame')}

    def velocity_test():
        x=base_track(); y=x.copy(); y[...,0]+=np.arange(6)[:,None]*.002; m=sequence_metrics(y,x,np.arange(6),fps=30.)
        assert abs(m['rte_mm_per_frame']-2)<1e-9 and abs(m['world_velocity_error_m_per_s']-.06)<1e-9 and m['world_acceleration_error_m_per_s2']<1e-9
        return {k:m[k] for k in ('rte_mm_per_frame','world_velocity_error_m_per_s','world_acceleration_error_m_per_s2','velocity_magnitude_ratio')}

    for name, fn in (
        ('barycentric_row_sum', bary_test), ('translation_equivariance', equivariance_test),
        ('gt_world_camera_world_round_trip', gt_roundtrip_test),
        ('prediction_camera_world_camera_round_trip', pred_roundtrip_test),
        ('pelvis_consistency', pelvis_test), ('identity_frame_mapping', identity_test),
        ('synthetic_perfect_prediction', perfect_test),
        ('fixed_translation_perturbation', translation_test),
        ('fixed_velocity_perturbation', velocity_test),
    ):
        add_test(results, name, fn)
    test_files = ['/tmp/test_camera_root_correspondence.py', '/tmp/test_camera_evaluator_v1.py']
    unit_runs = [subprocess.run(['/root/miniconda3/envs/duomo/bin/python', path], capture_output=True, text=True) for path in test_files]
    unit_returncode = max(run.returncode for run in unit_runs)
    unit_stdout = '\n'.join(run.stdout for run in unit_runs)
    unit_stderr = '\n'.join(run.stderr for run in unit_runs)
    all_pass = all(x['status'] == 'PASS' for x in results) and unit_returncode == 0
    report = {
        'schema': 'dynamesh-camera-evaluator-v1-tests',
        'verdict': 'ALL_PASS' if all_pass else 'FAIL',
        'tests': results,
        'python_unittest': {'status': 'PASS' if unit_returncode == 0 else 'FAIL', 'returncode': unit_returncode,
                            'stdout': unit_stdout, 'stderr': unit_stderr,
                            'test_files': test_files},
        'artifacts': {'surface_mapping_sha256': sha256(MAPPING), 'h36m_regressor_sha256': sha256(REGRESSOR),
                      'evaluator_sha256': sha256(ROOT/'scripts/camera_evaluator_v1.py')},
    }
    TESTS_OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not all_pass:
        raise RuntimeError('Evaluator v1 test gate failed')
    return report


if __name__ == '__main__':
    leakage = build_leakage_audit()
    tests = build_tests()
    print(json.dumps({'leakage': leakage['verdict'], 'tests': tests['verdict'],
                      'outputs': [str(LEAKAGE_OUT), str(TESTS_OUT)]}))
