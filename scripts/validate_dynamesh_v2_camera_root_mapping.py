"""Numerical gate for the explicit Camera-v0 common-root mapping.

This script consumes precomputed small arrays only.  It never runs SAM, Meta
Conversion, or an evaluator.  All coordinates must be metres; no empirical
offset or scale is accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/sam-body4d-0.2.0')
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/sam_3d_body'))

from sam_3d_body.models.modules.camera_root_geometry import (
    compare_common_roots,
    get_3dpw_gt_camera_root,
    get_pred_camera_root,
)


def _finite(name, value):
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return array


def evaluate_root_mapping(
    mhr_local_root,
    pred_cam_t,
    converted_smpl_vertices_local,
    gt_smpl_vertices_world,
    joint_regressor,
    root_index,
    cam_poses,
    unit='meters',
):
    if unit != 'meters':
        raise ValueError('root mapping gate accepts metres only')
    mhr_local_root = _finite('mhr_local_root', mhr_local_root)
    pred_cam_t = _finite('pred_cam_t', pred_cam_t)
    converted = _finite('converted_smpl_vertices_local', converted_smpl_vertices_local)
    gt_world = _finite('gt_smpl_vertices_world', gt_smpl_vertices_world)
    regressor = _finite('joint_regressor', joint_regressor)
    cameras = _finite('cam_poses', cam_poses)
    frames = mhr_local_root.shape[0]
    if mhr_local_root.shape != (frames, 3) or pred_cam_t.shape != (frames, 3):
        raise ValueError('roots and pred_cam_t must be aligned (T,3)')
    if converted.ndim != 3 or gt_world.ndim != 3 or converted.shape[0] != frames or gt_world.shape[0] != frames:
        raise ValueError('SMPL vertices must be aligned (T,V,3)')
    if cameras.shape != (frames, 4, 4):
        raise ValueError('cam_poses must be aligned (T,4,4)')
    if regressor.ndim != 2 or root_index < 0 or root_index >= regressor.shape[0]:
        raise ValueError('invalid joint regressor/root_index')
    if converted.shape[1] != regressor.shape[1] or gt_world.shape[1] != regressor.shape[1]:
        raise ValueError('joint regressor vertex topology mismatch')

    predicted = get_pred_camera_root(mhr_local_root, pred_cam_t, unit=unit)
    gt_camera = get_3dpw_gt_camera_root(gt_world, regressor, root_index, cameras, unit=unit)
    error = compare_common_roots(predicted, gt_camera, unit=unit)
    per_frame_error = np.asarray(error['per_frame'])
    offset = predicted - gt_camera

    converted_root = np.einsum('v,tvc->tc', regressor[root_index], converted)
    local_delta = converted_root - mhr_local_root
    local_delta_centered = local_delta - local_delta.mean(axis=0, keepdims=True)
    body_extent = np.linalg.norm(converted.max(axis=1) - converted.min(axis=1), axis=-1)
    delta_norm = np.linalg.norm(local_delta_centered, axis=-1)
    correlation = 0.0
    if frames > 1 and np.std(body_extent) > 0 and np.std(delta_norm) > 0:
        correlation = float(np.corrcoef(body_extent, delta_norm)[0, 1])

    return {
        'unit': unit,
        'predicted_camera_root': predicted,
        'gt_camera_root': gt_camera,
        'per_frame_error': per_frame_error,
        'error': {
            'mean': float(per_frame_error.mean()),
            'std': float(per_frame_error.std()),
            'max': float(per_frame_error.max()),
        },
        'systematic_offset': offset.mean(axis=0),
        'local_mapping': {
            'mean_offset': local_delta.mean(axis=0),
            'std_xyz': local_delta.std(axis=0),
            'std_norm': float(np.linalg.norm(local_delta.std(axis=0))),
            'extent_delta_correlation': correlation,
        },
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='NPZ with precomputed root/SMPL arrays')
    parser.add_argument('--output', required=True, help='JSON report path')
    parser.add_argument('--root-index', required=True, type=int)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        report = evaluate_root_mapping(
            data['mhr_local_root'], data['pred_cam_t'],
            data['converted_smpl_vertices_local'], data['gt_smpl_vertices_world'],
            data['joint_regressor'], args.root_index, data['cam_poses'],
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + '.tmp')
    temp.write_text(json.dumps(_jsonable(report), indent=2), encoding='utf-8')
    temp.replace(output)


if __name__ == '__main__':
    main()
