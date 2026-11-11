"""Camera Phase-1.7 temporal supervision audit on existing data only."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/sam-body4d-0.2.0')
OUT = ROOT/'outputs/dynamesh_v2_camera_phase17_supervision_audit'
PHASE1 = ROOT/'outputs/dynamesh_v2_camera_phase1'
PHASE15 = ROOT/'outputs/dynamesh_v2_camera_phase15_oracle'
PHASE16 = ROOT/'outputs/dynamesh_v2_camera_phase16_loss_rebalancing'
ROOT_GATE = ROOT/'outputs/dynamesh_v2_camera_root_gate_50f'
DATA = Path('/root/data1/sam-body4d-master/dataset/3DPW')
H36M_REGRESSOR = Path('/root/DuoMo-main/data/body_models/J_regressor_h36m.npy')
SMPL_NEUTRAL = Path('/root/DuoMo-main/data/body_models/smpl/SMPL_NEUTRAL.pkl')
SEED = 20260901

sys.path[:0] = [str(ROOT), str(ROOT/'models/sam_3d_body'), str(ROOT/'scripts')]

from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from ablate_dynamesh_v2_camera_phase16 import (
    adapter_forward,
    condition_metrics,
    projection,
    sha256,
)
from train_dynamesh_v2_camera_phase1 import (
    CONDITIONS,
    assert_camera_optimizer_ownership,
    load_frozen_model,
    masked_normalized_reprojection_loss,
    prepare_window,
    reliability_features,
    set_seed,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def first_difference(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError('trajectory must be (T,3), T>=2')
    if not np.isfinite(values).all():
        raise ValueError('trajectory contains NaN/Inf')
    return np.diff(values, axis=0)


def _corr(x, y):
    x, y = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _error_statistics(source, target):
    signed = np.asarray(source)-np.asarray(target)
    error = np.linalg.norm(signed, axis=-1)
    return {'mean': float(error.mean()), 'median': float(np.median(error)), 'std': float(error.std()),
            'p95': float(np.percentile(error, 95)), 'max': float(error.max()),
            'xyz_mean_signed_error': signed.mean(0).tolist(), 'xyz_std': signed.std(0).tolist(),
            'per_frame': error.tolist()}


def relative_relation_statistics(source, target):
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('source/target must be aligned (T,3)')
    source_delta, target_delta = first_difference(source), first_difference(target)
    first = _error_statistics(source_delta, target_delta)
    source_mag, target_mag = np.linalg.norm(source_delta, axis=-1), np.linalg.norm(target_delta, axis=-1)
    valid = (source_mag > 1e-8) & (target_mag > 1e-8)
    cosine = np.sum(source_delta[valid]*target_delta[valid], axis=-1)/(source_mag[valid]*target_mag[valid]) if np.any(valid) else np.array([])
    threshold = float(np.percentile(target_mag, 25))
    low = target_mag <= threshold
    first.update({
        'direction_cosine_mean': float(cosine.mean()) if len(cosine) else None,
        'direction_cosine_median': float(np.median(cosine)) if len(cosine) else None,
        'trajectory_component_correlation': _corr(source_delta, target_delta),
        'magnitude_correlation': _corr(source_mag, target_mag),
        'target_motion_mean': float(target_mag.mean()),
        'error_over_target_motion': float(first['mean']/max(target_mag.mean(), 1e-12)),
        'low_motion_threshold': threshold,
        'low_motion_mean_error': float(np.linalg.norm(source_delta[low]-target_delta[low], axis=-1).mean()) if np.any(low) else None,
        'non_low_motion_mean_error': float(np.linalg.norm(source_delta[~low]-target_delta[~low], axis=-1).mean()) if np.any(~low) else None,
    })
    second_source, second_target = first_difference(source_delta), first_difference(target_delta)
    return {'absolute': _error_statistics(source, target), 'first_difference': first,
            'second_difference': _error_statistics(second_source, second_target)}


def temporal_capacity_verdict(window_gain, failure_gain):
    return 'TEMPORAL_MAMBA_CAPACITY_CONFIRMED' if window_gain >= .20 and failure_gain >= .10 else 'TEMPORAL_MAPPING_CAPACITY_LIMITED'


def route_verdict(capacity, relative, camera_observable_supervision=True):
    if capacity == 'TEMPORAL_MAMBA_CAPACITY_CONFIRMED':
        if relative != 'RELATIVE_CAMERA_SUPERVISION_NOT_RELIABLE':
            return 'ROUTE_KEEP_LATENT_MAMBA_UPDATE_SUPERVISION'
        return 'ROUTE_KEEP_LATENT_MAMBA_REPROJ_ONLY_ISSUE'
    if relative == 'RELATIVE_CAMERA_SUPERVISION_NOT_RELIABLE' and not camera_observable_supervision:
        return 'ROUTE_DROP_CAMERA_BRANCH'
    return 'ROUTE_REDESIGN_CAMERA_MAPPING'


def reprojection_only_forward(adapter, model, tokens, meta, raw, condition):
    features = reliability_features(tokens[condition][None], meta, (0., .15, .35, .55)[condition])
    refined, reliability = adapter_forward(adapter, tokens[condition][None], features, 'fixed_open')
    cam = model.head_camera(refined.squeeze(0), model.init_camera.weight.expand(16, -1))
    projected = projection(model, raw[condition]['points3d'].detach(), cam, meta)
    gt = torch.as_tensor(meta['gt_xyc'], device='cuda')
    bbox = torch.as_tensor(meta['size'], device='cuda')
    loss = masked_normalized_reprojection_loss(projected['pred_keypoints_2d'], gt, bbox)
    return loss, {'pred_cam': cam, 'pred_cam_t': projected['pred_cam_t'],
                  'pred_xy': projected['pred_keypoints_2d'], 'reliability': reliability}


@torch.no_grad()
def evaluate_probe(adapter, model, tokens, meta, raw, failure_index):
    adapter.eval(); result = {}
    for condition, name in enumerate(CONDITIONS):
        _, out = reprojection_only_forward(adapter, model, tokens, meta, raw, condition)
        result[name] = condition_metrics(out, raw[condition], meta, failure_index)
    return result


def run_reprojection_probe(model, tokens, meta, raw, failure_index, requested_steps=400, lr=1e-3):
    set_seed(); adapter = CameraFeatureTemporalAdapter().cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    assert_camera_optimizer_ownership(optimizer, adapter)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError('frozen SAM unexpectedly trainable')
    directory = OUT/'reprojection_only'; directory.mkdir(parents=True, exist_ok=True)
    history, best, max_steps = [], math.inf, requested_steps
    for step in range(1, 801):
        adapter.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = reprojection_only_forward(adapter, model, tokens, meta, raw, 0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite clean reprojection at step {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
        value = float(loss.detach()); history.append(value)
        if value < best:
            best = value
            torch.save({'adapter_state_dict': adapter.state_dict(), 'best_step': step,
                        'best_reprojection': best, 'lr': lr}, directory/'best.pt')
        if step % 20 == 0:
            atomic_json(OUT/'status.json', {'state': 'running', 'stage': 'reprojection_only', 'step': step,
                        'planned_steps': max_steps, 'pid': os.getpid()})
            print(json.dumps({'stage': 'reprojection_only', 'step': step, 'loss': value, 'best': best}), flush=True)
        if step == requested_steps:
            previous = float(np.mean(history[-100:-50])); recent = float(np.mean(history[-50:]))
            material_decrease = (previous-recent)/max(previous, 1e-12) > .02
            max_steps = 800 if material_decrease else requested_steps
            if not material_decrease:
                break
        elif step >= max_steps:
            break
    torch.save({'adapter_state_dict': adapter.state_dict(), 'final_step': len(history), 'lr': lr}, directory/'final.pt')
    checkpoint = torch.load(directory/'best.pt', map_location='cuda', weights_only=False)
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    metrics = evaluate_probe(adapter, model, tokens, meta, raw, failure_index)
    clean = metrics['clean']
    verdict = temporal_capacity_verdict(clean['window_gain'], clean['failure_gain'])
    report = {'setting': {'condition_trained': 'clean only', 'gate': 'fixed_open', 'loss': '12-joint L_reproj only',
                          'optimizer': 'AdamW', 'lr': lr, 'requested_steps': requested_steps,
                          'actual_steps': len(history), 'seed': SEED},
              'best_step': checkpoint['best_step'], 'best_reprojection_loss': checkpoint['best_reprojection'],
              'history': history, 'metrics': metrics, 'verdict': verdict}
    atomic_json(OUT/'reprojection_only_probe.json', report)
    return report


def _load_pickle(path):
    with Path(path).open('rb') as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0); return pickle.load(handle, encoding='latin1')


def _native_regressor():
    model = _load_pickle(SMPL_NEUTRAL)
    regressor = model['J_regressor']
    return np.asarray(regressor.toarray() if hasattr(regressor, 'toarray') else regressor, dtype=np.float64)


def relative_verdict(primary, pose_correlation):
    first = primary['first_difference']; absolute = primary['absolute']
    reduction = first['mean']/max(absolute['mean'], 1e-12)
    cosine = first['direction_cosine_mean'] if first['direction_cosine_mean'] is not None else -1
    magnitude = first['magnitude_correlation'] if first['magnitude_correlation'] is not None else -1
    if first['error_over_target_motion'] < .5 and cosine > .8 and magnitude > .8 and abs(pose_correlation or 0) < .3:
        return 'RELATIVE_CAMERA_SUPERVISION_VALIDATED'
    if reduction < .5 and first['error_over_target_motion'] < 1.5 and cosine > .5 and magnitude > .5:
        return 'RELATIVE_CAMERA_SUPERVISION_USABLE_AS_AUXILIARY'
    return 'RELATIVE_CAMERA_SUPERVISION_NOT_RELIABLE'


def run_relative_gate():
    with np.load(ROOT_GATE/'gate_input_50f.npz', allow_pickle=False) as data:
        gate = {name: np.asarray(data[name]) for name in data.files}
    raw = _load_pickle(ROOT/'outputs/3dpw_protocol_isolation_50f.pkl')
    annotation = _load_pickle(DATA/'sequenceFiles/test/flat_guitar_01.pkl')
    frame_ids = np.asarray(gate['frame_id'], dtype=np.int64)
    original = np.asarray(annotation['img_frame_ids'], dtype=np.int64)
    lookup = {int(frame): index for index, frame in enumerate(original)}
    local = np.asarray([lookup[int(frame)] for frame in frame_ids], dtype=np.int64)
    transl_world = np.asarray(annotation['trans'], dtype=np.float64)[0, local]
    cameras = np.asarray(annotation['cam_poses'], dtype=np.float64)[local]
    gt_origin_camera = np.einsum('nij,nj->ni', cameras[:, :3, :3], transl_world)+cameras[:, :3, 3]
    pred_cam_t = np.asarray(gate['pred_cam_t'], dtype=np.float64)
    mhr_root = np.asarray(gate['mhr_local_root'], dtype=np.float64)+pred_cam_t
    meta_vertices = np.asarray(gate['meta_smpl_vertices_local'], dtype=np.float64)
    gt_vertices = np.asarray(gate['gt_smpl_vertices_camera'], dtype=np.float64)
    h36m = np.asarray(np.load(H36M_REGRESSOR), dtype=np.float64)
    native = _native_regressor()
    meta_h36m = np.einsum('v,tvc->tc', h36m[0], meta_vertices)+pred_cam_t
    gt_h36m = np.einsum('v,tvc->tc', h36m[0], gt_vertices)
    meta_native = np.einsum('v,tvc->tc', native[0], meta_vertices)+pred_cam_t
    gt_native = np.einsum('v,tvc->tc', native[0], gt_vertices)
    candidates = {
        'model_origin_pred_cam_t_vs_gt_Rtrans_plus_t': relative_relation_statistics(pred_cam_t, gt_origin_camera),
        'mhr_hip_midpoint_vs_gt_h36m_pelvis_diagnostic': relative_relation_statistics(mhr_root, gt_h36m),
        'meta_smpl_h36m_pelvis_vs_gt_h36m_pelvis': relative_relation_statistics(meta_h36m, gt_h36m),
        'meta_smpl_native_pelvis_vs_gt_native_pelvis': relative_relation_statistics(meta_native, gt_native),
        'representation_mapping_mhr_hip_vs_meta_h36m': relative_relation_statistics(mhr_root, meta_h36m),
    }
    body = np.stack([np.asarray(record['body_pose_params'], dtype=np.float64) for record in raw['dyna']])
    pose_motion = np.linalg.norm(np.diff(body, axis=0), axis=1)
    primary_error = np.asarray(candidates['model_origin_pred_cam_t_vs_gt_Rtrans_plus_t']['first_difference']['per_frame'])
    pose_correlation = _corr(primary_error, pose_motion)
    motion_correlation = _corr(primary_error, np.linalg.norm(np.diff(gt_origin_camera, axis=0), axis=1))
    previous = json.loads((ROOT_GATE/'summary.json').read_text())
    verdict = relative_verdict(candidates['model_origin_pred_cam_t_vs_gt_Rtrans_plus_t'], pose_correlation)
    report = {
        'sequence': 'flat_guitar_01', 'person_id': 0, 'num_frames': 50, 'frame_ids': frame_ids.tolist(),
        'definitions': {
            'primary_predicted': 'SAM/MHR model-origin camera placement = pred_cam_t, metres, SAM camera axes',
            'primary_gt': '3DPW SMPL model-origin placement = cam_pose[:3,:3] @ trans + cam_pose[:3,3], metres, camera axes',
            'h36m_candidate': 'diagnostic evaluation pelvis only; not treated as model translation root',
            'native_candidate': 'SMPL native J_regressor joint 0 applied to existing Meta/GT vertices',
            'fitted_transform': 'none',
        },
        'previous_absolute_gate': previous,
        'candidates': candidates,
        'primary_dependence': {'first_difference_error_vs_body_pose_change_correlation': pose_correlation,
                               'first_difference_error_vs_gt_motion_magnitude_correlation': motion_correlation},
        'verdict': verdict,
        'candidate_loss': None if verdict == 'RELATIVE_CAMERA_SUPERVISION_NOT_RELIABLE' else
            'L_relative_camera = SmoothL1(delta(pred_cam_t), delta(R_cam @ trans + t_cam)); auxiliary only until multi-sequence validation',
    }
    atomic_json(OUT/'relative_camera_gate.json', report)
    return report


def run(steps=400, lr=1e-3):
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    required = [PHASE1/'mining.json', PHASE15/'summary.json', PHASE16/'summary.json', ROOT_GATE/'gate_input_50f.npz']
    if any(not path.exists() for path in required):
        raise FileNotFoundError('required completed prior artifact missing')
    atomic_json(OUT/'status.json', {'state': 'initializing', 'pid': os.getpid()})
    mining = json.loads((PHASE1/'mining.json').read_text())
    model = load_frozen_model(); _, meta, tokens, _, raw, _ = prepare_window(model, mining)
    expected = list(range(506, 538, 2))
    if meta['sequence'] != 'courtyard_basketball_00' or meta['person'] != 1 or meta['frame_id'].astype(int).tolist() != expected:
        raise RuntimeError('fixed Probe-1 window mismatch')
    failure_index = int(np.where(meta['frame_id'] == 522)[0][0])
    body_checkpoint = ROOT/'outputs/dynamesh_v2_phase2_training/best.pt'; body_hash = sha256(body_checkpoint)
    probe = run_reprojection_probe(model, tokens, meta, raw, failure_index, steps, lr)
    atomic_json(OUT/'status.json', {'state': 'running', 'stage': 'relative_gate', 'pid': os.getpid()})
    relative = run_relative_gate()
    route = route_verdict(probe['verdict'], relative['verdict'], camera_observable_supervision=True)
    summary = {'state': 'complete', 'elapsed_seconds': time.time()-started, 'reprojection_probe': probe,
               'relative_gate': relative, 'route': route, 'body_checkpoint_sha256': body_hash,
               'prior_references': {'phase15': str(PHASE15/'summary.json'), 'phase16': str(PHASE16/'summary.json')}}
    if sha256(body_checkpoint) != body_hash:
        raise RuntimeError('Body checkpoint changed')
    atomic_json(OUT/'summary.json', summary)
    atomic_json(OUT/'status.json', {'state': 'complete', 'pid': os.getpid(), 'elapsed_seconds': summary['elapsed_seconds'],
                'temporal_verdict': probe['verdict'], 'relative_verdict': relative['verdict'], 'route': route})
    print(json.dumps({'state': 'complete', 'temporal_verdict': probe['verdict'],
                      'relative_verdict': relative['verdict'], 'route': route,
                      'elapsed_seconds': summary['elapsed_seconds']}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--steps', type=int, default=400); parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args(); set_seed(); run(args.steps, args.lr)


if __name__ == '__main__':
    main()
