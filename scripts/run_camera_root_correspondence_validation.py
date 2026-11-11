from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import smplx

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'models/sam_3d_body')]

import train_dynamesh_v2_camera_phase2 as p2
import train_dynamesh_v2_camera_phase110 as p
import ablate_dynamesh_v2_camera_phase111 as q
import train_dynamesh_v2_camera_phase112 as phase112
from evaluate_dynamesh_v2_camera_phase2 import load_module
from train_dynamesh_v2_camera_phase19 import load_frozen_model
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from validate_camera_root_correspondence import (
    apply_surface_mapping,
    normalize_joint_regressor,
    regress_common_pelvis,
    world_to_camera,
    camera_to_world,
)

DATA = Path('/root/data1/sam-body4d-master/dataset/3DPW')
MAPPING = Path('/root/data1/MHR-main/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz')
REGRESSOR = Path('/root/DuoMo-main/data/body_models/J_regressor_h36m.npy')
SMPL_DIR = Path('/root/DuoMo-main/data/body_models/smpl')
VALIDATION_OUT = ROOT / 'CAMERA_ROOT_CORRESPONDENCE_VALIDATION_20261002.json'
WORLD_OUT = ROOT / 'CAMERA_WORLD_SMOKE_20261002.json'
SEQUENCES = {
    'courtyard_arguing_00': 'train',
    'courtyard_relaxOnBench_00': 'train',
    'courtyard_jumpBench_01': 'validation',
}
METHODS = ('raw', 'framewise', 'full')
CONDITIONS = tuple(p2.CONDITIONS)
FPS = 30.0
J14 = np.asarray([6, 5, 4, 1, 2, 3, 13, 12, 11, 14, 15, 16, 0, 7], dtype=np.int64)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return {'n': int(len(x)), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'std': float(x.std()), 'p95': float(np.percentile(x, 95)), 'max': float(x.max())}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def select_central_windows():
    manifest = json.loads((p2.WINDOW_CACHE / 'manifest.json').read_text())
    selected = []
    for split in ('train', 'validation'):
        for entry in manifest[split]:
            if entry['sequence'] not in SEQUENCES:
                continue
            if SEQUENCES[entry['sequence']] != split:
                raise ValueError('split mismatch')
            windows = torch.load(entry['path'], map_location='cpu', weights_only=False)
            if not windows:
                raise ValueError(f'empty cache {entry["path"]}')
            selected.append((split, entry, windows[len(windows) // 2]))
    identities = {(x[1]['sequence'], int(x[1]['person_id'])) for x in selected}
    expected = {('courtyard_arguing_00', 0), ('courtyard_arguing_00', 1),
                ('courtyard_relaxOnBench_00', 0), ('courtyard_jumpBench_01', 0)}
    if identities != expected:
        raise ValueError(f'unexpected identities: {identities}')
    return selected


def move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def load_gt(split, sequence, person, selected, device, models):
    with (DATA / 'sequenceFiles' / split / f'{sequence}.pkl').open('rb') as f:
        ann = pickle.load(f, encoding='latin1')
    gender = str(ann['genders'][person]).lower()
    key = 'female' if gender.startswith('f') else 'male'
    if key not in models:
        models[key] = smplx.SMPL(model_path=str(SMPL_DIR / f'SMPL_{key.upper()}.pkl'), batch_size=len(selected)).to(device).eval()
    pose = torch.as_tensor(np.asarray(ann['poses'][person])[selected], device=device, dtype=torch.float32)
    betas = torch.as_tensor(np.broadcast_to(np.asarray(ann['betas'][person]), (len(selected), 10)).copy(), device=device, dtype=torch.float32)
    transl = torch.as_tensor(np.asarray(ann['trans'][person])[selected], device=device, dtype=torch.float32)
    with torch.no_grad():
        out = models[key](betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:72], transl=transl)
    vertices = out.vertices.detach().cpu().numpy().astype(np.float64)
    native_pelvis = out.joints[:, 0].detach().cpu().numpy().astype(np.float64)
    camera = np.asarray(ann['cam_poses'], dtype=np.float64)[selected]
    return vertices, native_pelvis, transl.detach().cpu().numpy().astype(np.float64), camera, key


def camera_output(method, models, sample, condition):
    if method == 'raw':
        return sample['raw'][condition]
    if method == 'framewise':
        return phase112.framewise_forward(models['framewise'], models['sam'], sample, condition)[1]
    return phase112.temporal_forward(models['temporal'], models['framewise'], models['sam'], sample, condition)


def metric_bundle(pred_joints_world, gt_joints_world, pred_pelvis_world, gt_pelvis_world,
                  translation_error, reprojection):
    pelvis_error = np.linalg.norm(pred_pelvis_world - gt_pelvis_world, axis=-1)
    joint_error = np.linalg.norm(pred_joints_world - gt_joints_world, axis=-1)
    first_offset = pred_pelvis_world[:1] - gt_pelvis_world[:1]
    aligned = pred_joints_world - first_offset[:, None, :]
    aligned_error = np.linalg.norm(aligned - gt_joints_world, axis=-1)
    dp, dg = np.diff(pred_pelvis_world, axis=0), np.diff(gt_pelvis_world, axis=0)
    ddp, ddg = np.diff(pred_pelvis_world, n=2, axis=0), np.diff(gt_pelvis_world, n=2, axis=0)
    jerk = np.diff(pred_pelvis_world, n=3, axis=0)
    speed_p, speed_g = np.linalg.norm(dp, axis=-1), np.linalg.norm(dg, axis=-1)
    return {
        'global_pelvis_position_error_m': pelvis_error.tolist(),
        'w_mpjpe_m': joint_error.mean(axis=1).tolist(),
        'wa_mpjpe_first_frame_translation_aligned_m': aligned_error.mean(axis=1).tolist(),
        'person_root_ate_rmse_m': [float(np.sqrt(np.mean(pelvis_error ** 2)))],
        'rte_m_per_frame': np.linalg.norm(dp - dg, axis=-1).tolist(),
        'world_velocity_error_m_per_s': (np.linalg.norm(dp - dg, axis=-1) * FPS).tolist(),
        'world_acceleration_error_m_per_s2': (np.linalg.norm(ddp - ddg, axis=-1) * FPS ** 2).tolist(),
        'trajectory_jitter_pred_jerk_m_per_s3': (np.linalg.norm(jerk, axis=-1) * FPS ** 3).tolist(),
        'pred_speed_m_per_frame': speed_p.tolist(),
        'gt_speed_m_per_frame': speed_g.tolist(),
        'camera_translation_error_m': np.asarray(translation_error, float).tolist(),
        'camera_reprojection_error_normalized': np.asarray(reprojection, float).tolist(),
    }


def aggregate(bundles):
    keys = tuple(bundles[0])
    flat = {k: [v for b in bundles for v in b[k]] for k in keys}
    out = {k: stats(v) for k, v in flat.items() if k not in ('pred_speed_m_per_frame', 'gt_speed_m_per_frame')}
    pred = np.asarray(flat['pred_speed_m_per_frame'], float)
    gt = np.asarray(flat['gt_speed_m_per_frame'], float)
    out['motion_magnitude_preservation_ratio'] = float(pred.mean() / max(gt.mean(), 1e-12))
    out['windows'] = len(bundles)
    return out


def main():
    started = time.time()
    device = torch.device('cuda')
    mapping = np.load(MAPPING)
    triangle_ids = np.asarray(mapping['triangle_ids'], np.int64)
    bary = np.asarray(mapping['baryc_coords'], np.float64)
    regressor_raw = np.asarray(np.load(REGRESSOR), np.float64)
    regressor = normalize_joint_regressor(regressor_raw)
    if triangle_ids.shape != (6890,) or bary.shape != (6890, 3) or regressor.shape[1] != 6890:
        raise ValueError('unexpected official mapping/regressor topology')
    if not np.allclose(bary.sum(1), 1., atol=1e-9) or not np.allclose(regressor.sum(1), 1., atol=1e-12):
        raise ValueError('affine invariance prerequisites failed')

    selected_windows = select_central_windows()
    models = {
        'sam': load_frozen_model(),
        'framewise': load_module(q.FramewiseLatentAdapter, p2.OUT / 'framewise/best.pt'),
        'temporal': load_module(CameraFeatureTemporalAdapter, p2.OUT / 'temporal_true/best.pt'),
    }
    smpl_models = {}
    validation_rows = []
    world = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pred_roundtrip, gt_roundtrip, equivariance = [], [], []
    split_records = []

    with torch.no_grad():
        for split, entry, sample in selected_windows:
            sample = move_tensors(sample, device)
            seq, person = entry['sequence'], int(entry['person_id'])
            local = np.asarray(sample['meta']['selected'], np.int64)
            original = np.load(sample['desc']['cache_path'], allow_pickle=False)
            gt_vertices, native_pelvis, gt_transl, cam_pose, gender = load_gt(split, seq, person, local, device, smpl_models)
            gt_h36m = np.einsum('jv,nvc->njc', regressor, gt_vertices)
            gt_pelvis = gt_h36m[:, 0]
            gt_camera = world_to_camera(gt_vertices, cam_pose[:, :3, :3], cam_pose[:, :3, 3])
            restored_gt = camera_to_world(gt_camera, cam_pose[:, :3, :3], cam_pose[:, :3, 3])
            gt_roundtrip.append(float(np.max(np.abs(restored_gt - gt_vertices))))
            split_records.append({'sequence': seq, 'person_id': person, 'split': split, 'gender_model': gender,
                                  'frame_ids': np.asarray(sample['meta']['frame_id'], int).tolist(),
                                  'local_indices': local.tolist()})

            for c, condition in enumerate(CONDITIONS):
                tokens = sample['tokens'][c].to(device=device, dtype=torch.float32)
                init = torch.as_tensor(original['mhr_init_pose'][c, local], device=device, dtype=torch.float32)
                pose = models['sam'].head_pose(tokens, init)
                vertices = pose['pred_vertices'].detach().cpu().numpy().astype(np.float64)
                keypoints = pose['pred_keypoints_3d'].detach().cpu().numpy().astype(np.float64)
                faces = np.asarray(pose['faces'], np.int64)
                mapped = apply_surface_mapping(vertices, faces, triangle_ids, bary)
                pred_h36m_local = np.einsum('jv,nvc->njc', regressor, mapped)
                common_pelvis = pred_h36m_local[:, 0]
                hip_midpoint = .5 * (keypoints[:, 9] + keypoints[:, 10])
                pose_mag = np.sqrt(np.mean((mapped - mapped[:1]) ** 2, axis=(1, 2)))
                shift = np.array([.37, -.21, 1.13])
                mapped_shift = apply_surface_mapping(vertices + shift, faces, triangle_ids, bary)
                pelvis_shift = regress_common_pelvis(mapped_shift, regressor)
                equivariance.append(float(np.max(np.abs(pelvis_shift - (common_pelvis + shift)))))
                for i in range(len(local)):
                    validation_rows.append({
                        'sequence': seq, 'person_id': person, 'split': split, 'condition': condition,
                        'frame_id': int(sample['meta']['frame_id'][i]),
                        'origin_to_common_pelvis_m': float(np.linalg.norm(common_pelvis[i])),
                        'hip_midpoint_to_common_pelvis_m': float(np.linalg.norm(hip_midpoint[i] - common_pelvis[i])),
                        'pose_magnitude_m': float(pose_mag[i]),
                        'gt_transl_to_native_smpl_pelvis_m': float(np.linalg.norm(native_pelvis[i] - gt_transl[i])),
                        'native_smpl_to_h36m_pelvis_m': float(np.linalg.norm(native_pelvis[i] - gt_pelvis[i])),
                    })

                for method in METHODS:
                    output = camera_output(method, models, sample, c)
                    cam_t = output['pred_cam_t'].detach().cpu().numpy().astype(np.float64)
                    pred_cam_joints = pred_h36m_local + cam_t[:, None, :]
                    pred_world_joints = camera_to_world(pred_cam_joints, cam_pose[:, :3, :3], cam_pose[:, :3, 3])
                    pred_cam_pelvis = common_pelvis + cam_t
                    pred_world_pelvis = camera_to_world(pred_cam_pelvis[:, None, :], cam_pose[:, :3, :3], cam_pose[:, :3, 3])[:, 0]
                    closed = world_to_camera(pred_world_pelvis[:, None, :], cam_pose[:, :3, :3], cam_pose[:, :3, 3])[:, 0]
                    pred_roundtrip.append(float(np.max(np.abs(closed - pred_cam_pelvis))))
                    reproj = p.frame_reprojection(output['pred_xy'].detach().cpu(), sample['meta']['gt_xyc'], sample['meta']['size'])
                    trans_error = np.linalg.norm(cam_t - np.asarray(sample['target'][c]['t_all']), axis=1)
                    bundle = metric_bundle(pred_world_joints[:, J14], gt_h36m[:, J14], pred_world_pelvis,
                                           gt_pelvis, trans_error, reproj)
                    world[seq][condition][method].append(bundle)

    offsets = np.asarray([r['origin_to_common_pelvis_m'] for r in validation_rows])
    hips = np.asarray([r['hip_midpoint_to_common_pelvis_m'] for r in validation_rows])
    pose = np.asarray([r['pose_magnitude_m'] for r in validation_rows])
    native_transl = np.asarray([r['gt_transl_to_native_smpl_pelvis_m'] for r in validation_rows])
    native_h36m = np.asarray([r['native_smpl_to_h36m_pelvis_m'] for r in validation_rows])
    gate_checks = {
        'official_mapping_topology': bool(triangle_ids.shape == (6890,) and bary.shape == (6890, 3)),
        'barycentric_rows_sum_to_one_max_abs': float(np.max(np.abs(bary.sum(1) - 1))),
        'h36m_pelvis_regressor_raw_sum': float(regressor_raw[0].sum()),
        'h36m_regressor_raw_max_row_sum_deviation': float(np.max(np.abs(regressor_raw.sum(1) - 1))),
        'h36m_pelvis_regressor_normalized_sum': float(regressor[0].sum()),
        'h36m_regressor_normalization_max_abs_weight_change': float(np.max(np.abs(regressor - regressor_raw))),
        'translation_equivariance_max_abs_m': max(equivariance),
        'gt_world_camera_world_roundtrip_max_abs_m': max(gt_roundtrip),
        'prediction_camera_world_camera_roundtrip_max_abs_m': max(pred_roundtrip),
        'test_split_used': any(r['split'] == 'test' for r in split_records),
    }
    reliable = (gate_checks['barycentric_rows_sum_to_one_max_abs'] < 1e-9 and
                abs(gate_checks['h36m_pelvis_regressor_normalized_sum'] - 1) < 1e-12 and
                gate_checks['translation_equivariance_max_abs_m'] < 1e-9 and
                gate_checks['gt_world_camera_world_roundtrip_max_abs_m'] < 1e-9 and
                gate_checks['prediction_camera_world_camera_roundtrip_max_abs_m'] < 1e-9 and
                not gate_checks['test_split_used'])
    validation = {
        'schema': 'dynamesh-camera-root-correspondence-v1',
        'verdict': 'ROOT_MAPPING_RELIABLE' if reliable else 'ROOT_MAPPING_NOT_RELIABLE',
        'common_geometry_mapping': 'COMMON_GEOMETRY_MAPPING_AVAILABLE' if reliable else 'NOT_AVAILABLE',
        'definition': {
            'mhr_local_origin': 'MHR rig/model coordinate [0,0,0] with head_pose global_trans forced to zero; not an anatomical pelvis token',
            'pred_cam_t': 'translation added uniformly to all local MHR points after SAM camera-axis conversion',
            'prediction_pelvis': 'row-normalized J_regressor_h36m[0] applied to official barycentric MHR-to-SMPL-6890 surface mapping',
            'gt_pelvis': 'the same row-normalized J_regressor_h36m[0] applied to reconstructed 3DPW SMPL-6890 vertices',
            'fixed_offset_used': False,
            'fitting_or_metric_selection_used': False,
        },
        'source_assets': {
            'surface_mapping': str(MAPPING), 'surface_mapping_sha256': sha256(MAPPING),
            'h36m_regressor': str(REGRESSOR), 'h36m_regressor_sha256': sha256(REGRESSOR),
            'framewise_checkpoint_sha256': sha256(p2.OUT / 'framewise/best.pt'),
            'temporal_checkpoint_sha256': sha256(p2.OUT / 'temporal_true/best.pt'),
        },
        'sample_scope': {'windows': len(selected_windows), 'frames_x_conditions': len(validation_rows),
                         'splits': sorted({r['split'] for r in split_records}), 'records': split_records},
        'gate_checks': gate_checks,
        'offset_diagnostics': {
            'origin_to_common_pelvis_m': stats(offsets),
            'hip_midpoint_to_common_pelvis_m': stats(hips),
            'gt_transl_to_native_smpl_pelvis_m': stats(native_transl),
            'native_smpl_to_h36m_pelvis_m': stats(native_h36m),
            'hip_common_offset_vs_pose_magnitude_correlation': corr(hips, pose),
            'interpretation': 'offset is pose/shape/orientation dependent and is not used; pelvis is recomputed from geometry per frame',
        },
        'elapsed_seconds': time.time() - started,
    }
    VALIDATION_OUT.write_text(json.dumps(validation, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'validation': str(VALIDATION_OUT), 'verdict': validation['verdict']}), flush=True)
    if not reliable:
        if WORLD_OUT.exists():
            raise RuntimeError('stale world smoke exists while gate failed')
        return

    results = {}
    all_bundles = defaultdict(lambda: defaultdict(list))
    for sequence in SEQUENCES:
        results[sequence] = {'split': SEQUENCES[sequence], 'conditions': {}}
        for condition in CONDITIONS:
            results[sequence]['conditions'][condition] = {}
            for method in METHODS:
                bundles = world[sequence][condition][method]
                results[sequence]['conditions'][condition][method] = aggregate(bundles)
                all_bundles[condition][method].extend(bundles)
    world_result = {
        'schema': 'dynamesh-camera-world-smoke-v1',
        'status': 'SMALL_DEVELOPMENT_SMOKE_ONLY_NOT_FULL_BENCHMARK',
        'root_mapping_verdict': validation['verdict'],
        'protocol': {
            'sequences': SEQUENCES, 'methods': METHODS, 'conditions': CONDITIONS,
            'selection': 'one deterministic central 16-frame window per person track; four tracks total',
            'common_pelvis': validation['definition']['prediction_pelvis'],
            'world_transform': 'X_world = R_cw^T (X_camera - t_cw), using oracle 3DPW extrinsics',
            'w_mpjpe': 'direct world H36M J14 mean joint error',
            'wa_mpjpe': 'per-window first-frame pelvis translation aligned, no rotation or scale',
            'ate': 'per-window RMS global pelvis position error without alignment',
            'rte': 'mean discrepancy of consecutive global pelvis translations',
            'camera_cross_check': 'same Phase-2 t_all and normalized reprojection definitions as frozen camera-space smoke',
        },
        'results': results,
        'overall': {c: {m: aggregate(all_bundles[c][m]) for m in METHODS} for c in CONDITIONS},
        'over_smoothing_gate': 'compare Full/Frame-wise world pelvis speed; risk if <0.8',
        'elapsed_seconds': time.time() - started,
    }
    WORLD_OUT.write_text(json.dumps(world_result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'world_smoke': str(WORLD_OUT), 'windows': len(selected_windows), 'elapsed_seconds': world_result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
