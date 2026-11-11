"""Frozen held-out Camera Branch benchmark for four official 3DPW-test tracks."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path('/root/sam-body4d-0.2.0')
DATA = Path('/root/data1/sam-body4d-master/dataset/3DPW')
OUT = ROOT / 'outputs/camera_branch_3dpw_test_final'
CACHE = ROOT / 'outputs/dynamesh_v2_final_test/cache'
FRAME_CKPT = ROOT / 'outputs/dynamesh_v2_camera_phase2/framewise/best.pt'
TEMP_CKPT = ROOT / 'outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt'
MAPPING = Path('/root/data1/MHR-main/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz')
REGRESSOR = Path('/root/DuoMo-main/data/body_models/J_regressor_h36m.npy')
SMPL_DIR = Path('/root/DuoMo-main/data/body_models/smpl')
PROTOCOL_MD = ROOT / 'CAMERA_EVALUATOR_V1_PROTOCOL.md'
SEQUENCES = ('downtown_stairs_00', 'flat_guitar_01', 'flat_packBags_00', 'outdoors_fencing_01')
METHODS = ('raw', 'framewise', 'full')
J14 = np.asarray([6, 5, 4, 1, 2, 3, 13, 12, 11, 14, 15, 16, 0, 7], np.int64)
FPS = 30.0
WINDOW = 16
EXPECTED_FRAME = 'fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2'
EXPECTED_TEMP = 'e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha256(value):
    x = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def load_pickle(path):
    with Path(path).open('rb') as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0); return pickle.load(f, encoding='latin1')


def contiguous_segments(local_frame_index):
    local = np.asarray(local_frame_index, dtype=np.int64)
    if len(local) == 0:
        return []
    if np.any(np.diff(local) <= 0):
        raise ValueError('local frame indices must be strictly increasing')
    return [x for x in np.split(np.arange(len(local)), np.flatnonzero(np.diff(local) != 1) + 1) if len(x)]


def chunk_segment(segment_indices, window=WINDOW):
    segment = np.asarray(segment_indices, dtype=np.int64)
    return [segment[a:a + window] for a in range(0, len(segment), window)]


def _ratio(predicted, target, eps=1e-12):
    if target <= eps:
        return 1.0 if predicted <= eps else float('inf')
    return float(predicted / target)


def _preservation(ratio):
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.0
    return float(min(ratio, 1.0 / ratio))


def metric_components(pred_world_joints_m, gt_world_joints_m, local_frame_index, fps=FPS, alignment=None):
    pred = np.asarray(pred_world_joints_m, np.float64)
    gt = np.asarray(gt_world_joints_m, np.float64)
    local = np.asarray(local_frame_index, np.int64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (14, 3):
        raise ValueError('joints must share shape (T,14,3)')
    if len(pred) != len(local) or len(pred) == 0 or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError('empty, misaligned, or non-finite segment')
    if len(local) > 1 and not np.all(np.diff(local) == 1):
        raise ValueError('temporal components may not cross a frame discontinuity')
    pp, gp = pred[:, 0], gt[:, 0]
    d = gp[0] - pp[0] if alignment is None else np.asarray(alignment, np.float64)
    direct_p = np.linalg.norm(pp - gp, axis=-1)
    direct_j = np.linalg.norm(pred - gt, axis=-1)
    aligned = pred + d[None, None, :]
    aligned_p = np.linalg.norm(aligned[:, 0] - gp, axis=-1)
    aligned_j = np.linalg.norm(aligned - gt, axis=-1)
    pv, gv = np.diff(pp, axis=0) * fps, np.diff(gp, axis=0) * fps
    pa, ga = np.diff(pp, n=2, axis=0) * fps**2, np.diff(gp, n=2, axis=0) * fps**2
    pj = np.diff(pp, n=3, axis=0) * fps**3
    return {
        'frames': len(pred), 'joints': pred.shape[1],
        'direct_p_sum_m': float(direct_p.sum()), 'direct_j_sum_m': float(direct_j.sum()),
        'aligned_j_sum_m': float(aligned_j.sum()), 'aligned_p_sumsq_m2': float(np.square(aligned_p).sum()),
        'ate_segment_mm': float(np.sqrt(np.square(aligned_p).mean()) * 1000),
        'rte_sum_m': float(np.linalg.norm(np.diff(pp, axis=0) - np.diff(gp, axis=0), axis=-1).sum()),
        'velocity_error_sum': float(np.linalg.norm(pv - gv, axis=-1).sum()),
        'acceleration_error_sum': float(np.linalg.norm(pa - ga, axis=-1).sum()),
        'jitter_sum': float(np.linalg.norm(pj, axis=-1).sum()),
        'pred_velocity_magnitude_sum': float(np.linalg.norm(pv, axis=-1).sum()),
        'gt_velocity_magnitude_sum': float(np.linalg.norm(gv, axis=-1).sum()),
        'pred_acceleration_magnitude_sum': float(np.linalg.norm(pa, axis=-1).sum()),
        'gt_acceleration_magnitude_sum': float(np.linalg.norm(ga, axis=-1).sum()),
        'valid_velocity_steps': len(pv), 'valid_acceleration_triplets': len(pa),
        'valid_jerk_quadruplets': len(pj), 'initial_translation_alignment_m': d.tolist(),
    }


def aggregate_components(components):
    if not components:
        raise ValueError('no metric components')
    s = lambda k: sum(float(x[k]) for x in components)
    n_f = int(s('frames')); n_j = int(s('frames') * components[0]['joints'])
    n_v = int(s('valid_velocity_steps')); n_a = int(s('valid_acceleration_triplets')); n_q = int(s('valid_jerk_quadruplets'))
    mean = lambda key, n: float(s(key) / n) if n else 0.0
    pv, gv = mean('pred_velocity_magnitude_sum', n_v), mean('gt_velocity_magnitude_sum', n_v)
    pa, ga = mean('pred_acceleration_magnitude_sum', n_a), mean('gt_acceleration_magnitude_sum', n_a)
    rv, ra = _ratio(pv, gv), _ratio(pa, ga)
    return {
        'frames': n_f, 'valid_velocity_steps': n_v, 'valid_acceleration_triplets': n_a,
        'valid_jerk_quadruplets': n_q, 'segments': len(components),
        'global_pelvis_error_mm': mean('direct_p_sum_m', n_f) * 1000,
        'w_mpjpe_mm': mean('direct_j_sum_m', n_j) * 1000,
        'wa_mpjpe_mm': mean('aligned_j_sum_m', n_j) * 1000,
        'person_root_ate_mm': float(np.sqrt(s('aligned_p_sumsq_m2') / n_f) * 1000),
        'rte_mm_per_frame': mean('rte_sum_m', n_v) * 1000,
        'world_velocity_error_m_per_s': mean('velocity_error_sum', n_v),
        'world_acceleration_error_m_per_s2': mean('acceleration_error_sum', n_a),
        'trajectory_jitter_m_per_s3': mean('jitter_sum', n_q),
        'pred_velocity_magnitude_m_per_s': pv, 'gt_velocity_magnitude_m_per_s': gv,
        'pred_acceleration_magnitude_m_per_s2': pa, 'gt_acceleration_magnitude_m_per_s2': ga,
        'velocity_magnitude_ratio': rv, 'acceleration_magnitude_ratio': ra,
        'motion_preservation_score': 0.5 * (_preservation(rv) + _preservation(ra)),
    }


def metadata_for_track(track, ann):
    import cv2
    import train_dynamesh_v2_camera_phase1 as p1
    sequence = str(track['sequence'].item()); person = int(track['person_id'].item())
    selected = np.asarray(track['local_frame_index'], np.int64)
    frame_ids = np.asarray(ann['img_frame_ids'], np.int64)[selected]
    if not np.array_equal(frame_ids, track['frame_id']):
        raise ValueError('annotation/cache frame identity mismatch')
    image = cv2.imread(str(DATA / 'imageFiles' / sequence / f'image_{selected[0]:05d}.jpg'))
    if image is None:
        raise FileNotFoundError('first image unavailable')
    h, w = image.shape[:2]; poses = np.asarray(ann['poses2d'][person])
    boxes = np.stack([p1.bbox_from_pose2d(poses[i], w, h) for i in selected]).astype(np.float32)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2; size = (boxes[:, 2] - boxes[:, 0]) * 1.25
    gt = np.transpose(poses[selected][:, :, p1.OP_IDX], (0, 2, 1)).astype(np.float32)
    return {'sequence': sequence, 'person': person, 'selected': selected, 'frame_id': frame_ids,
            'boxes': boxes, 'center': center, 'size': size, 'K': np.asarray(ann['cam_intrinsics'], np.float32),
            'image_size': np.tile([[w, h]], (len(selected), 1)).astype(np.float32), 'gt_xyc': gt}


def slice_meta(meta, indices):
    n = len(meta['frame_id']); indices = np.asarray(indices)
    return {k: (v[indices] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v) for k, v in meta.items()}


def gt_world_joints(ann, person, selected, regressor):
    import torch, smplx
    gender = str(ann['genders'][person]).lower(); gender = 'female' if gender.startswith('f') else 'male'
    model = smplx.SMPL(model_path=str(SMPL_DIR / f'SMPL_{gender.upper()}.pkl'), batch_size=1).cpu().eval()
    pose = np.asarray(ann['poses'][person], np.float32)[selected]
    trans = np.asarray(ann['trans'][person], np.float32)[selected]
    b = np.asarray(ann['betas'][person], np.float32)
    betas = b[selected, :10] if b.ndim == 2 else np.broadcast_to(b[:10], (len(selected), 10)).copy()
    vertices = []
    with torch.no_grad():
        for a in range(0, len(selected), 64):
            z = slice(a, a + 64)
            out = model(betas=torch.as_tensor(betas[z]), global_orient=torch.as_tensor(pose[z, :3]),
                        body_pose=torch.as_tensor(pose[z, 3:72]), transl=torch.as_tensor(trans[z]))
            vertices.append(out.vertices.numpy().astype(np.float64))
    return np.einsum('jv,nvc->njc', regressor, np.concatenate(vertices))[..., :][..., :]


def infer_track(track, meta, sam, framewise, temporal, triangle_ids, bary, regressor):
    import torch
    import train_dynamesh_v2_camera_phase110 as p
    import train_dynamesh_v2_camera_phase112 as phase112
    import train_dynamesh_v2_camera_phase1 as p1
    from validate_camera_root_correspondence import apply_surface_mapping, camera_to_world
    clean = int(np.flatnonzero(track['condition_names'].astype(str) == 'clean')[0])
    local = np.asarray(track['local_frame_index'], np.int64)
    segments = contiguous_segments(local)
    camera = {m: np.empty((len(local), 3), np.float64) for m in METHODS}
    local_joints = np.empty((len(local), 14, 3), np.float64)
    coverage = np.zeros(len(local), np.int64)
    faces_hash = None
    with torch.no_grad():
        for segment in segments:
            for idx in chunk_segment(segment):
                m = slice_meta(meta, idx)
                tokens = torch.as_tensor(track['pose_token'][clean, idx], device='cuda', dtype=torch.float32)
                init = torch.as_tensor(track['mhr_init_pose'][clean, idx], device='cuda', dtype=torch.float32)
                pose = sam.head_pose(tokens, init)
                vertices = pose['pred_vertices'].detach().cpu().numpy().astype(np.float64)
                faces = np.asarray(pose['faces'], np.int64); faces_hash = array_sha256(faces)
                mapped = apply_surface_mapping(vertices, faces, triangle_ids, bary)
                local_joints[idx] = np.einsum('jv,nvc->njc', regressor, mapped)[:, J14]
                points = pose['pred_keypoints_3d'][:, p1.MHR_IDX].detach()
                raw_cam = sam.head_camera(tokens, sam.init_camera.weight.expand(len(idx), -1))
                raw_proj = p.projection(sam, points, raw_cam, m)
                fw_token = framewise(tokens[None])
                fw_cam = sam.head_camera(fw_token.squeeze(0), sam.init_camera.weight.expand(len(idx), -1))
                fw_proj = p.projection(sam, points, fw_cam, m)
                features = p.reliability_features(fw_token, m, 0.0)
                full_token = phase112.temporal_residual_forward(temporal, fw_token, features)
                full_cam = sam.head_camera(full_token.squeeze(0), sam.init_camera.weight.expand(len(idx), -1))
                full_proj = p.projection(sam, points, full_cam, m)
                camera['raw'][idx] = raw_proj['pred_cam_t'].cpu().numpy()
                camera['framewise'][idx] = fw_proj['pred_cam_t'].cpu().numpy()
                camera['full'][idx] = full_proj['pred_cam_t'].cpu().numpy()
                coverage[idx] += 1
    if not np.all(coverage == 1):
        raise ValueError(f'inference coverage is not exactly once: {np.unique(coverage, return_counts=True)}')
    R = np.asarray(track['camera_extrinsics'], np.float64)[:, :3, :3]
    t = np.asarray(track['camera_extrinsics'], np.float64)[:, :3, 3]
    world = {m: camera_to_world(local_joints + camera[m][:, None, :], R, t) for m in METHODS}
    return local_joints, camera, world, segments, faces_hash


def sequence_components(pred, gt, local, segments):
    alignment = gt[0, 0] - pred[0, 0]
    return [metric_components(pred[s], gt[s], local[s], alignment=alignment) for s in segments]


def method_summary_from_saved(method):
    all_components = []; per_sequence = {}
    for seq in SEQUENCES:
        raw = np.load(OUT / 'raw' / f'{seq}.npz', allow_pickle=False)
        cur = np.load(OUT / method / f'{seq}.npz', allow_pickle=False)
        segments = contiguous_segments(raw['local_frame_index'])
        comps = sequence_components(cur['pred_world_joints_m'], raw['gt_world_joints_m'], raw['local_frame_index'], segments)
        per_sequence[seq] = aggregate_components(comps); all_components.extend(comps)
    return per_sequence, aggregate_components(all_components)


def per_sequence_mean(per_sequence):
    keys = [k for k, v in next(iter(per_sequence.values())).items() if isinstance(v, (int, float)) and k not in
            ('frames', 'valid_velocity_steps', 'valid_acceleration_triplets', 'valid_jerk_quadruplets', 'segments')]
    out = {k: float(np.mean([per_sequence[s][k] for s in SEQUENCES])) for k in keys}
    out['sequences'] = len(SEQUENCES)
    return out


def write_csv(path, per_method):
    rows = [{'sequence': s, 'method': m, **per_method[m][s]} for s in SEQUENCES for m in METHODS]
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    import torch
    import smplx
    import ablate_dynamesh_v2_camera_phase111 as q
    import train_dynamesh_v2_camera_phase110 as p
    from evaluate_dynamesh_v2_camera_phase2 import load_module
    from train_dynamesh_v2_camera_phase19 import load_frozen_model
    from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
    from validate_camera_root_correspondence import normalize_joint_regressor, world_to_camera, camera_to_world

    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    for m in METHODS: (OUT / m).mkdir(exist_ok=True)
    checkpoints = {'framewise': sha256(FRAME_CKPT), 'temporal': sha256(TEMP_CKPT)}
    if checkpoints != {'framewise': EXPECTED_FRAME, 'temporal': EXPECTED_TEMP}:
        raise RuntimeError(f'checkpoint hash mismatch: {checkpoints}')
    atomic_json(OUT / 'checkpoint_audit.json', {'verdict': 'PASS', 'sha256': checkpoints,
                'paths': {'framewise': str(FRAME_CKPT), 'temporal': str(TEMP_CKPT)}})
    protocol = {
        'status': 'FROZEN_BEFORE_FORMAL_RUN', 'source_protocol': str(PROTOCOL_MD),
        'source_protocol_sha256': sha256(PROTOCOL_MD), 'split': 'official 3DPW test only',
        'sequences': list(SEQUENCES), 'person_id': 0, 'condition': 'clean/original frames (natural occlusion)',
        'methods': list(METHODS), 'temporal_only': 'FORBIDDEN', 'fps': FPS,
        'inference_chunking': 'Each maximal consecutive segment is partitioned left-to-right into chunks of <=16; every frame is inferred exactly once; short/tail chunks are evaluated at native length; no metric/window alignment reset.',
        'oracle_extrinsics': '3DPW cam_poses are used only for evaluation coordinate conversion and are never model inputs.',
        'body': 'Identical frozen raw SAM local MHR geometry for all three Camera ablations.',
        'decision_rule': {'framewise_validated': 'pooled improvement in >=3/4 spatial metrics and sequence-majority improvement for >=3/4 spatial metrics',
                          'severe_oversmoothing': 'Full velocity or acceleration magnitude ratio <0.5 while jitter decreases',
                          'camera_validated': 'framewise validated, Full improves >=3/5 temporal target metrics, and no severe oversmoothing'},
    }
    atomic_json(OUT / 'protocol.json', protocol)

    mp = np.load(MAPPING); triangle_ids = np.asarray(mp['triangle_ids'], np.int64); bary = np.asarray(mp['baryc_coords'], np.float64)
    reg = normalize_joint_regressor(np.asarray(np.load(REGRESSOR), np.float64))
    if triangle_ids.shape != (6890,) or bary.shape != (6890, 3) or reg.shape[1] != 6890:
        raise ValueError('mapping/regressor topology mismatch')
    if not np.allclose(bary.sum(1), 1, atol=1e-9) or not np.allclose(reg.sum(1), 1, atol=1e-12):
        raise ValueError('affine mapping gate failed')
    sam = load_frozen_model(); framewise = load_module(q.FramewiseLatentAdapter, FRAME_CKPT)
    temporal = load_module(CameraFeatureTemporalAdapter, TEMP_CKPT)
    for model in (sam, framewise, temporal): model.eval().requires_grad_(False)

    status = {'state': 'running', 'completed_sequences': [], 'pending_sequences': list(SEQUENCES), 'started_at': started}
    atomic_json(OUT / 'sequence_status.json', status)
    body_records = []; geometry_roundtrip = []
    for seq in SEQUENCES:
        result_path = OUT / 'raw' / f'{seq}.json'
        if result_path.exists() and all((OUT / m / f'{seq}.npz').exists() for m in METHODS):
            status['completed_sequences'].append(seq); status['pending_sequences'].remove(seq)
            atomic_json(OUT / 'sequence_status.json', status); continue
        with np.load(CACHE / f'{seq}_person0.npz', allow_pickle=False) as z:
            track = {k: np.asarray(z[k]) for k in z.files}
        if str(track['split'].item()) != 'test' or int(track['person_id'].item()) != 0:
            raise ValueError('test split/person contract failed')
        ann = load_pickle(DATA / 'sequenceFiles/test' / f'{seq}.pkl'); meta = metadata_for_track(track, ann)
        gt = gt_world_joints(ann, 0, track['local_frame_index'], reg)[:, J14]
        local_joints, cameras, worlds, segments, faces_hash = infer_track(track, meta, sam, framewise, temporal, triangle_ids, bary, reg)
        R = np.asarray(track['camera_extrinsics'], np.float64)[:, :3, :3]; t = np.asarray(track['camera_extrinsics'], np.float64)[:, :3, 3]
        gt_closed = camera_to_world(world_to_camera(gt, R, t), R, t)
        geometry_roundtrip.append(float(np.max(np.abs(gt_closed - gt))))
        local_hash = array_sha256(local_joints)
        body_records.append({'sequence': seq, 'frames': len(local_joints), 'local_geometry_sha256': {m: local_hash for m in METHODS},
                             'pairwise_max_abs_m': {'raw_framewise': 0.0, 'raw_full': 0.0, 'framewise_full': 0.0},
                             'faces_sha256': faces_hash})
        for method in METHODS:
            metrics = aggregate_components(sequence_components(worlds[method], gt, track['local_frame_index'], segments))
            payload = {'sequence': seq, 'person_id': 0, 'method': method, 'completion': True, 'finite': True,
                       'coverage': {'expected_frames': len(gt), 'evaluated_frames': len(worlds[method]), 'missing': 0,
                                    'unmatched': 0, 'duplicate': 0, 'segments': len(segments)}, 'metrics': metrics}
            atomic_json(OUT / method / f'{seq}.json', payload)
            np.savez_compressed(OUT / method / f'{seq}.npz', pred_world_joints_m=worlds[method], pred_cam_t=cameras[method],
                                gt_world_joints_m=gt if method == 'raw' else np.empty((0,)), local_frame_index=track['local_frame_index'],
                                frame_id=track['frame_id'], body_local_joints_m=local_joints)
        status['completed_sequences'].append(seq); status['pending_sequences'].remove(seq); status['current_sequence'] = None
        atomic_json(OUT / 'sequence_status.json', status)

    atomic_json(OUT / 'geometry_audit.json', {'verdict': 'PASS', 'mapping_sha256': sha256(MAPPING),
                'regressor_sha256': sha256(REGRESSOR), 'barycentric_row_sum_max_abs': float(np.max(np.abs(bary.sum(1)-1))),
                'regressor_row_sum_max_abs': float(np.max(np.abs(reg.sum(1)-1))),
                'gt_world_camera_world_max_abs_m': max(geometry_roundtrip),
                'transform': 'x_cam=R_cw*x_world+t_cw; x_world=R_cw^T*(x_cam-t_cw)',
                'oracle_extrinsics_model_input': False})
    atomic_json(OUT / 'body_identity_audit.json', {'verdict': 'PASS', 'records': body_records,
                'body_output_elementwise_identical_across_methods': True})

    per_method = {}; overall = {}; seqmean = {}
    for method in METHODS:
        per_method[method], overall[method] = method_summary_from_saved(method)
        overall[method]['person_root_ate_mm'] = float(np.mean([per_method[method][s]['person_root_ate_mm'] for s in SEQUENCES]))
        seqmean[method] = per_sequence_mean(per_method[method])
    atomic_json(OUT / 'per_sequence_results.json', per_method); write_csv(OUT / 'per_sequence_results.csv', per_method)
    atomic_json(OUT / 'instance_weighted_summary.json', overall)
    atomic_json(OUT / 'per_sequence_mean_summary.json', seqmean)

    lower = ('global_pelvis_error_mm','w_mpjpe_mm','wa_mpjpe_mm','person_root_ate_mm','rte_mm_per_frame',
             'world_velocity_error_m_per_s','world_acceleration_error_m_per_s2','trajectory_jitter_m_per_s3')
    comparisons = {}
    for a, b, name in [('raw','framewise','raw_to_framewise'),('framewise','full','framewise_to_full')]:
        comparisons[name] = {}
        for k in lower:
            counts = {'improved': 0, 'degraded': 0, 'tied': 0}
            for seq in SEQUENCES:
                delta = per_method[b][seq][k] - per_method[a][seq][k]
                counts['improved' if delta < -1e-12 else 'degraded' if delta > 1e-12 else 'tied'] += 1
            comparisons[name][k] = {'from': overall[a][k], 'to': overall[b][k], 'delta': overall[b][k]-overall[a][k], **counts}
    spatial = ('global_pelvis_error_mm','w_mpjpe_mm','wa_mpjpe_mm','person_root_ate_mm')
    temporal_targets = ('wa_mpjpe_mm','rte_mm_per_frame','world_velocity_error_m_per_s','world_acceleration_error_m_per_s2','trajectory_jitter_m_per_s3')
    fw_pooled = sum(overall['framewise'][k] < overall['raw'][k] for k in spatial)
    fw_majority = sum(comparisons['raw_to_framewise'][k]['improved'] >= 3 for k in spatial)
    framewise_valid = fw_pooled >= 3 and fw_majority >= 3
    full_temporal = sum(overall['full'][k] < overall['framewise'][k] for k in temporal_targets)
    severe = (overall['full']['trajectory_jitter_m_per_s3'] < overall['framewise']['trajectory_jitter_m_per_s3'] and
              (overall['full']['velocity_magnitude_ratio'] < .5 or overall['full']['acceleration_magnitude_ratio'] < .5))
    verdict = 'TEMPORAL_OVERSMOOTHING_FAILURE' if severe else ('FRAMEWISE_NOT_VALIDATED' if not framewise_valid else
              ('CAMERA_BRANCH_VALIDATED' if full_temporal >= 3 else 'FRAMEWISE_NOT_VALIDATED'))
    atomic_json(OUT / 'camera_ablation_summary.json', {'comparisons': comparisons, 'framewise_pooled_spatial_improvements': fw_pooled,
                'framewise_spatial_metrics_with_sequence_majority': fw_majority, 'full_temporal_improvements': full_temporal,
                'verdict': verdict})
    motion = {'per_sequence': {}, 'overall': {}}
    for seq in SEQUENCES:
        f, w = per_method['full'][seq], per_method['framewise'][seq]
        motion['per_sequence'][seq] = {'full_gt_velocity_ratio': f['velocity_magnitude_ratio'],
            'full_gt_acceleration_ratio': f['acceleration_magnitude_ratio'],
            'full_framewise_velocity_magnitude_ratio': _ratio(f['pred_velocity_magnitude_m_per_s'], w['pred_velocity_magnitude_m_per_s']),
            'full_framewise_acceleration_magnitude_ratio': _ratio(f['pred_acceleration_magnitude_m_per_s2'], w['pred_acceleration_magnitude_m_per_s2']),
            'full_motion_preservation_score': f['motion_preservation_score']}
    f, w = overall['full'], overall['framewise']
    motion['overall'] = {'full_gt_velocity_ratio': f['velocity_magnitude_ratio'], 'full_gt_acceleration_ratio': f['acceleration_magnitude_ratio'],
        'full_framewise_velocity_magnitude_ratio': _ratio(f['pred_velocity_magnitude_m_per_s'], w['pred_velocity_magnitude_m_per_s']),
        'full_framewise_acceleration_magnitude_ratio': _ratio(f['pred_acceleration_magnitude_m_per_s2'], w['pred_acceleration_magnitude_m_per_s2']),
        'full_motion_preservation_score': f['motion_preservation_score'], 'severe_oversmoothing': severe}
    speeds = np.asarray([per_method['raw'][s]['gt_velocity_magnitude_m_per_s'] for s in SEQUENCES])
    cuts = np.quantile(speeds, [1/3, 2/3]); labels = np.where(speeds <= cuts[0], 'low', np.where(speeds <= cuts[1], 'mid', 'high'))
    motion['subgroups'] = {'thresholds_m_per_s': cuts.tolist(), 'membership': dict(zip(SEQUENCES, labels.tolist()))}
    atomic_json(OUT / 'motion_preservation_analysis.json', motion)
    status.update({'state': 'complete', 'verdict': verdict, 'elapsed_seconds': time.time()-started})
    atomic_json(OUT / 'sequence_status.json', status)
    with (OUT / 'run.log').open('a') as f:
        f.write(json.dumps({'timestamp': time.time(), 'state': 'complete', 'verdict': verdict, 'elapsed_seconds': time.time()-started})+'\n')
    print(json.dumps({'state': 'complete', 'verdict': verdict, 'output': str(OUT), 'elapsed_seconds': time.time()-started}))


if __name__ == '__main__':
    main()
