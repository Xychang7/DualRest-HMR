from __future__ import annotations

import numpy as np


SPATIAL_KEYS_MM = (
    'global_pelvis_error_mm',
    'w_mpjpe_mm',
    'wa_mpjpe_mm',
    'person_root_ate_mm',
    'rte_mm_per_frame',
)


def _finite(name, value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite values')
    return value


def validate_identity_frame_mapping(prediction_keys, ground_truth_keys):
    prediction_keys = [tuple(x) for x in prediction_keys]
    ground_truth_keys = [tuple(x) for x in ground_truth_keys]
    if len(set(prediction_keys)) != len(prediction_keys):
        raise ValueError('duplicate prediction identity/frame key')
    if len(set(ground_truth_keys)) != len(ground_truth_keys):
        raise ValueError('duplicate ground-truth identity/frame key')
    if prediction_keys != ground_truth_keys:
        raise ValueError('prediction and ground truth must match exactly in identity/frame order')
    return True


def _ratio(predicted, target, eps=1e-12):
    predicted, target = float(predicted), float(target)
    if target <= eps:
        return 1.0 if predicted <= eps else float('inf')
    return predicted / target


def _preservation(ratio):
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.0
    return float(min(ratio, 1.0 / ratio))


def sequence_metrics(pred_world_joints_m, gt_world_joints_m, local_frame_index, *, fps=30.0):
    pred = _finite('predicted world joints', pred_world_joints_m)
    gt = _finite('ground-truth world joints', gt_world_joints_m)
    local = np.asarray(local_frame_index, dtype=np.int64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError('predicted and GT joints must share shape (T,J,3)')
    if len(pred) < 4 or len(local) != len(pred):
        raise ValueError('at least four aligned frames are required')
    if not np.all(np.diff(local) == 1):
        raise ValueError('local_frame_index must be consecutive within a person track')
    if fps <= 0:
        raise ValueError('fps must be positive')

    pred_pelvis, gt_pelvis = pred[:, 0], gt[:, 0]
    direct_pelvis = np.linalg.norm(pred_pelvis - gt_pelvis, axis=-1)
    direct_joints = np.linalg.norm(pred - gt, axis=-1)

    initial_translation = gt_pelvis[0] - pred_pelvis[0]
    pred_aligned = pred + initial_translation[None, None, :]
    aligned_pelvis = np.linalg.norm(pred_aligned[:, 0] - gt_pelvis, axis=-1)
    aligned_joints = np.linalg.norm(pred_aligned - gt, axis=-1)

    pred_velocity = np.diff(pred_pelvis, axis=0) * fps
    gt_velocity = np.diff(gt_pelvis, axis=0) * fps
    pred_acceleration = np.diff(pred_pelvis, n=2, axis=0) * fps ** 2
    gt_acceleration = np.diff(gt_pelvis, n=2, axis=0) * fps ** 2
    pred_jerk = np.diff(pred_pelvis, n=3, axis=0) * fps ** 3

    velocity_ratio = _ratio(np.linalg.norm(pred_velocity, axis=-1).mean(),
                            np.linalg.norm(gt_velocity, axis=-1).mean())
    acceleration_ratio = _ratio(np.linalg.norm(pred_acceleration, axis=-1).mean(),
                                np.linalg.norm(gt_acceleration, axis=-1).mean())
    preservation = 0.5 * (_preservation(velocity_ratio) + _preservation(acceleration_ratio))

    return {
        'frames': int(len(pred)),
        'valid_velocity_steps': int(len(pred_velocity)),
        'valid_acceleration_triplets': int(len(pred_acceleration)),
        'valid_jerk_quadruplets': int(len(pred_jerk)),
        'global_pelvis_error_mm': float(direct_pelvis.mean() * 1000.0),
        'w_mpjpe_mm': float(direct_joints.mean() * 1000.0),
        'wa_mpjpe_mm': float(aligned_joints.mean() * 1000.0),
        'person_root_ate_mm': float(np.sqrt(np.mean(aligned_pelvis ** 2)) * 1000.0),
        'rte_mm_per_frame': float(np.linalg.norm(np.diff(pred_pelvis, axis=0) - np.diff(gt_pelvis, axis=0), axis=-1).mean() * 1000.0),
        'world_velocity_error_m_per_s': float(np.linalg.norm(pred_velocity - gt_velocity, axis=-1).mean()),
        'world_acceleration_error_m_per_s2': float(np.linalg.norm(pred_acceleration - gt_acceleration, axis=-1).mean()),
        'trajectory_jitter_m_per_s3': float(np.linalg.norm(pred_jerk, axis=-1).mean()),
        'pred_velocity_magnitude_m_per_s': float(np.linalg.norm(pred_velocity, axis=-1).mean()),
        'gt_velocity_magnitude_m_per_s': float(np.linalg.norm(gt_velocity, axis=-1).mean()),
        'pred_acceleration_magnitude_m_per_s2': float(np.linalg.norm(pred_acceleration, axis=-1).mean()),
        'gt_acceleration_magnitude_m_per_s2': float(np.linalg.norm(gt_acceleration, axis=-1).mean()),
        'velocity_magnitude_ratio': float(velocity_ratio),
        'acceleration_magnitude_ratio': float(acceleration_ratio),
        'motion_preservation_score': float(preservation),
        'initial_translation_alignment_m': initial_translation.tolist(),
    }


def assign_motion_subgroups(track_gt_velocity_magnitudes):
    values = _finite('GT track motion magnitudes', track_gt_velocity_magnitudes).reshape(-1)
    if len(values) < 3:
        raise ValueError('at least three person tracks are required for motion subgroups')
    low_cut, high_cut = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.where(values <= low_cut, 'low', np.where(values <= high_cut, 'mid', 'high'))
    return labels.tolist(), {'low_mid_cut_m_per_s': float(low_cut), 'mid_high_cut_m_per_s': float(high_cut)}
