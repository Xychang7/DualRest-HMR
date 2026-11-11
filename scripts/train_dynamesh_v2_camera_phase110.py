"""Camera Phase-1.10: small multi-window reliability/preservation training.

Uses only cached 3DPW-train tokens and Phase-1.8 targets. It never runs the
visual backbone, Meta conversion, or any validation/test data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_dynamesh_v2_camera_phase19 import (
    BODY, CACHE, CONDITIONS, RATIOS, ROOT, SEED, atomic_json,
    assert_inference_reliability_fields, assert_loss_contract,
    balance_gradient_lambdas, gradient_norm, load_adapter, load_frozen_model,
    prepare_candidate, read_phase18_rows, sha256, stats, target_reliability,
    total_loss, weighted_cam3d_loss,
)
from ablate_dynamesh_v2_camera_phase16 import adapter_forward, projection
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase1 import (
    assert_camera_optimizer_ownership, frame_reprojection,
    masked_normalized_reprojection_loss, reliability_features,
    set_seed, weighted_smooth_l1,
)
from validate_dynamesh_v2_camera_multijoint_gt import SMPL_MODEL

OUT = ROOT / 'outputs/dynamesh_v2_camera_phase110_multiwindow'
INTERRUPTED_OUT = ROOT / 'outputs/dynamesh_v2_camera_phase110_multiwindow_interrupted_step1360'
LOCKED_PROBE = INTERRUPTED_OUT / 'loss_scale_probe.json'
CATEGORIES = ('stable', 'medium', 'failure')
WINDOW = 16


def _finite(values):
    x = np.asarray(values, dtype=np.float64)
    return x[np.isfinite(x)]


def derive_quality_thresholds(rows):
    """Derive all labels from the complete clean train distribution."""
    if any(r.get('split', 'train') != 'train' for r in rows):
        raise ValueError('quality thresholds accept 3DPW train split only')
    clean = [r for r in rows if r['condition'] == 'clean']
    if not clean:
        raise ValueError('empty clean train distribution')
    fields = {
        'camera': 'camera_error', 'fit': 'fit_residual',
        'ul': 'upper_lower_disagreement', 'reproj': 'reprojection_error',
        'body': 'body_teacher_geodesic_deg',
    }
    out = {'source': 'full 3DPW train clean quantiles', 'count': len(clean)}
    for short, key in fields.items():
        values = _finite([r[key] for r in clean])
        if not len(values):
            raise ValueError(f'no finite train values for {key}')
        for q in (25, 40, 50, 60, 75):
            out[f'{short}_p{q}'] = float(np.percentile(values, q))
    out['tau_fit'] = out['fit_p50']
    out['tau_ul'] = out['ul_p50']
    out['need_low'] = out['camera_p25']
    out['need_high'] = out['camera_p75']
    return out


def classify_quality(row, t):
    target_ok = (row['fit_residual'] <= t['fit_p75'] and
                 row['upper_lower_disagreement'] <= t['ul_p75'] and
                 row['body_teacher_geodesic_deg'] <= t['body_p75'])
    if (target_ok and row['camera_error'] <= t['camera_p25'] and
            row['reprojection_error'] <= t['reproj_p50']):
        return 'stable'
    if (target_ok and row['camera_error'] >= t['camera_p75'] and
            row['reprojection_error'] >= t['reproj_p75']):
        return 'failure'
    if (row['camera_error'] >= t['camera_p40'] and
            row['camera_error'] <= t['camera_p60'] and
            row['fit_residual'] <= t['fit_p75'] and
            row['upper_lower_disagreement'] <= t['ul_p75'] and
            row['reprojection_error'] <= t['reproj_p75'] and
            row['body_teacher_geodesic_deg'] <= t['body_p75']):
        return 'medium'
    return None


def camera_need(error, low, high):
    if not high > low:
        raise ValueError('need_high must exceed need_low')
    return np.clip((np.asarray(error, dtype=np.float64) - low) / (high - low), 0, 1)


def reliability_target(need):
    return 1.0 - np.asarray(need, dtype=np.float64)


def category_anchor(category, w_target, need):
    category = np.asarray(category)
    multiplier = np.zeros(category.shape, dtype=np.float64)
    multiplier[category == 'stable'] = 1.0
    multiplier[category == 'medium'] = 0.25
    return np.asarray(w_target, dtype=np.float64) * (1.0 - np.asarray(need)) * multiplier


def assert_disjoint_tracks(train_windows, heldout_windows):
    key = lambda x: (str(x['sequence']), int(x['person_id']))
    overlap = {key(x) for x in train_windows} & {key(x) for x in heldout_windows}
    if overlap:
        raise ValueError(f'person-track leakage: {sorted(overlap)}')


def assert_inference_contract(fields):
    forbidden = {
        'gt_2d', 'gt_3d', 't_all', 'fit_residual',
        'upper_lower_disagreement', 'w_target', 'camera_error', 'need',
    }
    leaked = forbidden.intersection(fields)
    if leaked:
        raise ValueError(f'GT/target leak into inference: {sorted(leaked)}')


def balanced_schedule(steps, categories=CATEGORIES, n_conditions=4, seed=SEED):
    rng = random.Random(seed)
    base = [(categories[i % len(categories)], i % n_conditions) for i in range(steps)]
    # Shuffle only complete category/condition blocks; totals remain balanced.
    block = len(categories) * n_conditions
    result = []
    for start in range(0, steps, block):
        chunk = base[start:start + block]
        rng.shuffle(chunk)
        result.extend(chunk)
    return result


def training_checkpoint(adapter, optimizer, current_step, counters, metadata):
    """Capture every state required to continue the exact optimization path."""
    rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return {
        'adapter_state_dict': adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_step': int(current_step),
        'rng_state': rng_state,
        'sampler_state': {'counters': dict(counters), 'schedule_seed': SEED},
        'metadata': dict(metadata),
    }


def locked_gradient_probe(path=LOCKED_PROBE):
    """Reuse the protocol-locked probe so GPU reduction noise cannot alter lambdas."""
    value = json.loads(Path(path).read_text())
    required = {'raw_losses', 'raw_gradient_norms', 'weighted_gradient_norms', 'lambdas'}
    # Unit tests may exercise a minimal payload; production locked probes are strict.
    if Path(path) == LOCKED_PROBE and not required.issubset(value):
        raise ValueError('locked Phase-1.10 gradient probe is incomplete')
    return value


def stable_output_layer_norm(adapter, value):
    """Evaluate LayerNorm statistics in float64 to avoid FP32 x^2 overflow."""
    norm = adapter.output_norm
    weight = norm.weight.double() if norm.weight is not None else None
    bias = norm.bias.double() if norm.bias is not None else None
    return F.layer_norm(value.double(), norm.normalized_shape, weight, bias, norm.eps).to(value.dtype)


def stable_mamba_stack(layers, value, diagnostics=None, prefix='mamba'):
    """Prevent unnormalised stacked SSM activations from cascading to overflow."""
    for index, layer in enumerate(layers):
        value = layer(value)
        if diagnostics is not None:
            diagnostics[f'{prefix}_layer_{index}_pre_norm_max_abs'] = float(value.detach().abs().max())
            diagnostics[f'{prefix}_layer_{index}_pre_norm_finite'] = bool(torch.isfinite(value).all())
        value = F.layer_norm(value.double(), (value.shape[-1],), eps=1e-5).to(value.dtype)
    return value


def numerically_stable_adapter_forward(adapter, raw_pose_token, features, diagnostics=None):
    tokens = raw_pose_token if raw_pose_token.ndim == 3 else raw_pose_token.unsqueeze(0)
    values, mask = features.packed()
    if values.ndim == 2: values, mask = values.unsqueeze(0), mask.unsqueeze(0)
    condition = torch.cat((values.to(tokens), mask.to(tokens)), -1)
    reliability = torch.sigmoid(adapter.reliability_head(condition))
    latent = adapter.input_projection(adapter.input_norm(tokens)) + adapter.condition_embedding(condition)
    forward = stable_mamba_stack(adapter.layers_fwd, latent, diagnostics, 'forward')
    backward = stable_mamba_stack(adapter.layers_bwd, torch.flip(latent, (1,)), diagnostics, 'backward')
    backward = torch.flip(backward, (1,))
    delta = adapter.output_projection(stable_output_layer_norm(adapter, forward + backward))
    return tokens + (1.0 - reliability) * delta, reliability


def _track_key(row):
    return str(row['sequence']), int(row['person_id'])


def build_window_pool(rows, thresholds):
    clean = [r for r in rows if r['condition'] == 'clean']
    groups = defaultdict(list)
    for row in clean:
        groups[_track_key(row)].append(row)
    pool = defaultdict(lambda: defaultdict(list))
    for (sequence, person), track_rows in sorted(groups.items()):
        track_rows.sort(key=lambda r: int(r['frame_id']))
        cache_path = CACHE / f'{sequence}_person{person}.npz'
        if not cache_path.exists():
            continue
        track = np.load(cache_path, allow_pickle=False)
        ids = track['frame_id'].astype(int)
        local = track['local_frame_index'].astype(int)
        lookup = {int(v): i for i, v in enumerate(ids)}
        for row in track_rows:
            category = classify_quality(row, thresholds)
            if category is None or int(row['frame_id']) not in lookup:
                continue
            center = lookup[int(row['frame_id'])]
            start = max(0, min(center - WINDOW // 2, len(ids) - WINDOW))
            if start < 0 or len(ids[start:start + WINDOW]) != WINDOW:
                continue
            if not np.all(np.diff(local[start:start + WINDOW]) == 1):
                continue
            descriptor = {
                'sequence': sequence, 'person_id': person,
                'category': category, 'center_frame': int(row['frame_id']),
                'window_start': int(start),
                'window_frame_ids': ids[start:start + WINDOW].tolist(),
                'cache_path': str(cache_path),
                'center_camera_error': float(row['camera_error']),
            }
            existing = pool[(sequence, person)][category]
            if not any(abs(x['window_start'] - start) < WINDOW for x in existing):
                existing.append(descriptor)
    return pool


def choose_manifest(pool, seed=SEED, train_tracks=4, heldout_tracks=2):
    eligible = [k for k, by_cat in sorted(pool.items()) if all(by_cat[c] for c in CATEGORIES)]
    if len(eligible) < train_tracks + heldout_tracks:
        raise RuntimeError(f'need {train_tracks + heldout_tracks} tracks with all categories, found {len(eligible)}')
    rng = random.Random(seed)
    rng.shuffle(eligible)
    train_keys, held_keys = eligible[:train_tracks], eligible[train_tracks:train_tracks + heldout_tracks]

    def select(keys, per_category):
        selected = []
        for key in keys:
            for category in CATEGORIES:
                candidates = sorted(pool[key][category], key=lambda x: (x['window_start'], x['center_frame']))
                # Spread selections across a track when two are requested.
                indices = [0] if per_category == 1 or len(candidates) == 1 else [0, len(candidates) - 1]
                for i in indices[:per_category]:
                    selected.append(dict(candidates[i]))
        return selected

    train = select(train_keys, 2)      # up to 24 windows
    heldout = select(held_keys, 2)     # up to 12 windows
    assert_disjoint_tracks(train, heldout)
    return {
        'seed': seed, 'split': '3dpw_train_only', 'window_length': WINDOW,
        'train_tracks': [{'sequence': k[0], 'person_id': k[1]} for k in train_keys],
        'heldout_tracks': [{'sequence': k[0], 'person_id': k[1]} for k in held_keys],
        'train_windows': train, 'heldout_windows': heldout,
    }


def prepare_window(desc, model, smpl_model, thresholds):
    candidate = {
        **desc, 'frame_id': desc['center_frame'],
        'window_frame_ids': desc['window_frame_ids'],
    }
    _, meta, tokens, raw, target = prepare_candidate(
        candidate, model, smpl_model,
        {'tau_fit': thresholds['tau_fit'], 'tau_ul': thresholds['tau_ul'],
         'camera_close_threshold': thresholds['camera_p25']},
    )
    category = desc['category']
    for c in range(4):
        raw_t = raw[c]['pred_cam_t'].detach().cpu().numpy()
        error = np.linalg.norm(raw_t - target[c]['t_all'], axis=1)
        need = camera_need(error, thresholds['need_low'], thresholds['need_high'])
        target[c]['need'] = need
        target[c]['reliability_target'] = reliability_target(need)
        target[c]['anchor_weight'] = category_anchor(
            np.full(WINDOW, category), target[c]['weight'], need,
        )
    return {'desc': desc, 'meta': meta, 'tokens': tokens, 'raw': raw, 'target': target}


def phase110_forward(adapter, model, sample, condition, diagnostics=None):
    meta, tokens, raw, target = sample['meta'], sample['tokens'], sample['raw'], sample['target']
    features = reliability_features(tokens[condition][None], meta, RATIOS[condition])
    fields = {'visibility', 'bbox_center_delta', 'bbox_scale_delta', 'temporal_latent_disagreement'}
    assert_inference_contract(fields)
    assert_inference_reliability_fields(fields)
    refined, reliability = numerically_stable_adapter_forward(adapter, tokens[condition][None], features, diagnostics)
    cam = model.head_camera(refined.squeeze(0), model.init_camera.weight.expand(WINDOW, -1))
    projected = projection(model, raw[condition]['points3d'].detach(), cam, meta)
    t = torch.as_tensor(target[condition]['t_all'], device='cuda', dtype=torch.float32)
    w = torch.as_tensor(target[condition]['weight'], device='cuda', dtype=torch.float32)
    r_target = torch.as_tensor(target[condition]['reliability_target'], device='cuda', dtype=torch.float32)
    anchor = torch.as_tensor(target[condition]['anchor_weight'], device='cuda', dtype=torch.float32)
    gt = torch.as_tensor(meta['gt_xyc'], device='cuda')
    bbox = torch.as_tensor(meta['size'], device='cuda')
    rel_per_frame = F.binary_cross_entropy(reliability.reshape(-1), r_target.detach(), reduction='none')
    terms = {
        'cam3d': weighted_cam3d_loss(projected['pred_cam_t'], t, w),
        'reproj': masked_normalized_reprojection_loss(projected['pred_keypoints_2d'], gt, bbox),
        'reliability': (rel_per_frame * w).sum() / w.sum().clamp_min(1e-8),
        'anchor': weighted_smooth_l1(cam, raw[condition]['pred_cam'].detach(), anchor),
    }
    assert_loss_contract(tuple(terms))
    return terms, {
        'pred_cam': cam, 'pred_cam_t': projected['pred_cam_t'],
        'pred_xy': projected['pred_keypoints_2d'],
        'reliability': reliability.reshape(-1), 'refined_token': refined,
    }


def gradient_probe(adapter, model, samples):
    selected = []
    for category in CATEGORIES:
        selected.append(next(s for s in samples if s['desc']['category'] == category))
    losses = defaultdict(list)
    for i, sample in enumerate(selected):
        terms, _ = phase110_forward(adapter, model, sample, i % 4)
        for key, value in terms.items():
            losses[key].append(value)
    mean = {k: torch.stack(v).mean() for k, v in losses.items()}
    params = [p for p in adapter.parameters() if p.requires_grad]
    raw = {k: gradient_norm(v, params) for k, v in mean.items()}
    # Phase-1.9 balancing, with small anchor if identity makes its initial gradient zero.
    balance = balance_gradient_lambdas(raw)
    if raw['anchor'] > 0:
        max_anchor = 0.5 * raw['cam3d']
        balance['lambdas']['anchor'] = min(0.1, max_anchor / raw['anchor'])
        balance['weighted_gradient_norms']['anchor'] = raw['anchor'] * balance['lambdas']['anchor']
    balance.update({'raw_losses': {k: float(v.detach()) for k, v in mean.items()},
                    'raw_gradient_norms': raw, 'adjustments': 1})
    return balance


def evaluate_loss(adapter, model, samples, lambdas):
    adapter.eval()
    values = []
    with torch.no_grad():
        for sample in samples:
            for c in range(4):
                values.append(float(total_loss(phase110_forward(adapter, model, sample, c)[0], lambdas)))
    return float(np.mean(values))


def train(adapter, model, train_samples, heldout_samples, lambdas, steps, lr):
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    assert_camera_optimizer_ownership(optimizer, adapter)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError('SAM/head parameters must stay frozen')
    by_cat = {c: [s for s in train_samples if s['desc']['category'] == c] for c in CATEGORIES}
    if any(not by_cat[c] for c in CATEGORIES):
        raise RuntimeError('balanced sampling requires every category')
    schedule = balanced_schedule(steps)
    counters = Counter()
    history, best, best_step = [], math.inf, 0
    for step, (category, condition) in enumerate(schedule, 1):
        sample = by_cat[category][counters[category] % len(by_cat[category])]
        counters[category] += 1
        adapter.train(); optimizer.zero_grad(set_to_none=True)
        terms, _ = phase110_forward(adapter, model, sample, condition)
        loss = total_loss(terms, lambdas)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite training loss at step {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
        if step % 40 == 0 or step == steps:
            held = evaluate_loss(adapter, model, heldout_samples, lambdas)
            row = {'step': step, 'heldout_total': held,
                   'train_total': float(loss.detach()),
                   'category': category, 'condition': CONDITIONS[condition]}
            history.append(row)
            if held < best:
                best, best_step = held, step
                torch.save(training_checkpoint(
                    adapter, optimizer, step, counters,
                    {'kind': 'best', 'best_step': step, 'best_heldout_total': best,
                     'lambdas': lambdas, 'total_steps': steps},
                ), OUT / 'best.pt')
            torch.save(training_checkpoint(
                adapter, optimizer, step, counters,
                {'kind': 'latest', 'best_step': best_step, 'best_heldout_total': best,
                 'lambdas': lambdas, 'total_steps': steps},
            ), OUT / 'latest.pt')
            atomic_json(OUT / 'status.json', {'state': 'training', 'step': step,
                        'best_step': best_step, 'best_heldout_total': best, 'pid': os.getpid()})
            print(json.dumps(row), flush=True)
    torch.save(training_checkpoint(
        adapter, optimizer, steps, counters,
        {'kind': 'final', 'best_step': best_step, 'best_heldout_total': best,
         'lambdas': lambdas, 'total_steps': steps},
    ), OUT / 'final.pt')
    return history, best_step, best


def _auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg): return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def _corr(a, b, rank=False):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    if rank:
        from scipy.stats import rankdata
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(adapter, model, samples):
    records = []
    for sample in samples:
        desc, meta = sample['desc'], sample['meta']
        for c, condition in enumerate(CONDITIONS):
            terms, out = phase110_forward(adapter, model, sample, c)
            raw_t = sample['raw'][c]['pred_cam_t'].cpu().numpy()
            raw_cam = sample['raw'][c]['pred_cam'].cpu().numpy()
            pred_t = out['pred_cam_t'].cpu().numpy(); pred_cam = out['pred_cam'].cpu().numpy()
            target = sample['target'][c]; t = target['t_all']
            raw_error = np.linalg.norm(raw_t - t, axis=1); error = np.linalg.norm(pred_t - t, axis=1)
            correction = np.linalg.norm(pred_t - raw_t, axis=1)
            raw_rep = frame_reprojection(sample['raw'][c]['pred_xy'].cpu(), meta['gt_xyc'], meta['size'])
            rep = frame_reprojection(out['pred_xy'].cpu(), meta['gt_xyc'], meta['size'])
            rel = out['reliability'].cpu().numpy()
            for i, frame in enumerate(meta['frame_id']):
                records.append({
                    'sequence': desc['sequence'], 'person_id': desc['person_id'],
                    'frame_id': int(frame), 'category': desc['category'], 'condition': condition,
                    'raw_error': float(raw_error[i]), 'v2_error': float(error[i]),
                    'raw_reprojection': float(raw_rep[i]), 'v2_reprojection': float(rep[i]),
                    'correction': float(correction[i]), 'reliability': float(rel[i]),
                    'need': float(target['need'][i]), 'w_target': float(target['weight'][i]),
                    'raw_z': float(raw_t[i, 2]), 'v2_z': float(pred_t[i, 2]),
                    'target_z': float(t[i, 2]), 'raw_scale': float(raw_cam[i, 0]),
                    'v2_scale': float(pred_cam[i, 0]),
                })
    grouped = {}
    for category in CATEGORIES:
        grouped[category] = {}
        for condition in CONDITIONS:
            rr = [r for r in records if r['category'] == category and r['condition'] == condition]
            def arr(k): return np.asarray([r[k] for r in rr])
            grouped[category][condition] = {
                'frames': len(rr), 'raw_camera_error': stats(arr('raw_error')),
                'v2_camera_error': stats(arr('v2_error')),
                'improvement_percent': float(100 * (arr('raw_error').mean() - arr('v2_error').mean()) / arr('raw_error').mean()),
                'raw_reprojection': stats(arr('raw_reprojection')),
                'v2_reprojection': stats(arr('v2_reprojection')),
                'correction': stats(arr('correction')), 'reliability': stats(arr('reliability')),
                'w_target': stats(arr('w_target')),
                'depth': {'raw_mean': float(arr('raw_z').mean()), 'v2_mean': float(arr('v2_z').mean()),
                          'target_mean': float(arr('target_z').mean()),
                          'v2_target_abs_error': float(np.abs(arr('v2_z') - arr('target_z')).mean()),
                          'raw_range': float(np.ptp(arr('raw_z'))), 'v2_range': float(np.ptp(arr('v2_z')))},
            }
    rel = np.asarray([r['reliability'] for r in records]); need = np.asarray([r['need'] for r in records])
    raw_error = np.asarray([r['raw_error'] for r in records]); correction = np.asarray([r['correction'] for r in records])
    sf = [r for r in records if r['category'] in ('stable', 'failure')]
    labels = np.asarray([1 if r['category'] == 'stable' else 0 for r in sf])
    scores = np.asarray([r['reliability'] for r in sf])
    ranking = {
        'reliability_vs_need_pearson': _corr(rel, need),
        'reliability_vs_need_spearman': _corr(rel, need, True),
        'stable_vs_failure_auroc': _auc(labels, scores),
        'correction_vs_raw_error_pearson': _corr(correction, raw_error),
        'correction_vs_raw_error_spearman': _corr(correction, raw_error, True),
        'reliability_mean': {c: float(np.mean([r['reliability'] for r in records if r['category'] == c])) for c in CATEGORIES},
        'correction_mean': {c: float(np.mean([r['correction'] for r in records if r['category'] == c])) for c in CATEGORIES},
    }
    stable = [r for r in records if r['category'] == 'stable']
    stable_delta = float(np.mean([r['v2_error'] - r['raw_error'] for r in stable]))
    stable_reproj_delta = float(np.mean([r['v2_reprojection'] - r['raw_reprojection'] for r in stable]))
    failure = [r for r in records if r['category'] == 'failure']
    failure_improve = float(np.mean([r['raw_error'] - r['v2_error'] for r in failure]))
    all_target_z = np.asarray([r['target_z'] for r in records]); all_v2_z = np.asarray([r['v2_z'] for r in records])
    depth_collapse = bool(np.std(all_v2_z) < 0.1 * np.std(all_target_z) and np.mean(np.abs(all_v2_z - all_target_z)) > 0.25)
    diagnostics = {'stable_camera_error_delta': stable_delta, 'stable_reprojection_delta': stable_reproj_delta,
                   'failure_absolute_improvement': failure_improve, 'depth_collapse': depth_collapse}
    return {'grouped': grouped, 'ranking': ranking, 'diagnostics': diagnostics, 'records': records}


def choose_verdict(evaluation):
    d, r = evaluation['diagnostics'], evaluation['ranking']
    stable_ok = d['stable_camera_error_delta'] <= 0.10 and d['stable_reprojection_delta'] <= 0.02
    failure_ok = d['failure_absolute_improvement'] > 0.20
    ranking_ok = ((r['stable_vs_failure_auroc'] or 0) >= 0.70 and
                  r['reliability_mean']['stable'] > r['reliability_mean']['failure'])
    if failure_ok and stable_ok and not d['depth_collapse']:
        return 'CAMERA_RELIABILITY_AND_PRESERVATION_CONFIRMED' if ranking_ok else 'CAMERA_RESTORATION_GENERALIZES_BUT_RELIABILITY_WEAK'
    if failure_ok and (not stable_ok or d['depth_collapse']):
        return 'CAMERA_PRESERVATION_FAILS'
    return 'MULTIWINDOW_GENERALIZATION_WEAK'


def run(steps=1600, lr=1e-3):
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); set_seed(); body_hash = sha256(BODY)
    (OUT / 'training.pid').write_text(f'{os.getpid()}\n')
    atomic_json(OUT / 'status.json', {'state': 'initializing', 'step': 0,
                'total_steps': steps, 'pid': os.getpid()})
    rows = read_phase18_rows(); thresholds = derive_quality_thresholds(rows)
    manifest = choose_manifest(build_window_pool(rows, thresholds))
    atomic_json(OUT / 'manifest.json', manifest)
    atomic_json(OUT / 'thresholds.json', thresholds)
    model = load_frozen_model()
    import smplx
    smpl_model = smplx.SMPL(model_path=str(SMPL_MODEL), batch_size=1).cuda().eval()
    train_samples = [prepare_window(x, model, smpl_model, thresholds) for x in manifest['train_windows']]
    heldout_samples = [prepare_window(x, model, smpl_model, thresholds) for x in manifest['heldout_windows']]
    assert_disjoint_tracks([s['desc'] for s in train_samples], [s['desc'] for s in heldout_samples])
    adapter = CameraFeatureTemporalAdapter().cuda()
    probe = locked_gradient_probe(); atomic_json(OUT / 'loss_scale_probe.json', probe)
    config = {'seed': SEED, 'steps': steps, 'lr': lr, 'window_length': WINDOW,
              'conditions': CONDITIONS, 'categories': CATEGORIES, 'thresholds': thresholds,
              'losses': ['cam3d', 'reproj', 'reliability', 'anchor'],
              'selection': 'aggregate held-out 3DPW-train-track total loss',
              'trainable': 'CameraFeatureTemporalAdapter only', 'body_sha256': body_hash}
    atomic_json(OUT / 'config.json', config)
    history, best_step, best = train(adapter, model, train_samples, heldout_samples,
                                     probe['lambdas'], steps, lr)
    atomic_json(OUT / 'history.json', {'history': history})
    best_adapter = load_adapter(OUT / 'best.pt')
    evaluation = evaluate(best_adapter, model, heldout_samples)
    verdict = choose_verdict(evaluation)
    summary = {'state': 'complete', 'verdict': verdict, 'elapsed_seconds': time.time() - started,
               'manifest_counts': {'train_windows': len(train_samples), 'heldout_windows': len(heldout_samples),
                                   'train_tracks': len(manifest['train_tracks']), 'heldout_tracks': len(manifest['heldout_tracks']),
                                   'train_categories': dict(Counter(s['desc']['category'] for s in train_samples)),
                                   'heldout_categories': dict(Counter(s['desc']['category'] for s in heldout_samples))},
               'thresholds': thresholds, 'gradient_probe': probe,
               'training': {'steps': steps, 'best_step': best_step, 'best_heldout_total': best},
               'heldout': evaluation, 'body_sha256': body_hash,
               'checkpoint': str(OUT / 'best.pt')}
    if sha256(BODY) != body_hash:
        raise RuntimeError('Body checkpoint changed')
    if not all(np.isfinite(r[k]) for r in evaluation['records'] for k in
               ('raw_error', 'v2_error', 'correction', 'reliability', 'raw_z', 'v2_z', 'target_z')):
        raise FloatingPointError('non-finite held-out output')
    atomic_json(OUT / 'summary.json', summary)
    atomic_json(OUT / 'status.json', {'state': 'complete', 'verdict': verdict,
                'best_step': best_step, 'pid': os.getpid()})
    print(json.dumps({'verdict': verdict, 'best_step': best_step,
                      'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args(); run(args.steps, args.lr)


if __name__ == '__main__':
    main()
