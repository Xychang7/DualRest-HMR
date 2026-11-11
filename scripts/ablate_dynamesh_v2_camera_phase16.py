"""One-window Camera Phase-1.6 loss-rebalancing ablation.

This runner reuses Phase-1 data and Phase-1.5 oracle outputs without rerunning
either phase.  It never modifies the production Camera or Body modules.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/root/sam-body4d-0.2.0')
OUT = ROOT/'outputs/dynamesh_v2_camera_phase16_loss_rebalancing'
PHASE1 = ROOT/'outputs/dynamesh_v2_camera_phase1'
PHASE15 = ROOT/'outputs/dynamesh_v2_camera_phase15_oracle'
SEED = 20260901

import sys
sys.path[:0] = [str(ROOT), str(ROOT/'models/sam_3d_body'), str(ROOT/'scripts')]

from diagnose_dynamesh_v2_camera_phase15 import gradient_norm, sha256
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase1 import (
    CONDITIONS,
    RATIOS,
    assert_camera_optimizer_ownership,
    camera_loss_terms,
    centred_innovation,
    frame_reprojection,
    load_frozen_model,
    masked_normalized_reprojection_loss,
    prepare_window,
    reliability_features,
    set_seed,
    soft_reliability_target,
    teacher_stability_weight,
    weighted_smooth_l1,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def derive_balanced_lambdas(raw_gradient_norms: dict[str, float]):
    """Derive one deterministic, non-searched balance from measured gradients."""
    required = {'teacher', 'reproj', 'reliability', 'anchor'}
    if set(raw_gradient_norms) != required or any(raw_gradient_norms[k] <= 0 for k in required):
        raise ValueError('all four raw gradient norms must be positive')
    reference = float(raw_gradient_norms['reproj'])
    targets = {'teacher': 1.5*reference, 'reproj': reference,
               'reliability': .25*reference, 'anchor': 1.5*reference}
    lambdas = {name: targets[name]/float(raw_gradient_norms[name]) for name in required}
    lambdas['reproj'] = 1.0
    weighted = {name: float(raw_gradient_norms[name])*lambdas[name] for name in required}
    return lambdas, weighted


def predicted_reliability_weight(reliability: torch.Tensor, stability: torch.Tensor):
    reliability, stability = torch.broadcast_tensors(reliability, stability)
    return reliability.detach().clamp(0, 1)*stability.detach().clamp(0, 1)


def apply_gate_mode(raw: torch.Tensor, delta: torch.Tensor, reliability: torch.Tensor, mode: str):
    if mode == 'learned':
        gate = reliability
    elif mode == 'fixed_open':
        gate = torch.zeros_like(reliability)
    else:
        raise ValueError('gate mode must be learned or fixed_open')
    return raw+(1.-gate)*delta


def adapter_forward(adapter, raw_pose_token, features, gate_mode='learned'):
    """Exact production architecture with a diagnostic-only gate override."""
    tokens = raw_pose_token if raw_pose_token.ndim == 3 else raw_pose_token.unsqueeze(0)
    values, mask = features.packed()
    if values.ndim == 2:
        values, mask = values.unsqueeze(0), mask.unsqueeze(0)
    condition = torch.cat((values.to(tokens), mask.to(tokens)), -1)
    reliability = torch.sigmoid(adapter.reliability_head(condition))
    latent = adapter.input_projection(adapter.input_norm(tokens))+adapter.condition_embedding(condition)
    forward = latent
    for layer in adapter.layers_fwd:
        forward = layer(forward)
    backward = torch.flip(latent, (1,))
    for layer in adapter.layers_bwd:
        backward = layer(backward)
    backward = torch.flip(backward, (1,))
    delta = adapter.output_projection(adapter.output_norm(forward+backward))
    return apply_gate_mode(tokens, delta, reliability, gate_mode), reliability


def projection(model, points, pred_cam, meta):
    n = len(pred_cam)
    return model.head_camera.perspective_projection(
        points, pred_cam,
        torch.as_tensor(meta['center'], device='cuda'),
        torch.as_tensor(meta['size'], device='cuda'),
        torch.as_tensor(meta['image_size'], device='cuda'),
        torch.as_tensor(np.broadcast_to(meta['K'], (n, 3, 3)).copy(), device='cuda'),
    )


def per_frame_raw_error(pred_xy, meta):
    gt = torch.as_tensor(meta['gt_xyc'], device='cuda')
    bbox = torch.as_tensor(meta['size'], device='cuda')
    return torch.stack([
        masked_normalized_reprojection_loss(pred_xy[i:i+1], gt[i:i+1], bbox[i:i+1])
        for i in range(len(pred_xy))
    ])


def observation_stability(features):
    center_delta = features.bbox_center_delta.squeeze(0)
    scale_delta = features.bbox_scale_delta.squeeze(0)
    return torch.exp(-10*(torch.linalg.vector_norm(center_delta, dim=-1)+torch.abs(scale_delta[:, 0]))).clamp(0, 1)


def camera_loss_terms_variant(adapter, model, tokens, meta, raw, reliability_scale, condition, variant):
    ratio = RATIOS[condition]
    selected = tokens[condition][None]
    features = reliability_features(selected, meta, ratio)
    gate_mode = 'fixed_open' if variant == 'C' else 'learned'
    refined, reliability = adapter_forward(adapter, selected, features, gate_mode)
    cam = model.head_camera(refined.squeeze(0), model.init_camera.weight.expand(len(refined.squeeze(0)), -1))
    projected = projection(model, raw[condition]['points3d'].detach(), cam, meta)
    gt = torch.as_tensor(meta['gt_xyc'], device='cuda')
    bbox = torch.as_tensor(meta['size'], device='cuda')
    raw_error = per_frame_raw_error(raw[condition]['pred_xy'].detach(), meta)
    target = soft_reliability_target(raw_error, reliability_scale)
    stability = observation_stability(features)
    if variant == 'A':
        clean_target = soft_reliability_target(per_frame_raw_error(raw[0]['pred_xy'].detach(), meta), reliability_scale)
        teacher_weight = teacher_stability_weight(clean_target, stability)
        anchor_weight = teacher_stability_weight(target, stability)
    elif variant in ('B', 'C'):
        predicted = reliability.squeeze(0).squeeze(-1)
        teacher_weight = predicted_reliability_weight(predicted, stability)
        anchor_weight = predicted_reliability_weight(predicted, stability)
    else:
        raise ValueError('variant must be A, B, or C')
    terms = {
        'teacher': weighted_smooth_l1(cam, raw[0]['pred_cam'].detach(), teacher_weight),
        'reproj': masked_normalized_reprojection_loss(projected['pred_keypoints_2d'], gt, bbox),
        'reliability': F.binary_cross_entropy(reliability.squeeze(0).squeeze(-1), target.detach()),
        'anchor': weighted_smooth_l1(cam, raw[condition]['pred_cam'].detach(), anchor_weight),
    }
    out = {'pred_cam': cam, 'pred_cam_t': projected['pred_cam_t'],
           'pred_xy': projected['pred_keypoints_2d'], 'reliability': reliability,
           'target': target, 'teacher_weight': teacher_weight, 'anchor_weight': anchor_weight}
    return terms, out


def weighted_total(terms, lambdas):
    return sum(float(lambdas[name])*terms[name] for name in lambdas)


def measure_phase1_gradients(adapter, model, tokens, meta, raw, scale):
    adapter.train()
    per_condition = [camera_loss_terms(adapter, model, tokens, meta, raw, scale, c)[0] for c in range(4)]
    averaged = {name: torch.stack([x[name] for x in per_condition]).mean()
                for name in ('teacher', 'reproj', 'reliability', 'anchor')}
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    return {name: gradient_norm(loss, parameters) for name, loss in averaged.items()}, {
        name: float(loss.detach()) for name, loss in averaged.items()
    }


def depth_scale_diagnostic(raw_cam, refined_cam, raw_cam_t, refined_cam_t, failure_index):
    def array(x):
        return torch.as_tensor(x).detach().cpu().float().numpy()
    raw_cam, refined_cam, raw_t, refined_t = map(array, (raw_cam, refined_cam, raw_cam_t, refined_cam_t))
    raw_z, new_z = raw_t[:, 2], refined_t[:, 2]
    raw_s, new_s = raw_cam[:, 0], refined_cam[:, 0]
    z_change = 100.*(new_z[failure_index]-raw_z[failure_index])/max(abs(raw_z[failure_index]), 1e-8)
    s_change = 100.*(new_s[failure_index]-raw_s[failure_index])/max(abs(raw_s[failure_index]), 1e-8)
    median_z_change = float(np.median(np.abs(new_z-raw_z)/np.maximum(np.abs(raw_z), 1e-8))*100.)
    median_s_change = float(np.median(np.abs(new_s-raw_s)/np.maximum(np.abs(raw_s), 1e-8))*100.)
    unsafe = bool(np.any(new_z <= 0) or abs(z_change) > 50. or median_z_change > 35. or median_s_change > 50.)
    return {
        'raw_failure_z': float(raw_z[failure_index]), 'refined_failure_z': float(new_z[failure_index]),
        'failure_z_change_percent': float(z_change), 'raw_z_range': [float(raw_z.min()), float(raw_z.max())],
        'refined_z_range': [float(new_z.min()), float(new_z.max())], 'median_abs_z_change_percent': median_z_change,
        'raw_failure_scale': float(raw_s[failure_index]), 'refined_failure_scale': float(new_s[failure_index]),
        'failure_scale_change_percent': float(s_change), 'raw_scale_range': [float(raw_s.min()), float(raw_s.max())],
        'refined_scale_range': [float(new_s.min()), float(new_s.max())], 'median_abs_scale_change_percent': median_s_change,
        'depth_scale_degeneracy': unsafe,
    }


def condition_metrics(out, raw_condition, meta, failure_index):
    raw_error = frame_reprojection(raw_condition['pred_xy'].detach().cpu(), meta['gt_xyc'], meta['size'])
    new_error = frame_reprojection(out['pred_xy'].detach().cpu(), meta['gt_xyc'], meta['size'])
    raw_cam = raw_condition['pred_cam'].detach().cpu().numpy()
    new_cam = out['pred_cam'].detach().cpu().numpy()
    raw_t = raw_condition['pred_cam_t'].detach().cpu().numpy()
    new_t = out['pred_cam_t'].detach().cpu().numpy()
    reliability = out['reliability'].detach().cpu().numpy().reshape(-1)
    neighbours = [failure_index-1, failure_index+1]
    stable = np.ones(len(raw_error), dtype=bool)
    stable[[failure_index, *neighbours]] = False
    safety = depth_scale_diagnostic(raw_cam, new_cam, raw_t, new_t, failure_index)
    result = {
        'baseline_mean_reprojection': float(np.mean(raw_error)), 'refined_mean_reprojection': float(np.mean(new_error)),
        'baseline_median_reprojection': float(np.median(raw_error)), 'refined_median_reprojection': float(np.median(new_error)),
        'baseline_p95_reprojection': float(np.percentile(raw_error, 95)), 'refined_p95_reprojection': float(np.percentile(new_error, 95)),
        'failure': {'frame_id': int(meta['frame_id'][failure_index]), 'baseline_reprojection': float(raw_error[failure_index]),
                    'refined_reprojection': float(new_error[failure_index]), 'raw_pred_cam': raw_cam[failure_index].tolist(),
                    'refined_pred_cam': new_cam[failure_index].tolist(), 'raw_pred_cam_t': raw_t[failure_index].tolist(),
                    'refined_pred_cam_t': new_t[failure_index].tolist(), 'reliability': float(reliability[failure_index])},
        'neighbours': [{'frame_id': int(meta['frame_id'][i]), 'baseline_reprojection': float(raw_error[i]),
                        'refined_reprojection': float(new_error[i]), 'pred_cam_deviation': float(np.linalg.norm(new_cam[i]-raw_cam[i])),
                        'reliability': float(reliability[i])} for i in neighbours],
        'camera_innovation_raw': float(np.mean(centred_innovation(raw_cam))),
        'camera_innovation_refined': float(np.mean(centred_innovation(new_cam))),
        'stable_frame_deviation': float(np.linalg.norm(new_cam[stable]-raw_cam[stable], axis=-1).mean()),
        'mean_reliability': float(np.mean(reliability)), 'depth_scale': safety,
        'per_frame_reprojection': new_error.tolist(),
    }
    result['failure_gain'] = (result['failure']['baseline_reprojection']-result['failure']['refined_reprojection'])/max(result['failure']['baseline_reprojection'], 1e-8)
    result['window_gain'] = (result['baseline_mean_reprojection']-result['refined_mean_reprojection'])/max(result['baseline_mean_reprojection'], 1e-8)
    result['preserved'] = bool(max(x['refined_reprojection']-x['baseline_reprojection'] for x in result['neighbours']) < .01 and result['stable_frame_deviation'] < .05)
    result['safe'] = not safety['depth_scale_degeneracy']
    return result


@torch.no_grad()
def evaluate_variant(adapter, variant, model, tokens, meta, raw, scale, failure_index):
    adapter.eval()
    result = {}
    for condition, name in enumerate(CONDITIONS):
        _, out = camera_loss_terms_variant(adapter, model, tokens, meta, raw, scale, condition, variant)
        result[name] = condition_metrics(out, raw[condition], meta, failure_index)
    return result


def train_variant(variant, lambdas, model, tokens, meta, raw, scale, failure_index, steps, lr):
    set_seed()
    directory = OUT/f'exp_{variant.lower()}'
    directory.mkdir(parents=True, exist_ok=True)
    adapter = CameraFeatureTemporalAdapter().cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    assert_camera_optimizer_ownership(optimizer, adapter)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError('frozen SAM unexpectedly trainable')
    history, best = [], math.inf
    for step in range(1, steps+1):
        adapter.train(); condition = (step-1) % 4
        optimizer.zero_grad(set_to_none=True)
        terms, _ = camera_loss_terms_variant(adapter, model, tokens, meta, raw, scale, condition, variant)
        total = weighted_total(terms, lambdas)
        if not torch.isfinite(total):
            raise FloatingPointError(f'Exp-{variant} non-finite at step {step}')
        total.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
        item = {'step': step, 'condition': CONDITIONS[condition], 'total': float(total.detach()),
                **{name: float(value.detach()) for name, value in terms.items()}}
        history.append(item)
        if step % 20 == 0:
            adapter.eval()
            with torch.no_grad():
                four = float(np.mean([float(weighted_total(camera_loss_terms_variant(adapter, model, tokens, meta, raw, scale, c, variant)[0], lambdas)) for c in range(4)]))
            item['four_condition_total'] = four
            if four < best:
                best = four
                torch.save({'adapter_state_dict': adapter.state_dict(), 'variant': variant,
                            'best_step': step, 'best_total': best, 'lambdas': lambdas}, directory/'best.pt')
            atomic_json(OUT/'status.json', {'state': 'running', 'current_experiment': variant,
                        'current_step': step, 'steps_per_experiment': steps})
            print(json.dumps({'experiment': variant, **item}), flush=True)
    torch.save({'adapter_state_dict': adapter.state_dict(), 'variant': variant,
                'final_step': steps, 'lambdas': lambdas}, directory/'final.pt')
    atomic_json(directory/'history.json', {'history': history})
    checkpoint = torch.load(directory/'best.pt', map_location='cuda', weights_only=False)
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    metrics = evaluate_variant(adapter, variant, model, tokens, meta, raw, scale, failure_index)
    atomic_json(directory/'metrics.json', metrics)
    return {'best_step': checkpoint['best_step'], 'best_total': checkpoint['best_total'], 'metrics': metrics}


def select_verdict(exp_a, exp_b, exp_c):
    candidates = [exp_a, exp_b, exp_c]
    if any((x['failure_gain'] > .10 or x['window_gain'] > .05) and not x['safe'] for x in candidates):
        return 'REPROJECTION_CAUSES_DEPTH_DEGENERACY'
    if exp_c['safe'] and (exp_c['window_gain']-exp_b['window_gain'] > .05 or exp_c['failure_gain']-exp_b['failure_gain'] > .10):
        return 'RELIABILITY_GATING_BOTTLENECK'
    if any(x['safe'] and x['preserved'] and x['failure_gain'] > .10 and x['window_gain'] > .02 for x in (exp_a, exp_b)):
        return 'LOSS_REBALANCING_CONFIRMED'
    return 'CAMERA_RECOVERY_STILL_WEAK'


def run(steps=400, lr=1e-3):
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    if not (PHASE1/'best.pt').exists() or not (PHASE15/'summary.json').exists():
        raise FileNotFoundError('completed Phase-1/1.5 artifacts are required')
    if (OUT/'summary.json').exists():
        existing = json.loads((OUT/'summary.json').read_text())
        if existing.get('state') == 'complete':
            print(json.dumps({'state': 'already_complete', 'summary': str(OUT/'summary.json')})); return existing
    atomic_json(OUT/'status.json', {'state': 'initializing', 'pid': os.getpid(), 'started_at': started})
    mining = json.loads((PHASE1/'mining.json').read_text())
    model = load_frozen_model()
    _, meta, tokens, _, raw, scale = prepare_window(model, mining)
    expected_frames = list(range(506, 538, 2))
    if meta['sequence'] != 'courtyard_basketball_00' or meta['person'] != 1 or meta['frame_id'].astype(int).tolist() != expected_frames:
        raise RuntimeError('fixed Phase-1.6 window contract mismatch')
    failure_index = int(np.where(meta['frame_id'] == 522)[0][0])
    phase1_adapter = CameraFeatureTemporalAdapter().cuda()
    phase1_checkpoint = torch.load(PHASE1/'best.pt', map_location='cuda', weights_only=False)
    phase1_adapter.load_state_dict(phase1_checkpoint['adapter_state_dict'])
    raw_gradients, raw_losses = measure_phase1_gradients(phase1_adapter, model, tokens, meta, raw, scale)
    lambdas, weighted = derive_balanced_lambdas(raw_gradients)
    core = [weighted[x] for x in ('teacher', 'reproj', 'anchor')]
    balance = {'source': 'Phase-1 best checkpoint, four-condition mean', 'raw_loss': raw_losses,
               'raw_gradient_norm': raw_gradients, 'lambda': lambdas, 'weighted_gradient_norm': weighted,
               'core_max_min_ratio': max(core)/min(core)}
    atomic_json(OUT/'gradient_balance.json', balance)
    body_checkpoint = ROOT/'outputs/dynamesh_v2_phase2_training/best.pt'
    body_hash = sha256(body_checkpoint)
    config = {'seed': SEED, 'steps_per_experiment': steps, 'lr': lr, 'optimizer': 'AdamW',
              'weight_decay': 1e-4, 'sequence': meta['sequence'], 'person_id': meta['person'],
              'frame_ids': expected_frames, 'failure_frame_id': 522, 'conditions': list(CONDITIONS),
              'lambdas': lambdas, 'phase1_checkpoint': str(PHASE1/'best.pt'),
              'phase15_summary': str(PHASE15/'summary.json'), 'body_checkpoint_sha256': body_hash,
              'full_training': False, 'architecture_changed': False}
    atomic_json(OUT/'config.json', config)
    experiments = {}
    for variant in ('A', 'B', 'C'):
        experiments[variant] = train_variant(variant, lambdas, model, tokens, meta, raw, scale, failure_index, steps, lr)
    phase1_summary = json.loads((PHASE1/'summary.json').read_text())
    oracle = json.loads((PHASE15/'summary.json').read_text())
    clean = {name: experiments[name]['metrics']['clean'] for name in ('A', 'B', 'C')}
    verdict = select_verdict(clean['A'], clean['B'], clean['C'])
    summary = {'state': 'complete', 'elapsed_seconds': time.time()-started, 'gradient_balance': balance,
               'experiments': experiments, 'references': {'phase1': phase1_summary, 'phase15': oracle},
               'verdict': verdict, 'body_checkpoint_sha256': body_hash}
    if sha256(body_checkpoint) != body_hash:
        raise RuntimeError('Body checkpoint changed')
    atomic_json(OUT/'summary.json', summary)
    atomic_json(OUT/'status.json', {'state': 'complete', 'pid': os.getpid(), 'elapsed_seconds': summary['elapsed_seconds'],
                'completed_experiments': ['A', 'B', 'C'], 'verdict': verdict})
    print(json.dumps({'state': 'complete', 'verdict': verdict, 'elapsed_seconds': summary['elapsed_seconds']}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    run(args.steps, args.lr)


if __name__ == '__main__':
    main()
