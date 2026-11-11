"""Engineering-only strict resume of Phase-1.10 from step760 with stable norms."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import train_dynamesh_v2_camera_phase110 as phase110
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase19 import load_adapter, load_frozen_model
from validate_dynamesh_v2_camera_multijoint_gt import SMPL_MODEL

ROOT = phase110.ROOT
SOURCE = phase110.OUT
OUT = ROOT / 'outputs/dynamesh_v2_camera_phase110_debug_resume_step760'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


def grad_stats(adapter):
    sq = 0.0; maximum = 0.0; first_bad = None
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None: continue
        if not torch.isfinite(parameter.grad).all() and first_bad is None: first_bad = name
        if torch.isfinite(parameter.grad).all():
            sq += float(torch.sum(parameter.grad.detach().float() ** 2))
            maximum = max(maximum, float(parameter.grad.detach().abs().max()))
    return math.sqrt(sq), maximum, first_bad


def max_parameter(adapter):
    first_bad = next((n for n, p in adapter.named_parameters() if not torch.isfinite(p).all()), None)
    maximum = max(float(p.detach().abs().max()) for p in adapter.parameters())
    return maximum, first_bad


def optimizer_finite(optimizer):
    first_bad = None; max_avg = 0.0; max_sq = 0.0
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state.get(key)
            if not torch.is_tensor(value): continue
            if not torch.isfinite(value).all() and first_bad is None: first_bad = key
            if torch.isfinite(value).all():
                if key == 'exp_avg': max_avg = max(max_avg, float(value.abs().max()))
                else: max_sq = max(max_sq, float(value.abs().max()))
    return first_bad, max_avg, max_sq


def run():
    # The nohup launcher creates OUT first so it can redirect stdout/stderr to
    # OUT/run.log before this Python process starts.
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'training.pid').write_text(f'{os.getpid()}\n')
    atomic_json(OUT / 'status.json', {'state': 'initializing', 'current_step': 760,
                'target_step': 1600, 'pid': os.getpid(),
                'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX'})
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    thresholds = json.loads((SOURCE / 'thresholds.json').read_text())
    config = json.loads((SOURCE / 'config.json').read_text())
    checkpoint = torch.load(SOURCE / 'latest.pt', map_location='cuda', weights_only=False)
    if checkpoint['current_step'] != 760: raise ValueError('source checkpoint must be step760')
    for filename in ('manifest.json', 'thresholds.json', 'config.json', 'loss_scale_probe.json'):
        shutil.copy2(SOURCE / filename, OUT / filename)
    model = load_frozen_model()
    import smplx
    smpl_model = smplx.SMPL(model_path=str(SMPL_MODEL), batch_size=1).cuda().eval()
    train_samples = [phase110.prepare_window(x, model, smpl_model, thresholds) for x in manifest['train_windows']]
    heldout_samples = [phase110.prepare_window(x, model, smpl_model, thresholds) for x in manifest['heldout_windows']]
    adapter = CameraFeatureTemporalAdapter().cuda(); adapter.load_state_dict(checkpoint['adapter_state_dict'], strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['lr'], weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']); restore_rng(checkpoint['rng_state'])
    counters = Counter({k: int(v) for k, v in checkpoint['sampler_state']['counters'].items()})
    schedule = phase110.balanced_schedule(config['steps'])
    expected = Counter(category for category, _ in schedule[:760])
    if any(counters[c] != expected[c] for c in phase110.CATEGORIES): raise RuntimeError('sampler state mismatch')
    by_category = {c: [s for s in train_samples if s['desc']['category'] == c] for c in phase110.CATEGORIES}
    lambdas = checkpoint['metadata']['lambdas']
    prior_best = torch.load(SOURCE / 'best.pt', map_location='cpu', weights_only=False)
    best = float(checkpoint['metadata']['best_heldout_total']); best_step = int(checkpoint['metadata']['best_step'])
    torch.save(prior_best, OUT / 'best.pt')
    history = []; maxima = {'activation': 0.0, 'grad_norm': 0.0, 'grad_abs': 0.0,
                            'parameter_abs': 0.0, 'exp_avg_abs': 0.0, 'exp_avg_sq_abs': 0.0}
    for step in range(761, 1601):
        category, condition = schedule[step - 1]
        sample = by_category[category][counters[category] % len(by_category[category])]; counters[category] += 1
        adapter.train(); optimizer.zero_grad(set_to_none=True); diagnostics = {}
        terms, _ = phase110.phase110_forward(adapter, model, sample, condition, diagnostics)
        loss = phase110.total_loss(terms, lambdas)
        if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss at step {step}: {terms}')
        loss.backward(); grad_norm, grad_abs, bad_grad = grad_stats(adapter)
        if bad_grad: raise FloatingPointError(f'non-finite gradient at step {step}: {bad_grad}')
        maxima['activation'] = max(maxima['activation'], *[v for k, v in diagnostics.items() if k.endswith('max_abs')])
        maxima['grad_norm'] = max(maxima['grad_norm'], grad_norm); maxima['grad_abs'] = max(maxima['grad_abs'], grad_abs)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
        param_abs, bad_param = max_parameter(adapter); bad_opt, avg, sq = optimizer_finite(optimizer)
        if bad_param or bad_opt: raise FloatingPointError(f'non-finite state at step {step}: {bad_param or bad_opt}')
        maxima['parameter_abs'] = max(maxima['parameter_abs'], param_abs)
        maxima['exp_avg_abs'] = max(maxima['exp_avg_abs'], avg); maxima['exp_avg_sq_abs'] = max(maxima['exp_avg_sq_abs'], sq)
        if step % 40 == 0:
            heldout = phase110.evaluate_loss(adapter, model, heldout_samples, lambdas)
            history.append({'step': step, 'heldout_total': heldout, 'train_total': float(loss.detach()),
                            'category': category, 'condition': phase110.CONDITIONS[condition]})
            if heldout < best:
                best, best_step = heldout, step
                torch.save(phase110.training_checkpoint(adapter, optimizer, step, counters,
                    {'kind': 'best', 'best_step': step, 'best_heldout_total': best,
                     'lambdas': lambdas, 'total_steps': 1600,
                     'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX'}), OUT / 'best.pt')
            torch.save(phase110.training_checkpoint(adapter, optimizer, step, counters,
                {'kind': 'latest', 'best_step': best_step, 'best_heldout_total': best,
                 'lambdas': lambdas, 'total_steps': 1600,
                 'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX'}), OUT / 'latest.pt')
            atomic_json(OUT / 'status.json', {'state': 'training', 'current_step': step,
                        'target_step': 1600, 'best_step': best_step, 'pid': os.getpid(),
                        'maxima': maxima, 'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX'})
            print(json.dumps(history[-1]), flush=True)
    torch.save(phase110.training_checkpoint(adapter, optimizer, 1600, counters,
        {'kind': 'final', 'best_step': best_step, 'best_heldout_total': best,
         'lambdas': lambdas, 'total_steps': 1600,
         'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX'}), OUT / 'final.pt')
    atomic_json(OUT / 'history.json', {'history': history})
    best_adapter = load_adapter(OUT / 'best.pt'); evaluation = phase110.evaluate(best_adapter, model, heldout_samples)
    summary = {'state': 'complete', 'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX',
               'source_step': 760, 'final_step': 1600, 'nonfinite_detected': False,
               'best_step': best_step, 'best_heldout_total': best, 'maxima': maxima,
               'heldout_preliminary': evaluation, 'elapsed_seconds': time.time() - started,
               'body_sha256': phase110.sha256(phase110.BODY)}
    atomic_json(OUT / 'summary.json', summary)
    atomic_json(OUT / 'status.json', {'state': 'complete', 'current_step': 1600,
                'best_step': best_step, 'nonfinite_detected': False, 'pid': os.getpid(),
                'result_type': 'DEBUG_RESUME_WITH_ARCHITECTURE_FIX'})
    print(json.dumps({'state': 'complete', 'best_step': best_step, 'maxima': maxima}), flush=True)


if __name__ == '__main__': run()
