"""One-window Camera Phase-1.5 oracle capacity diagnosis."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/root/sam-body4d-0.2.0'); OUT=ROOT/'outputs/dynamesh_v2_camera_phase15_oracle'
sys.path[:0]=[str(ROOT),str(ROOT/'models/sam_3d_body'),str(ROOT/'scripts')]

from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase1 import (
    CONDITIONS,LAMBDAS,OUT as PHASE1_OUT,camera_loss_terms,centred_innovation,
    evaluate_adapter,frame_reprojection,load_frozen_model,masked_normalized_reprojection_loss,
    prepare_window,set_seed,
)


def clone_independent_condition_variables(raw):
    return [torch.nn.Parameter(raw[index].detach().clone()) for index in range(raw.shape[0])]


def gradient_norm(loss,parameters):
    grads=torch.autograd.grad(loss,list(parameters),retain_graph=True,allow_unused=True)
    total=torch.zeros((),device=loss.device)
    for grad in grads:
        if grad is not None: total=total+grad.detach().square().sum()
    return float(torch.sqrt(total))


def weighted_gradient_report(name,loss,parameters,weight):
    raw=gradient_norm(loss,parameters)
    return {'loss':float(loss.detach()),'raw_gradient_norm':raw,'lambda':float(weight),'weighted_gradient_norm':raw*float(weight)}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def project(model,points,pred_cam,meta):
    n=len(pred_cam); device=pred_cam.device
    return model.head_camera.perspective_projection(
        points,pred_cam,
        torch.as_tensor(meta['center'],device=device),
        torch.as_tensor(meta['size'],device=device),
        torch.as_tensor(meta['image_size'],device=device),
        torch.as_tensor(np.broadcast_to(meta['K'],(n,3,3)).copy(),device=device),
    )


def optimize_oracle(model,points,raw_cam,raw_token,meta,kind,steps=400,lr=1e-2):
    if kind=='pred_cam': variable=torch.nn.Parameter(raw_cam.detach().clone())
    elif kind=='token': variable=torch.nn.Parameter(raw_token.detach().clone())
    else: raise ValueError('kind must be pred_cam or token')
    optimizer=torch.optim.Adam([variable],lr=lr); gt=torch.as_tensor(meta['gt_xyc'],device='cuda'); bbox=torch.as_tensor(meta['size'],device='cuda')
    best=math.inf; best_value=None; history=[]
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        cam=variable if kind=='pred_cam' else model.head_camera(variable,model.init_camera.weight.expand(len(variable),-1))
        out=project(model,points,cam,meta); loss=masked_normalized_reprojection_loss(out['pred_keypoints_2d'],gt,bbox)
        if not torch.isfinite(loss): raise FloatingPointError(f'{kind} oracle non-finite at step {step}')
        loss.backward(); optimizer.step(); value=float(loss.detach()); history.append(value)
        if value<best: best=value; best_value=variable.detach().clone()
    with torch.no_grad():
        cam=best_value if kind=='pred_cam' else model.head_camera(best_value,model.init_camera.weight.expand(len(best_value),-1)); out=project(model,points,cam,meta)
    return {'kind':kind,'best_loss':best,'history':history,'variable':best_value,'pred_cam':cam.detach(),'pred_cam_t':out['pred_cam_t'].detach(),'pred_xy':out['pred_keypoints_2d'].detach()}


def method_metrics(pred_xy,pred_cam,pred_cam_t,meta,center):
    error=frame_reprojection(pred_xy.detach().cpu(),meta['gt_xyc'],meta['size']); camera=pred_cam.detach().cpu().numpy()
    return {'failure_reprojection':float(error[center]),'mean_reprojection':float(np.mean(error)),'p95_reprojection':float(np.percentile(error,95)),'camera_innovation_mean':float(np.mean(centred_innovation(camera))),'failure_pred_cam':camera[center].tolist(),'failure_pred_cam_t':pred_cam_t.detach().cpu().numpy()[center].tolist()}


def gradient_decomposition(adapter,model,tokens,meta,raw,scale):
    adapter.train(); per_condition=[camera_loss_terms(adapter,model,tokens,meta,raw,scale,c)[0] for c in range(4)]
    averaged={name:torch.stack([terms[name] for terms in per_condition]).mean() for name in LAMBDAS}
    params=[p for p in adapter.parameters() if p.requires_grad]
    return {name:weighted_gradient_report(name,averaged[name],params,LAMBDAS[name]) for name in LAMBDAS}


def run(steps=400,lr=1e-2):
    set_seed(); OUT.mkdir(parents=True,exist_ok=True); started=time.time()
    mining=json.loads((PHASE1_OUT/'mining.json').read_text()); model=load_frozen_model(); _,meta,tokens,_,raw,scale=prepare_window(model,mining); center=int(np.where(meta['frame_id']==mining['selected']['frame_id'])[0][0])
    # Explicitly clone all condition variables before optimization to prove no storage sharing.
    direct_variables=clone_independent_condition_variables(torch.stack([x['pred_cam'].detach() for x in raw])); token_variables=clone_independent_condition_variables(tokens)
    results={}; saved={'direct':{},'token':{}}
    phase1_adapter=CameraFeatureTemporalAdapter().cuda(); checkpoint=torch.load(PHASE1_OUT/'best.pt',map_location='cuda',weights_only=False); phase1_adapter.load_state_dict(checkpoint['adapter_state_dict']); current=evaluate_adapter(phase1_adapter,model,tokens,meta,raw,scale)
    for condition,name in enumerate(CONDITIONS):
        # Variables are reconstructed from the corresponding condition only.
        direct=optimize_oracle(model,raw[condition]['points3d'],direct_variables[condition],token_variables[condition],meta,'pred_cam',steps,lr)
        free=optimize_oracle(model,raw[condition]['points3d'],direct_variables[condition],token_variables[condition],meta,'token',steps,lr)
        raw_metrics=method_metrics(raw[condition]['pred_xy'],raw[condition]['pred_cam'],raw[condition]['pred_cam_t'],meta,center)
        results[name]={'raw_sam':raw_metrics,'direct_pred_cam_oracle':method_metrics(direct['pred_xy'],direct['pred_cam'],direct['pred_cam_t'],meta,center),'free_token_oracle':method_metrics(free['pred_xy'],free['pred_cam'],free['pred_cam_t'],meta,center),'camera_v0':{'failure_reprojection':current[name]['failure']['refined_reprojection'],'mean_reprojection':current[name]['refined_mean_reprojection'],'p95_reprojection':current[name]['refined_p95_reprojection'],'camera_innovation_mean':current[name]['refined_camera_innovation_mean'],'failure_pred_cam':current[name]['failure']['refined_pred_cam'],'failure_pred_cam_t':current[name]['failure']['refined_pred_cam_t']},'direct_history':direct['history'],'token_history':free['history']}
        saved['direct'][name]=direct['variable'].cpu(); saved['token'][name]=free['variable'].cpu()
        print(json.dumps({'condition':name,'raw':raw_metrics['mean_reprojection'],'direct':results[name]['direct_pred_cam_oracle']['mean_reprojection'],'token':results[name]['free_token_oracle']['mean_reprojection']}),flush=True)
    gradients=gradient_decomposition(phase1_adapter,model,tokens,meta,raw,scale)
    clean=results['clean']; raw_failure=clean['raw_sam']['failure_reprojection']; direct_gain=(raw_failure-clean['direct_pred_cam_oracle']['failure_reprojection'])/raw_failure; token_gain=(raw_failure-clean['free_token_oracle']['failure_reprojection'])/raw_failure; v0_gain=(raw_failure-clean['camera_v0']['failure_reprojection'])/raw_failure
    weighted=[v['weighted_gradient_norm'] for v in gradients.values() if v['weighted_gradient_norm']>0]; imbalance=max(weighted)/min(weighted) if weighted else math.inf
    if direct_gain>.20 and token_gain>.20: diagnosis='ADAPTER_GATING_OR_LOSS_BOTTLENECK'
    elif direct_gain>.20 and token_gain<=.20: diagnosis='FROZEN_PERSPECTIVE_TOKEN_CONTROLLABILITY_BOTTLENECK'
    elif direct_gain<=.05: diagnosis='REPROJECTION_OBJECTIVE_OR_DATA_CAPACITY_LIMIT'
    else: diagnosis='MIXED_CAPACITY_LIMIT'
    summary={'state':'complete','window':{'sequence':meta['sequence'],'person_id':meta['person'],'frame_ids':meta['frame_id'].astype(int).tolist(),'failure_frame_id':int(meta['frame_id'][center])},'protocol':{'conditions_independent':True,'own_condition_initialization':True,'loss':'12-joint normalized L_reproj only','steps':steps,'lr':lr,'optimizer':'Adam','stopping':'fixed steps; best full-window reprojection','full_training':False},'results':results,'gradient_decomposition':gradients,'clean_failure_relative_gain':{'direct_pred_cam_oracle':direct_gain,'free_token_oracle':token_gain,'camera_v0':v0_gain},'weighted_gradient_imbalance_ratio':imbalance,'diagnosis':diagnosis,'optional_gate_zero_run':False,'body_checkpoint_sha256':sha256(ROOT/'outputs/dynamesh_v2_phase2_training/best.pt'),'elapsed_seconds':time.time()-started}
    torch.save(saved,OUT/'oracle_variables.pt'); (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps({'state':'complete','diagnosis':diagnosis,'clean_failure_gain':summary['clean_failure_relative_gain'],'gradient_imbalance':imbalance},indent=2)); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--steps',type=int,default=400); parser.add_argument('--lr',type=float,default=1e-2); args=parser.parse_args(); run(args.steps,args.lr)


if __name__=='__main__': main()

