"""Phase-1.12 frozen frame-wise calibration plus temporal residual ablation."""
from __future__ import annotations
import json,math,os,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import train_dynamesh_v2_camera_phase110 as p
import ablate_dynamesh_v2_camera_phase111 as q
from resume_dynamesh_v2_camera_phase110_debug import atomic_json,grad_stats,max_parameter,optimizer_finite
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase19 import load_frozen_model
from train_dynamesh_v2_camera_phase1 import masked_normalized_reprojection_loss
from validate_dynamesh_v2_camera_multijoint_gt import SMPL_MODEL

ROOT=p.ROOT; P111=ROOT/'outputs/dynamesh_v2_camera_phase111_temporal_ablation'; REF=p.ROOT/'outputs/dynamesh_v2_camera_phase110_multiwindow_stable_norm'
OUT=ROOT/'outputs/dynamesh_v2_camera_phase112_temporal_residual'; CONDITIONS=p.CONDITIONS

def pair_weights(w): return torch.minimum(w[1:],w[:-1])
def velocity_loss(pred,target,w):
 d=(pred[1:]-pred[:-1])-(target[1:]-target[:-1]); pw=pair_weights(w); return (F.smooth_l1_loss(d,target=torch.zeros_like(d),reduction='none').mean(-1)*pw).sum()/pw.sum().clamp_min(1e-8)
def acceleration_loss(pred,target,w):
 d=(pred[2:]-2*pred[1:-1]+pred[:-2])-(target[2:]-2*target[1:-1]+target[:-2]); tw=torch.minimum(torch.minimum(w[2:],w[1:-1]),w[:-2]);return(F.smooth_l1_loss(d,torch.zeros_like(d),reduction='none').mean(-1)*tw).sum()/tw.sum().clamp_min(1e-8)
def reject_split(x):
 if x not in ('train','3dpw_train_only'):raise ValueError('train only')
def assert_inference_fields(fields):
 bad=set(fields)&{'t_all','weight','gt_2d','gt_3d','camera_error'}
 if bad:raise ValueError(f'inference GT leak: {bad}')
def assert_temporal_optimizer_ownership(opt,temporal,framewise):
 expected={id(x) for x in temporal.parameters()};actual={id(x) for g in opt.param_groups for x in g['params']}
 if actual!=expected or any(x.requires_grad for x in framewise.parameters()):raise AssertionError('optimizer/freeze boundary violation')
def temporal_residual_forward(adapter,tokens,features):
 if tokens.device.type!='cuda' and torch.count_nonzero(adapter.output_projection.weight)==0 and torch.count_nonzero(adapter.output_projection.bias)==0:return tokens
 values,mask=features.packed();
 if values.ndim==2:values,mask=values.unsqueeze(0),mask.unsqueeze(0)
 condition=torch.cat((values.to(tokens),mask.to(tokens)),-1);latent=adapter.input_projection(adapter.input_norm(tokens))+adapter.condition_embedding(condition)
 fw=p.stable_mamba_stack(adapter.layers_fwd,latent);bw=torch.flip(p.stable_mamba_stack(adapter.layers_bwd,torch.flip(latent,(1,))),(1,))
 delta=adapter.output_projection(p.stable_output_layer_norm(adapter,fw+bw));return tokens+delta
def trajectory_diagnostics(pred,target):
 pred=np.asarray(pred);target=np.asarray(target);pv=np.diff(pred,axis=0);tv=np.diff(target,axis=0);pa=np.diff(pred,n=2,axis=0);ta=np.diff(target,n=2,axis=0)
 ve=np.linalg.norm(pv-tv,axis=1);ae=np.linalg.norm(pa-ta,axis=1)
 return{'velocity_mean':float(ve.mean()) if len(ve) else 0.,'velocity_p95':float(np.percentile(ve,95)) if len(ve) else 0.,'acceleration_mean':float(ae.mean()) if len(ae) else 0.,'acceleration_p95':float(np.percentile(ae,95)) if len(ae) else 0.,'max_local_jump_error':float(ve.max()) if len(ve) else 0.,'pred_velocity_magnitude':float(np.linalg.norm(pv,axis=1).mean()) if len(pv) else 0.,'gt_velocity_magnitude':float(np.linalg.norm(tv,axis=1).mean()) if len(tv) else 0.}

def load_framewise():
 m=q.FramewiseLatentAdapter().cuda();ck=torch.load(P111/'framewise/best.pt',map_location='cuda',weights_only=False);m.load_state_dict(ck['adapter_state_dict']);m.eval().requires_grad_(False);return m
def framewise_forward(framewise,model,s,c):
 with torch.no_grad():token=framewise(s['tokens'][c][None]);cam=model.head_camera(token.squeeze(0),model.init_camera.weight.expand(16,-1));proj=p.projection(model,s['raw'][c]['points3d'],cam,s['meta'])
 return token,{'pred_cam':cam,'pred_cam_t':proj['pred_cam_t'],'pred_xy':proj['pred_keypoints_2d']}
def temporal_forward(adapter,framewise,model,s,c):
 base,_=framewise_forward(framewise,model,s,c);feat=p.reliability_features(base,s['meta'],p.RATIOS[c]);assert_inference_fields({'visibility','bbox_center_delta','bbox_scale_delta','temporal_latent_disagreement'})
 token=temporal_residual_forward(adapter,base,feat);cam=model.head_camera(token.squeeze(0),model.init_camera.weight.expand(16,-1));proj=p.projection(model,s['raw'][c]['points3d'],cam,s['meta']);return{'pred_cam':cam,'pred_cam_t':proj['pred_cam_t'],'pred_xy':proj['pred_keypoints_2d'],'token':token}
def loss_terms(out,base,s,c):
 tar=s['target'][c];t=torch.as_tensor(tar['t_all'],device='cuda',dtype=torch.float32);w=torch.as_tensor(tar['weight'],device='cuda',dtype=torch.float32)
 # Reliable targets anchor to t_all; unreliable ones anchor to the frozen frame-wise prediction.
 abs_target=w[:,None]*t+(1-w[:,None])*base['pred_cam_t'].detach()
 absolute=p.weighted_cam3d_loss(out['pred_cam_t'],abs_target,torch.ones_like(w));gt=torch.as_tensor(s['meta']['gt_xyc'],device='cuda');bbox=torch.as_tensor(s['meta']['size'],device='cuda')
 return{'velocity':velocity_loss(out['pred_cam_t'],t,w),'acceleration':acceleration_loss(out['pred_cam_t'],t,w),'absolute':absolute,'reproj':masked_normalized_reprojection_loss(out['pred_xy'],gt,bbox)}
def total(terms,l):return sum(l[k]*terms[k] for k in l)
def gradient_probe(adapter,framewise,model,samples):
 vals=defaultdict(list)
 for i,cat in enumerate(p.CATEGORIES):
  s=next(x for x in samples if x['desc']['category']==cat);_,base=framewise_forward(framewise,model,s,i);out=temporal_forward(adapter,framewise,model,s,i);tt=loss_terms(out,base,s,i)
  for k,v in tt.items():vals[k].append(v)
 mean={k:torch.stack(v).mean() for k,v in vals.items()};params=[x for x in adapter.parameters() if x.requires_grad];raw={k:p.gradient_norm(v,params) for k,v in mean.items()};ref=raw['velocity'];targets={'velocity':ref,'acceleration':.5*ref,'absolute':ref,'reproj':.5*ref};l={k:targets[k]/max(raw[k],1e-12) for k in raw};return{'raw_loss':{k:float(v) for k,v in mean.items()},'raw_grad':raw,'lambdas':l,'weighted_grad':{k:raw[k]*l[k] for k in raw}}
def eval_objective(adapter,framewise,model,samples,l):
 x=[];adapter.eval()
 with torch.no_grad():
  for s in samples:
   for c in range(4):_,base=framewise_forward(framewise,model,s,c);x.append(float(total(loss_terms(temporal_forward(adapter,framewise,model,s,c),base,s,c),l)))
 return float(np.mean(x))
def checkpoint(a,o,step,counters,best,beststep,l,kind):return p.training_checkpoint(a,o,step,counters,{'kind':kind,'best_heldout':best,'best_step':beststep,'lambdas':l,'total_steps':1600})
def train_variant(name,a,framewise,model,train,held,l):
 folder=OUT/name;folder.mkdir();o=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=1e-4);assert_temporal_optimizer_ownership(o,a,framewise);cats={c:[s for s in train if s['desc']['category']==c] for c in p.CATEGORIES};cnt=Counter();best=math.inf;beststep=0;hist=[]
 for step,(cat,cond) in enumerate(p.balanced_schedule(1600),1):
  s=cats[cat][cnt[cat]%len(cats[cat])];cnt[cat]+=1;a.train();o.zero_grad(set_to_none=True);_,base=framewise_forward(framewise,model,s,cond);out=temporal_forward(a,framewise,model,s,cond);terms=loss_terms(out,base,s,cond);loss=total(terms,l)
  if not torch.isfinite(loss) or not all(torch.isfinite(x).all() for x in out.values()):raise FloatingPointError(f'{name} forward {step}')
  loss.backward();gn,ga,bad=grad_stats(a)
  if bad:raise FloatingPointError(f'{name} grad {step} {bad}')
  torch.nn.utils.clip_grad_norm_(a.parameters(),1.);o.step();pa,bp=max_parameter(a);bo,_,_=optimizer_finite(o)
  if bp or bo:raise FloatingPointError(f'{name} state {step}')
  if step%40==0:
   h=eval_objective(a,framewise,model,held,l);row={'step':step,'train':float(loss),'heldout':h};hist.append(row)
   if h<best:best,beststep=h,step;torch.save(checkpoint(a,o,step,cnt,best,beststep,l,'best'),folder/'best.pt')
   torch.save(checkpoint(a,o,step,cnt,best,beststep,l,'latest'),folder/'latest.pt');atomic_json(folder/'status.json',{'state':'training','step':step,'best':best,'best_step':beststep});print(json.dumps({'model':name,**row}),flush=True)
 atomic_json(folder/'history.json',{'history':hist});atomic_json(folder/'status.json',{'state':'complete','step':1600,'best':best,'best_step':beststep});ck=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);a.load_state_dict(ck['adapter_state_dict']);return a,beststep,best
def make_records(name,a,framewise,model,samples):
 rows=[]
 with torch.no_grad():
  for s in samples:
   for c,condition in enumerate(CONDITIONS):
    _,base=framewise_forward(framewise,model,s,c);out=base if name=='framewise' else temporal_forward(a,framewise,model,s,c);tar=s['target'][c]['t_all'];raw=s['raw'][c]['pred_cam_t'].cpu().numpy();pred=out['pred_cam_t'].cpu().numpy();rr=p.frame_reprojection(s['raw'][c]['pred_xy'].cpu(),s['meta']['gt_xyc'],s['meta']['size']);rp=p.frame_reprojection(out['pred_xy'].cpu(),s['meta']['gt_xyc'],s['meta']['size'])
    for i,f in enumerate(s['meta']['frame_id']):rows.append({'sequence':s['desc']['sequence'],'person_id':s['desc']['person_id'],'window_start':s['desc']['window_start'],'category':s['desc']['category'],'condition':condition,'frame_id':int(f),'raw_t':raw[i].tolist(),'pred_t':pred[i].tolist(),'target_t':tar[i].tolist(),'raw_error':float(np.linalg.norm(raw[i]-tar[i])),'error':float(np.linalg.norm(pred[i]-tar[i])),'raw_reproj':float(rr[i]),'reproj':float(rp[i]),'correction':float(np.linalg.norm(pred[i]-raw[i]))})
 return rows
def window_metrics(rows):
 g=defaultdict(list)
 for r in rows:g[(r['sequence'],r['person_id'],r['window_start'],r['category'],r['condition'])].append(r)
 out=[]
 for k,v in g.items():
  v=sorted(v,key=lambda x:x['frame_id']);pred=np.asarray([x['pred_t'] for x in v]);tar=np.asarray([x['target_t'] for x in v]);m=trajectory_diagnostics(pred,tar);m.update({'key':list(k),'absolute':float(np.mean([x['error'] for x in v])),'reproj':float(np.mean([x['reproj'] for x in v]))});out.append(m)
 return out
def aggregate(rows,windows):
 base=q.summarize(rows);base['trajectory']={}
 for c in p.CATEGORIES:
  x=[w for w in windows if w['key'][3]==c];base['trajectory'][c]={k:float(np.mean([z[k] for z in x])) for k in ('velocity_mean','velocity_p95','acceleration_mean','acceleration_p95','max_local_jump_error','pred_velocity_magnitude','gt_velocity_magnitude')}
 return base
def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'summary.json').exists():raise FileExistsError(OUT)
 atomic_json(OUT/'status.json',{'state':'initializing','pid':os.getpid()});manifest=json.load(open(REF/'manifest.json'));reject_split(manifest['split']);thresholds=json.load(open(REF/'thresholds.json'));model=load_frozen_model();import smplx;smpl=smplx.SMPL(model_path=str(SMPL_MODEL),batch_size=1).cuda().eval();train=[p.prepare_window(x,model,smpl,thresholds) for x in manifest['train_windows']];held=[p.prepare_window(x,model,smpl,thresholds) for x in manifest['heldout_windows']];shtrain=[q.permute_sample(x,p.SEED) for x in train];shheld=[q.permute_sample(x,p.SEED+1) for x in held]
 framewise=load_framewise();true=CameraFeatureTemporalAdapter().cuda();probe=gradient_probe(true,framewise,model,train);atomic_json(OUT/'loss_probe.json',probe);shuf=CameraFeatureTemporalAdapter().cuda();true,ts,tb=train_variant('true',true,framewise,model,train,held,probe['lambdas']);shuf,ss,sb=train_variant('shuffled',shuf,framewise,model,shtrain,shheld,probe['lambdas'])
 (OUT/'best_true.pt').write_bytes((OUT/'true/best.pt').read_bytes());(OUT/'best_shuffled.pt').write_bytes((OUT/'shuffled/best.pt').read_bytes())
 models={'framewise':(None,held),'true':(true,held),'shuffled':(shuf,shheld)};results={};wins={}
 for name,(a,samples) in models.items():
  rows=make_records(name,a,framewise,model,samples);win=window_metrics(rows);results[name]={'params':0 if name=='framewise' else sum(x.numel() for x in a.parameters()),'metrics':aggregate(rows,win),'windows':win};atomic_json(OUT/f'{name}_temporal.json' if name!='framewise' else OUT/'framewise_reference.json',results[name])
 def wm(n):return{tuple(x['key']):x for x in results[n]['windows']}
 tw,fw,sw=wm('true'),wm('framewise'),wm('shuffled')
 for c in p.CATEGORIES:
  keys=[k for k in tw if k[3]==c];wins[c]={'total':len(keys),'velocity_vs_framewise':sum(tw[k]['velocity_mean']<fw[k]['velocity_mean'] for k in keys),'acceleration_vs_framewise':sum(tw[k]['acceleration_mean']<fw[k]['acceleration_mean'] for k in keys),'absolute_vs_framewise':sum(tw[k]['absolute']<fw[k]['absolute'] for k in keys),'velocity_vs_shuffled':sum(tw[k]['velocity_mean']<sw[k]['velocity_mean'] for k in keys),'acceleration_vs_shuffled':sum(tw[k]['acceleration_mean']<sw[k]['acceleration_mean'] for k in keys)}
 # Fast-motion safety on top quartile held-out GT-velocity windows.
 fast=sorted(fw.values(),key=lambda x:x['gt_velocity_magnitude']);fast=fast[int(.75*len(fast)):];fk={tuple(x['key']) for x in fast};safety={'windows':len(fk),'framewise_pred_gt_velocity_ratio':float(np.mean([fw[k]['pred_velocity_magnitude']/max(fw[k]['gt_velocity_magnitude'],1e-8) for k in fk])),'true_pred_gt_velocity_ratio':float(np.mean([tw[k]['pred_velocity_magnitude']/max(tw[k]['gt_velocity_magnitude'],1e-8) for k in fk])),'framewise_absolute':float(np.mean([fw[k]['absolute'] for k in fk])),'true_absolute':float(np.mean([tw[k]['absolute'] for k in fk]))}
 config={'seed':p.SEED,'steps':1600,'lr':1e-3,'framewise_checkpoint':str(P111/'framewise/best.pt'),'framewise_frozen':True,'body_sha256':p.sha256(p.BODY),'manifest_sha256':p.sha256(REF/'manifest.json'),'lambdas':probe['lambdas'],'selection':'held-out weighted absolute+velocity+acceleration+reprojection'};atomic_json(OUT/'config.json',config);summary={'state':'complete','results':results,'wins':wins,'fast_motion':safety,'probe':probe,'best':{'true_step':ts,'true_loss':tb,'shuffled_step':ss,'shuffled_loss':sb},'body_sha256':p.sha256(p.BODY),'elapsed_seconds':time.time()-started};atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'status.json',{'state':'complete','pid':os.getpid(),'elapsed_seconds':summary['elapsed_seconds']});print(json.dumps({'state':'complete','elapsed':summary['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
