"""Phase-1.11 temporal necessity and calibration ablation (3DPW-train only)."""
from __future__ import annotations
import copy, hashlib, json, math, os, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import train_dynamesh_v2_camera_phase110 as p
from resume_dynamesh_v2_camera_phase110_debug import atomic_json, grad_stats, max_parameter, optimizer_finite
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase19 import load_adapter, load_frozen_model
from train_dynamesh_v2_camera_phase1 import masked_normalized_reprojection_loss
from validate_dynamesh_v2_camera_multijoint_gt import SMPL_MODEL

ROOT=p.ROOT; REF=ROOT/'outputs/dynamesh_v2_camera_phase110_multiwindow_stable_norm'
OUT=ROOT/'outputs/dynamesh_v2_camera_phase111_temporal_ablation'; CONDITIONS=p.CONDITIONS

class GlobalAffineCalibration(nn.Module):
 def __init__(self,mode='xyz'):
  super().__init__(); self.mode=mode
  n=3 if mode=='xyz' else 1
  if mode not in ('xyz','depth'): raise ValueError(mode)
  self.scale=nn.Parameter(torch.ones(n)); self.bias=nn.Parameter(torch.zeros(n))
 def forward(self,t):
  if self.mode=='xyz': return t*self.scale+self.bias
  out=t.clone(); out[...,2]=t[...,2]*self.scale[0]+self.bias[0]; return out

class FramewiseLatentAdapter(nn.Module):
 def __init__(self):
  super().__init__(); self.norm=nn.LayerNorm(1024); self.fc1=nn.Linear(1024,128); self.fc2=nn.Linear(128,1024)
  nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
 def forward(self,x): return x+self.fc2(F.gelu(self.fc1(self.norm(x))))

def deterministic_permutation(length,seed,key):
 h=int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16],16); rng=np.random.default_rng(h)
 q=rng.permutation(length)
 if np.array_equal(q,np.arange(length)): q=np.roll(q,1)
 return q

def permute_time_aligned(mapping,permutation,fields):
 out=dict(mapping)
 for k in fields:
  v=mapping[k]; idx=torch.as_tensor(permutation,device=v.device) if torch.is_tensor(v) else permutation
  out[k]=v[idx]
 return out

def permute_sample(sample,seed):
 q=deterministic_permutation(16,seed,f"{sample['desc']['sequence']}:{sample['desc']['person_id']}:{sample['desc']['window_start']}")
 out=copy.deepcopy(sample)
 out['meta']={k:(v[q] if isinstance(v,np.ndarray) and v.ndim and len(v)==16 else v) for k,v in sample['meta'].items()}
 out['tokens']=[x[torch.as_tensor(q,device=x.device)] for x in sample['tokens']]
 out['raw']=[{k:(v[torch.as_tensor(q,device=v.device)] if torch.is_tensor(v) and v.ndim and len(v)==16 else v) for k,v in x.items()} for x in sample['raw']]
 out['target']=[{k:(v[q] if isinstance(v,np.ndarray) and v.ndim and len(v)==16 else v) for k,v in x.items()} for x in sample['target']]
 out['permutation']=q.tolist(); return out

def assert_matching_supervision(a,b):
 for k in ('t_all','weight'):
  if not np.array_equal(np.asarray(a[k]),np.asarray(b[k])): raise AssertionError(f'{k} mismatch')

def reject_non_train_split(split):
 if split not in ('train','3dpw_train_only'): raise ValueError('Phase-1.11 accepts 3DPW train only')

def project_points_with_translation(points,t,K):
 cam=points+t[:,None,:]; z=cam[...,2:3]
 if torch.any(z<=0): raise FloatingPointError('non-positive projected depth')
 K=K.to(points)
 if K.ndim==2: K=K.expand(len(points),-1,-1)
 uv=torch.einsum('bij,bnj->bni',K,cam); return uv[...,:2]/uv[...,2:3]

def trajectory_errors(pred,target):
 pred=np.asarray(pred); target=np.asarray(target)
 v=np.diff(pred,axis=0)-np.diff(target,axis=0); a=np.diff(pred,n=2,axis=0)-np.diff(target,n=2,axis=0)
 return {'velocity_mean':float(np.linalg.norm(v,axis=1).mean()) if len(v) else 0.,'acceleration_mean':float(np.linalg.norm(a,axis=1).mean()) if len(a) else 0.}

def direct_forward(module,sample,c):
 raw=sample['raw'][c]; target=sample['target'][c]; meta=sample['meta']
 t=module(raw['pred_cam_t']); K=torch.as_tensor(np.broadcast_to(meta['K'],(16,3,3)).copy(),device=t.device,dtype=t.dtype)
 xy=project_points_with_translation(raw['points3d'],t,K)
 return {'pred_cam_t':t,'pred_xy':xy,'pred_cam':raw['pred_cam']}

def frame_forward(module,model,sample,c):
 token=module(sample['tokens'][c][None]).squeeze(0)
 cam=model.head_camera(token,model.init_camera.weight.expand(16,-1)); projected=p.projection(model,sample['raw'][c]['points3d'].detach(),cam,sample['meta'])
 return {'pred_cam_t':projected['pred_cam_t'],'pred_xy':projected['pred_keypoints_2d'],'pred_cam':cam}

def simple_terms(out,sample,c):
 target=sample['target'][c]; meta=sample['meta']; t=torch.as_tensor(target['t_all'],device='cuda',dtype=torch.float32); w=torch.as_tensor(target['weight'],device='cuda',dtype=torch.float32)
 gt=torch.as_tensor(meta['gt_xyc'],device='cuda'); bbox=torch.as_tensor(meta['size'],device='cuda')
 return {'cam3d':p.weighted_cam3d_loss(out['pred_cam_t'],t,w),'reproj':masked_normalized_reprojection_loss(out['pred_xy'],gt,bbox)}

def save_ckpt(path,module,opt,step,counters,best,best_step,kind):
 torch.save(p.training_checkpoint(module,opt,step,counters,{'kind':kind,'best_heldout_total':best,'best_step':best_step,'total_steps':1600,'seed':p.SEED}),path)

def eval_loss(kind,module,model,samples,lambdas):
 vals=[]; module.eval()
 with torch.no_grad():
  for s in samples:
   for c in range(4):
    if kind=='shuffled': vals.append(float(p.total_loss(p.phase110_forward(module,model,s,c)[0],lambdas)))
    else:
     out=direct_forward(module,s,c) if kind.startswith('global') else frame_forward(module,model,s,c)
     z=simple_terms(out,s,c); vals.append(float(z['cam3d']+lambdas['reproj']*z['reproj']))
 return float(np.mean(vals))

def train_variant(kind,module,model,train_samples,held,lambdas,folder):
 folder.mkdir(parents=True,exist_ok=False); opt=torch.optim.AdamW(module.parameters(),lr=1e-3,weight_decay=1e-4)
 cats={c:[s for s in train_samples if s['desc']['category']==c] for c in p.CATEGORIES}; sched=p.balanced_schedule(1600); counters=Counter(); best=math.inf; best_step=0; hist=[]
 for step,(cat,cond) in enumerate(sched,1):
  s=cats[cat][counters[cat]%len(cats[cat])]; counters[cat]+=1; module.train(); opt.zero_grad(set_to_none=True)
  if kind=='shuffled': terms,_=p.phase110_forward(module,model,s,cond); loss=p.total_loss(terms,lambdas)
  else:
   out=direct_forward(module,s,cond) if kind.startswith('global') else frame_forward(module,model,s,cond); terms=simple_terms(out,s,cond); loss=terms['cam3d']+lambdas['reproj']*terms['reproj']
  if not torch.isfinite(loss): raise FloatingPointError(f'{kind} loss step {step}')
  loss.backward(); gn,ga,bad=grad_stats(module)
  if bad: raise FloatingPointError(f'{kind} grad {bad} step {step}')
  torch.nn.utils.clip_grad_norm_(module.parameters(),1.0); opt.step(); pa,bp=max_parameter(module); bo,_,_=optimizer_finite(opt)
  if bp or bo: raise FloatingPointError(f'{kind} state step {step}')
  if step%40==0:
   held_loss=eval_loss(kind,module,model,held,lambdas); row={'step':step,'train':float(loss),'heldout':held_loss}; hist.append(row)
   if held_loss<best: best,best_step=held_loss,step; save_ckpt(folder/'best.pt',module,opt,step,counters,best,best_step,'best')
   save_ckpt(folder/'latest.pt',module,opt,step,counters,best,best_step,'latest'); atomic_json(folder/'status.json',{'state':'training','step':step,'best_step':best_step,'best':best}); print(json.dumps({'model':kind,**row}),flush=True)
 save_ckpt(folder/'final.pt',module,opt,1600,counters,best,best_step,'final'); atomic_json(folder/'history.json',{'history':hist}); atomic_json(folder/'status.json',{'state':'complete','step':1600,'best_step':best_step,'best':best})
 state=torch.load(folder/'best.pt',map_location='cuda',weights_only=False); module.load_state_dict(state['adapter_state_dict']); return module,best_step,best

def records_for(kind,module,model,samples):
 rows=[]; module.eval()
 with torch.no_grad():
  for s in samples:
   meta=s['meta']
   for c,condition in enumerate(CONDITIONS):
    if kind in ('reference','shuffled'):
     _,o=p.phase110_forward(module,model,s,c); out={'pred_cam_t':o['pred_cam_t'],'pred_xy':o['pred_xy']}
    elif kind.startswith('global'): out=direct_forward(module,s,c)
    else: out=frame_forward(module,model,s,c)
    raw=s['raw'][c]; target=s['target'][c]['t_all']; gt=meta['gt_xyc']; bbox=meta['size']
    raw_rep=p.frame_reprojection(raw['pred_xy'].cpu(),gt,bbox); rep=p.frame_reprojection(out['pred_xy'].cpu(),gt,bbox)
    rt=raw['pred_cam_t'].cpu().numpy(); vt=out['pred_cam_t'].cpu().numpy()
    for i,f in enumerate(meta['frame_id']): rows.append({'sequence':s['desc']['sequence'],'person_id':s['desc']['person_id'],'window_start':s['desc']['window_start'],'frame_id':int(f),'category':s['desc']['category'],'condition':condition,'raw_t':rt[i].tolist(),'pred_t':vt[i].tolist(),'target_t':target[i].tolist(),'raw_error':float(np.linalg.norm(rt[i]-target[i])),'error':float(np.linalg.norm(vt[i]-target[i])),'raw_reproj':float(raw_rep[i]),'reproj':float(rep[i]),'correction':float(np.linalg.norm(vt[i]-rt[i]))})
 return rows

def summarize(rows):
 def group(rr):
  def a(k): return np.asarray([x[k] for x in rr],float)
  raw=a('raw_error'); e=a('error'); rp=a('raw_reproj'); pr=a('reproj'); rz=np.abs(np.asarray([x['raw_t'][2]-x['target_t'][2] for x in rr])); z=np.abs(np.asarray([x['pred_t'][2]-x['target_t'][2] for x in rr]));
  return {'n':len(rr),'camera_mean':float(e.mean()),'camera_median':float(np.median(e)),'camera_p95':float(np.percentile(e,95)),'camera_improvement_pct':float(100*(raw.mean()-e.mean())/raw.mean()),'reprojection_mean':float(pr.mean()),'reprojection_p95':float(np.percentile(pr,95)),'reprojection_improvement_pct':float(100*(rp.mean()-pr.mean())/rp.mean()),'z_error':float(z.mean()),'raw_z_error':float(rz.mean()),'target_z':float(np.mean([x['target_t'][2] for x in rr])),'correction':float(a('correction').mean())}
 return {'category':{c:group([x for x in rows if x['category']==c]) for c in p.CATEGORIES},'condition':{c:group([x for x in rows if x['condition']==c]) for c in CONDITIONS}}

def per_window(rows):
 g=defaultdict(list)
 for r in rows:g[(r['sequence'],r['person_id'],r['window_start'],r['category'],r['condition'])].append(r)
 return [{'key':list(k),'improvement':float(np.mean([x['raw_error']-x['error'] for x in v])),'trajectory':trajectory_errors(np.asarray([x['pred_t'] for x in sorted(v,key=lambda y:y['frame_id'])]),np.asarray([x['target_t'] for x in sorted(v,key=lambda y:y['frame_id'])]))} for k,v in g.items()]

def main():
 started=time.time(); OUT.mkdir(parents=True,exist_ok=True)
 if any((OUT/x).exists() for x in ('summary.json','global_depth','global_xyz','framewise','shuffled')): raise FileExistsError(f'non-empty scientific output: {OUT}')
 (OUT/'run.pid').write_text(str(os.getpid())); atomic_json(OUT/'status.json',{'state':'initializing','pid':os.getpid()})
 for x in ('manifest.json','thresholds.json','loss_scale_probe.json'):
  (OUT/x).write_bytes((REF/x).read_bytes())
 manifest=json.load(open(REF/'manifest.json')); reject_non_train_split(manifest['split']); thresholds=json.load(open(REF/'thresholds.json')); probe=json.load(open(REF/'loss_scale_probe.json')); lambdas=probe['lambdas']
 model=load_frozen_model(); import smplx; smpl=smplx.SMPL(model_path=str(SMPL_MODEL),batch_size=1).cuda().eval()
 train=[p.prepare_window(x,model,smpl,thresholds) for x in manifest['train_windows']]; held=[p.prepare_window(x,model,smpl,thresholds) for x in manifest['heldout_windows']]
 shuffled_train=[permute_sample(x,p.SEED) for x in train]; shuffled_held=[permute_sample(x,p.SEED+1) for x in held]
 for a,b in zip(train,shuffled_train):
  for c in range(4): assert_matching_supervision(a['target'][c],permute_time_aligned(b['target'][c],np.argsort(b['permutation']),{'t_all','weight'}))
 modules={'global_depth':GlobalAffineCalibration('depth').cuda(),'global_xyz':GlobalAffineCalibration('xyz').cuda(),'framewise':FramewiseLatentAdapter().cuda(),'shuffled':CameraFeatureTemporalAdapter().cuda()}
 result={}; allrows={}
 for kind,module in modules.items():
  tr,ho=(shuffled_train,shuffled_held) if kind=='shuffled' else (train,held)
  module,bs,bv=train_variant(kind,module,model,tr,ho,lambdas,OUT/kind)
  rows=records_for(kind,module,model,ho); allrows[kind]=rows; result[kind]={'params':sum(x.numel() for x in module.parameters()),'best_step':bs,'best_loss':bv,'metrics':summarize(rows),'windows':per_window(rows)}; atomic_json(OUT/f'{kind}.json',result[kind])
 ref=load_adapter(REF/'best.pt'); refrows=records_for('reference',ref,model,held); allrows['reference']=refrows; result['reference']={'params':sum(x.numel() for x in ref.parameters()),'metrics':summarize(refrows),'windows':per_window(refrows)}; atomic_json(OUT/'temporal_reference.json',result['reference'])
 # Train/held-out depth affine diagnostic.
 def zr2(samples):
  z=[];t=[]
  for s in samples:
   for c in range(4): z.extend(s['raw'][c]['pred_cam_t'][:,2].cpu().numpy()); t.extend(s['target'][c]['t_all'][:,2])
  z=np.asarray(z);t=np.asarray(t); co=np.polyfit(z,t,1); pred=np.polyval(co,z); return z,t,co,float(1-np.sum((t-pred)**2)/np.sum((t-t.mean())**2)),float(np.std(t-pred))
 z,t,co,r2,rs=zr2(train); zh,th,_,_,_=zr2(held); heldres=float(np.std(th-np.polyval(co,zh)))
 # wins and temporal gains, using matching window keys.
 def winmap(k): return {tuple(x['key']):x for x in result[k]['windows']}
 rw=winmap('reference'); comparisons={}
 for other in ('global_depth','global_xyz','framewise','shuffled'):
  ow=winmap(other); comparisons[other]={c:{'temporal_wins':sum(rw[k]['improvement']>ow[k]['improvement'] for k in rw if k[3]==c),'total':sum(k[3]==c for k in rw),'mean_camera_gain':float(np.mean([ow[k]['improvement']-rw[k]['improvement'] for k in rw if k[3]==c]))} for c in p.CATEGORIES}
 config={'seed':p.SEED,'steps':1600,'lr':1e-3,'same_manifest':True,'same_targets':True,'body_sha256':p.sha256(p.BODY),'reference_sha256':p.sha256(REF/'best.pt'),'models':{k:sum(x.numel() for x in v.parameters()) for k,v in modules.items()}|{'reference':sum(x.numel() for x in ref.parameters())}}
 atomic_json(OUT/'config.json',config); atomic_json(OUT/'manifest_hashes.json',{x:p.sha256(REF/x) for x in ('manifest.json','thresholds.json','loss_scale_probe.json')})
 summary={'state':'complete','results':result,'comparisons':comparisons,'depth_bias':{'a':float(co[0]),'b':float(co[1]),'train_r2':r2,'train_residual_std':rs,'heldout_residual_std':heldres},'elapsed_seconds':time.time()-started,'body_sha256':p.sha256(p.BODY)}
 atomic_json(OUT/'summary.json',summary); atomic_json(OUT/'status.json',{'state':'complete','pid':os.getpid(),'elapsed_seconds':summary['elapsed_seconds']}); print(json.dumps({'state':'complete','elapsed':summary['elapsed_seconds']}),flush=True)

if __name__=='__main__': main()
