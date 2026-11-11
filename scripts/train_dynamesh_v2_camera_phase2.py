"""Camera Phase-2: full cached 3DPW train/validation, two-stage pipeline.

This process only decodes existing feature-cache tensors with frozen heads. It
never invokes the visual backbone or conversion tools. Stage A trains the
frame-wise calibrator; Stage B freezes it and trains stable-normalized temporal
residual models for true and deterministically shuffled order.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import cv2

import train_dynamesh_v2_camera_phase110 as p
import ablate_dynamesh_v2_camera_phase111 as q
import train_dynamesh_v2_camera_phase112 as r
import train_dynamesh_v2_camera_phase19 as phase19
from train_dynamesh_v2_camera_phase1 import assert_aligned_camera_fields
import train_dynamesh_v2_camera_phase1 as phase1
from train_dynamesh_v2_camera_phase19 import load_frozen_model
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from validate_dynamesh_v2_camera_multijoint_gt import SMPL_MODEL

ROOT=Path('/root/sam-body4d-0.2.0'); OUT=ROOT/'outputs/dynamesh_v2_camera_phase2'
CACHE=ROOT/'data/dynamesh_v2/cache_v2'; WINDOW_CACHE=OUT/'camera_window_cache'
EXPECTED_BODY_SHA='f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329'
SEED=20260902; CONDITIONS=p.CONDITIONS; MAX_EPOCHS=12; PATIENCE=3
RESUME_KEYS={'adapter_state_dict','optimizer_state_dict','current_step','current_epoch','rng_state','sampler_state','best_validation_state','config_hash','manifest_hash'}

def sha(path):
 h=hashlib.sha256();
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def atomic_json(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,path)
def reject_split(split):
 if split=='test':raise ValueError('3DPW test is forbidden')
 if split not in ('train','validation'):raise ValueError(split)
def load_phase2_track(path,purpose):
 if purpose not in ('optimization','validation'):raise ValueError(f'unknown Phase-2 loader purpose: {purpose}')
 with np.load(path,allow_pickle=False) as z:track={k:np.asarray(z[k]) for k in z.files}
 assert_aligned_camera_fields(track['pose_token'],track['mhr_init_pose'],track['frame_id'],track['person_id'].item(),tuple(track['condition_names'].astype(str)))
 split=str(track['split'].item())
 if split=='test':raise ValueError('3DPW test is forbidden')
 expected='train' if purpose=='optimization' else 'validation'
 if split!=expected:raise ValueError(f'Phase-2 {purpose} requires {expected} cache, got {split}')
 return track
def annotation_split_for_purpose(purpose):
 if purpose=='optimization':return 'train'
 if purpose=='validation':return 'validation'
 raise ValueError(f'unknown Phase-2 loader purpose: {purpose}')
def phase2_metadata_for_track(track,purpose):
 split=annotation_split_for_purpose(purpose);sequence=str(track['sequence'].item());person=int(track['person_id'].item())
 ann=phase1.load_pickle(phase1.DATA/'sequenceFiles'/split/f'{sequence}.pkl');selected=np.asarray(track['local_frame_index'],np.int64)
 frame_ids=np.asarray(ann['img_frame_ids'],np.int64)[selected]
 if not np.array_equal(frame_ids,track['frame_id']):raise ValueError(f'{sequence}/person{person}: annotation/cache frame mismatch')
 image_path=phase1.DATA/'imageFiles'/sequence/f'image_{selected[0]:05d}.jpg';image=cv2.imread(str(image_path))
 if image is None:raise FileNotFoundError(image_path)
 h,w=image.shape[:2];poses=np.asarray(ann['poses2d'][person]);boxes=np.stack([phase1.bbox_from_pose2d(poses[i],w,h) for i in selected]).astype(np.float32)
 center=(boxes[:,:2]+boxes[:,2:])/2.;size=(boxes[:,2]-boxes[:,0])*1.25;K=np.asarray(ann['cam_intrinsics'],np.float32);gt=np.transpose(poses[selected][:,:,phase1.OP_IDX],(0,2,1)).astype(np.float32)
 return {'sequence':sequence,'person':person,'selected':selected,'frame_id':frame_ids,'boxes':boxes,'center':center,'size':size,'K':K,'image_size':np.tile(np.array([[w,h]],np.float32),(len(selected),1)),'gt_xyc':gt}
def prepare_phase2_window(desc,model,smpl,thresholds,purpose):
 """Inject the Phase-2 split contract only around legacy target preparation."""
 original_load,original_guard,original_meta=phase19.load_track,phase19.assert_train_only,phase19.metadata_for_track
 phase19.load_track=lambda path:load_phase2_track(path,purpose)
 phase19.assert_train_only=lambda split: None
 phase19.metadata_for_track=lambda track:phase2_metadata_for_track(track,purpose)
 try:return p.prepare_window(desc,model,smpl,thresholds)
 finally:phase19.load_track,phase19.assert_train_only,phase19.metadata_for_track=original_load,original_guard,original_meta
def audit_feature_manifest(path):
 m=json.load(open(path)); expected={'train_tracks':34,'validation_tracks':16,'train_frames':22735,'validation_frames':10413,'test_cache_files':0}
 for k,v in expected.items():
  if m.get(k)!=v:raise ValueError(f'manifest {k}: {m.get(k)} != {v}')
 seq={s:set() for s in ('train','validation')}
 for e in m['entries']:reject_split(e['split']);seq[e['split']].add(e['sequence'])
 if seq['train']&seq['validation']:raise ValueError('train/validation sequence leak')
 return m
def contiguous_windows(local,length=16):
 local=np.asarray(local);out=[]
 if not len(local):return out
 start=0
 for i in range(1,len(local)+1):
  if i==len(local) or local[i]!=local[i-1]+1:
   out += [(j,j+length) for j in range(start,i-length+1,length)];start=i
 return out
def stage_dir(name):return OUT/name
def temporal_permutation(n,seed,key):return q.deterministic_permutation(n,seed,key)
def protocol_config():
 return {'seed':SEED,'body_sha256':EXPECTED_BODY_SHA,'tau_source':'3DPW train clean only','validation_thresholds_source':'frozen train thresholds','window_length':16,'conditions':list(CONDITIONS),'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'lr':1e-3,'weight_decay':1e-4,'stage_a_architecture':'Phase-1.11 FramewiseLatentAdapter','stage_b_architecture':'Phase-1.12 stable-norm temporal residual','forbidden':['3DPW test','SAM visual backbone','Meta conversion','Body training']}
def assert_optimizer_ownership(opt,trainable,frozen=()):
 actual={id(x) for g in opt.param_groups for x in g['params']};expected={id(x) for x in trainable.parameters() if x.requires_grad}
 if actual!=expected or any(x.requires_grad for m in frozen for x in m.parameters()):raise AssertionError('optimizer/freeze ownership violation')
def _cpu(x):
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return {k:_cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [_cpu(v) for v in x]
 return x
def _cuda(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:_cuda(v) for k,v in x.items()}
 if isinstance(x,list):return [_cuda(v) for v in x]
 return x
def enforce_target_reliability_contract(target):
 """Undefined fit diagnostics mean an unusable 3D target, not a dropped frame."""
 out=dict(target);weight=np.asarray(target['weight']).copy();fit=np.asarray(target['fit']);ul=np.asarray(target['ul'])
 undefined=~np.isfinite(weight)|~np.isfinite(fit)|~np.isfinite(ul)
 weight[undefined]=0.;out['weight']=weight
 if 'anchor_weight' in out:
  anchor=np.asarray(out['anchor_weight']).copy();anchor[undefined|~np.isfinite(anchor)]=0.;out['anchor_weight']=anchor
 return out
def enforce_sample_target_contract(sample):
 out=dict(sample);out['target']=[enforce_target_reliability_contract(x) for x in sample['target']];return out
def descriptors(manifest,split):
 reject_split(split);out=[]
 if split=='validation':
  for w in manifest['validation_windows_length16_nonoverlap']:
   out.append({'sequence':w['sequence'],'person_id':int(w['person_id']),'category':'medium','center_frame':int(w['frame_ids'][8]),'window_start':int(w['cache_start']),'window_frame_ids':w['frame_ids'],'cache_path':str(CACHE/split/f"{w['sequence']}_person{w['person_id']}.npz")})
 else:
  for e in manifest['entries']:
   if e['split']!=split:continue
   z=np.load(e['path'],allow_pickle=False)
   for a,b in contiguous_windows(z['local_frame_index'],16):
    ids=z['frame_id'][a:b].astype(int).tolist();out.append({'sequence':e['sequence'],'person_id':int(e['person_id']),'category':'medium','center_frame':ids[8],'window_start':a,'window_frame_ids':ids,'cache_path':e['path']})
 return out
def config_hash(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
def resume_cursor(current_step,current_epoch,windows_per_epoch):
 offset=int(current_step)-(int(current_epoch)-1)*int(windows_per_epoch)
 if offset<0 or offset>windows_per_epoch:raise ValueError('checkpoint step/epoch cursor is inconsistent')
 return (int(current_epoch)+1,0) if offset==windows_per_epoch else (int(current_epoch),offset)
def assert_resume_identity(checkpoint,expected_config_hash,expected_manifest_hash):
 if checkpoint.get('config_hash')!=expected_config_hash:raise ValueError('resume config hash mismatch')
 if checkpoint.get('manifest_hash')!=expected_manifest_hash:raise ValueError('resume manifest hash mismatch')
def restore_rng_state(state):
 random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])
def update(stage,**kw):atomic_json(OUT/'status.json',{'state':'running','stage':stage,'pid':os.getpid(),'updated_at':time.time(),**kw})
def prepare_cache(manifest,model,smpl,thresholds):
 index={}
 for split in ('train','validation'):
  desc=descriptors(manifest,split);by={}
  for d in desc:by.setdefault((d['sequence'],d['person_id']),[]).append(d)
  index[split]=[]
  for ti,(key,items) in enumerate(sorted(by.items()),1):
   folder=WINDOW_CACHE/split;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{key[0]}_person{key[1]}.pt'
   if not path.exists():
    samples=[]
    for wi,d in enumerate(items):
     purpose='optimization' if split=='train' else 'validation'
     samples.append(_cpu(prepare_phase2_window(d,model,smpl,thresholds,purpose)));update('preparing',split=split,track=f'{key[0]}/person{key[1]}',track_index=ti,window=wi+1,windows=len(items))
    tmp=path.with_suffix('.pt.tmp');torch.save(samples,tmp);os.replace(tmp,path)
   index[split].append({'sequence':key[0],'person_id':key[1],'path':str(path),'windows':len(items),'sha256':sha(path)})
 atomic_json(WINDOW_CACHE/'manifest.json',index);return index
def load_samples(index,split):
 for track in index[split]:
  for s in torch.load(track['path'],map_location='cpu',weights_only=False):yield _cuda(enforce_sample_target_contract(s))
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}
def checkpoint(module,opt,step,epoch,counters,best,cfg,mhash):
 return {'adapter_state_dict':module.state_dict(),'optimizer_state_dict':opt.state_dict(),'current_step':step,'current_epoch':epoch,'rng_state':rng_state(),'sampler_state':{'counters':dict(counters),'seed':SEED},'best_validation_state':best,'config_hash':config_hash(cfg),'manifest_hash':mhash}
def frame_loss(module,model,s,c,lambdas):return q.simple_terms(q.frame_forward(module,model,s,c),s,c)
def eval_frame(module,model,index,lambdas):
 vals=[];module.eval()
 with torch.no_grad():
  for s in load_samples(index,'validation'):
   for c in range(4):
    z=frame_loss(module,model,s,c,lambdas);vals.append(float(z['cam3d']+lambdas['reproj']*z['reproj']))
 return float(np.mean(vals))
def train_framewise(model,index,cfg,mhash,lambdas):
 folder=stage_dir('framewise');folder.mkdir(parents=True,exist_ok=True);m=q.FramewiseLatentAdapter().cuda();o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);assert_optimizer_ownership(o,m)
 best={'value':float('inf'),'epoch':0};bad=0;step=0;history=[];start_epoch=1;skip=0
 latest=folder/'latest.pt';windows_per_epoch=sum(x['windows'] for x in index['train'])
 if latest.exists():
  ck=torch.load(latest,map_location='cpu',weights_only=False);assert_resume_identity(ck,config_hash(cfg),mhash);m.load_state_dict(ck['adapter_state_dict']);o.load_state_dict(ck['optimizer_state_dict']);step=int(ck['current_step']);best=ck['best_validation_state'];start_epoch,skip=resume_cursor(step,ck['current_epoch'],windows_per_epoch);restore_rng_state(ck['rng_state']);update('framewise_training',step=step,epoch=start_epoch,resumed_from=step)
 for epoch in range(start_epoch,MAX_EPOCHS+1):
  for window_index,s in enumerate(load_samples(index,'train')):
   if epoch==start_epoch and window_index<skip:continue
   c=step%4;o.zero_grad(set_to_none=True);z=frame_loss(m,model,s,c,lambdas);loss=z['cam3d']+lambdas['reproj']*z['reproj']
   if not torch.isfinite(loss):raise FloatingPointError('framewise non-finite')
   loss.backward()
   if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in m.parameters()):raise FloatingPointError('framewise non-finite gradient')
   torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step();step+=1
   if any(not torch.isfinite(x).all() for x in m.parameters()):raise FloatingPointError('framewise non-finite parameter')
   if step%40==0:torch.save(checkpoint(m,o,step,epoch,Counter(),best,cfg,mhash),folder/'latest.pt');update('framewise_training',step=step,epoch=epoch)
  skip=0
  val=eval_frame(m,model,index,lambdas);history.append({'epoch':epoch,'validation':val})
  if val<best['value']:best={'value':val,'epoch':epoch};torch.save(checkpoint(m,o,step,epoch,Counter(),best,cfg,mhash),folder/'best.pt');bad=0
  else:bad+=1
  update('framewise_validation',step=step,epoch=epoch,best=best)
  if bad>=PATIENCE:break
 atomic_json(folder/'history.json',{'history':history});return m
def temporal_loss(adapter,framewise,model,s,c,l):
 _,base=r.framewise_forward(framewise,model,s,c);out=r.temporal_forward(adapter,framewise,model,s,c);z=r.loss_terms(out,base,s,c);return sum(l[k]*z[k] for k in z)
def eval_temporal(a,f,model,index,l,shuffled):
 vals=[];a.eval()
 with torch.no_grad():
  for i,s in enumerate(load_samples(index,'validation')):
   if shuffled:s=q.permute_sample(s,SEED+1)
   for c in range(4):vals.append(float(temporal_loss(a,f,model,s,c,l)))
 return float(np.mean(vals))
def train_temporal(name,framewise,model,index,cfg,mhash,lambdas,shuffled=False):
 folder=stage_dir(name);folder.mkdir(parents=True,exist_ok=True);a=CameraFeatureTemporalAdapter().cuda();o=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=1e-4);assert_optimizer_ownership(o,a,[framewise]);best={'value':float('inf'),'epoch':0};bad=0;step=0;hist=[]
 for epoch in range(1,MAX_EPOCHS+1):
  for i,s in enumerate(load_samples(index,'train')):
   if shuffled:s=q.permute_sample(s,SEED)
   c=step%4;o.zero_grad(set_to_none=True);loss=temporal_loss(a,framewise,model,s,c,lambdas);loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1);o.step();step+=1
   if not torch.isfinite(loss):raise FloatingPointError(f'{name} non-finite')
   if step%40==0:torch.save(checkpoint(a,o,step,epoch,Counter(),best,cfg,mhash),folder/'latest.pt');update('temporal_training',variant=name,step=step,epoch=epoch)
  val=eval_temporal(a,framewise,model,index,lambdas,shuffled);hist.append({'epoch':epoch,'validation':val})
  if val<best['value']:best={'value':val,'epoch':epoch};torch.save(checkpoint(a,o,step,epoch,Counter(),best,cfg,mhash),folder/'best.pt');bad=0
  else:bad+=1
  update('temporal_validation',variant=name,step=step,epoch=epoch,best=best)
  if bad>=PATIENCE:break
 atomic_json(folder/'history.json',{'history':hist});return best
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'phase2.pid').write_text(str(os.getpid()));random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 cfg=protocol_config();manifest_path=CACHE/'manifest.json';manifest=audit_feature_manifest(manifest_path);mhash=sha(manifest_path)
 if sha(p.BODY)!=EXPECTED_BODY_SHA:raise RuntimeError('Body SHA256 mismatch')
 thresholds=json.load(open(ROOT/'outputs/dynamesh_v2_camera_phase110_multiwindow_stable_norm/thresholds.json'));probe111=json.load(open(ROOT/'outputs/dynamesh_v2_camera_phase111_temporal_ablation/loss_scale_probe.json'));probe112=json.load(open(ROOT/'outputs/dynamesh_v2_camera_phase112_temporal_residual/loss_probe.json'))
 cfg.update({'manifest_hash':mhash,'thresholds_hash':hashlib.sha256(json.dumps(thresholds,sort_keys=True).encode()).hexdigest(),'stage_a_lambdas':probe111['lambdas'],'stage_b_lambdas':probe112['lambdas']});atomic_json(OUT/'config.json',cfg);update('preparing',step=0,epoch=0)
 model=load_frozen_model();import smplx;smpl=smplx.SMPL(model_path=str(SMPL_MODEL),batch_size=1).cuda().eval();index=prepare_cache(manifest,model,smpl,thresholds)
 frame=train_framewise(model,index,cfg,mhash,probe111['lambdas']);ck=torch.load(stage_dir('framewise')/'best.pt',map_location='cuda',weights_only=False);frame.load_state_dict(ck['adapter_state_dict']);frame.eval().requires_grad_(False)
 true=train_temporal('temporal_true',frame,model,index,cfg,mhash,probe112['lambdas'],False);shuf=train_temporal('temporal_shuffled',frame,model,index,cfg,mhash,probe112['lambdas'],True)
 atomic_json(OUT/'summary.json',{'state':'complete','framewise_best':ck['best_validation_state'],'temporal_true_best':true,'temporal_shuffled_best':shuf});atomic_json(OUT/'status.json',{'state':'complete','stage':'complete','pid':os.getpid()})
if __name__=='__main__':main()
