"""Pure cached validation evaluation for Camera Phase-2 best checkpoints."""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root/sam-body4d-0.2.0/scripts')
import train_dynamesh_v2_camera_phase2 as p2
import train_dynamesh_v2_camera_phase110 as p
import ablate_dynamesh_v2_camera_phase111 as q
import train_dynamesh_v2_camera_phase112 as r
from train_dynamesh_v2_camera_phase19 import load_frozen_model
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter

ROOT=p2.ROOT;OUT=p2.OUT;CONDITIONS=p2.CONDITIONS
METHODS=('raw_sam','global_xyz_affine','framewise','framewise_shuffled_temporal','framewise_true_temporal')
def atomic_json(path,value):
 t=Path(str(path)+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,path)
def assert_validation_only(split):
 if split!='validation':raise ValueError('final evaluation accepts validation only; test is forbidden')
def scalar_stats(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 return {'n':int(len(x)),'mean':float(x.mean()),'median':float(np.median(x)),'std':float(x.std()),'p95':float(np.percentile(x,95)),'max':float(x.max())}
def trajectory_metrics(pred,target):
 pred=np.asarray(pred,float);target=np.asarray(target,float);dv=np.diff(pred,axis=0)-np.diff(target,axis=0);da=np.diff(pred,n=2,axis=0)-np.diff(target,n=2,axis=0);ve=np.linalg.norm(dv,axis=1);ae=np.linalg.norm(da,axis=1)
 return {'velocity_error_mean':float(ve.mean()) if len(ve) else 0.,'velocity_error_p95':float(np.percentile(ve,95)) if len(ve) else 0.,'acceleration_error_mean':float(ae.mean()) if len(ae) else 0.,'acceleration_error_p95':float(np.percentile(ae,95)) if len(ae) else 0.,'trajectory_p95':float(np.percentile(ve,95)) if len(ve) else 0.,'max_local_jump':float(ve.max()) if len(ve) else 0.,'pred_velocity_magnitude':float(np.linalg.norm(np.diff(pred,axis=0),axis=1).mean()) if len(pred)>1 else 0.,'gt_velocity_magnitude':float(np.linalg.norm(np.diff(target,axis=0),axis=1).mean()) if len(target)>1 else 0.}
def quality_bucket(row,t):
 ok=np.isfinite(row['fit_residual']) and np.isfinite(row['upper_lower_disagreement']) and row['fit_residual']<=t['fit_p75'] and row['upper_lower_disagreement']<=t['ul_p75']
 if ok and row['camera_error']<=t['camera_p25'] and row['reprojection_error']<=t['reproj_p50']:return 'stable'
 if ok and row['camera_error']>=t['camera_p75'] and row['reprojection_error']>=t['reproj_p75']:return 'failure'
 if ok and t['camera_p40']<=row['camera_error']<=t['camera_p60'] and row['reprojection_error']<=t['reproj_p75']:return 'medium'
 return 'other'
def load_module(cls,path):
 m=cls().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['adapter_state_dict']);m.eval().requires_grad_(False);return m
def output(method,models,s,c):
 raw=s['raw'][c]
 if method=='raw_sam':return {'pred_cam_t':raw['pred_cam_t'],'pred_xy':raw['pred_xy']},s
 if method=='global_xyz_affine':return q.direct_forward(models['global'],s,c),s
 if method=='framewise':return r.framewise_forward(models['framewise'],models['sam'],s,c)[1],s
 if method=='framewise_true_temporal':return r.temporal_forward(models['true'],models['framewise'],models['sam'],s,c),s
 ps=q.permute_sample(s,p.SEED+1);return r.temporal_forward(models['shuffled'],models['framewise'],models['sam'],ps,c),ps
def aggregate(rows,windows):
 e=np.asarray([x['error'] for x in rows]);z=np.asarray([x['z_error'] for x in rows]);rp=np.asarray([x['reproj'] for x in rows]);xyz=np.asarray([x['xyz_abs'] for x in rows]);corr=np.asarray([x['correction'] for x in rows])
 result={'translation':scalar_stats(e),'depth_abs':scalar_stats(z),'reprojection':scalar_stats(rp),'xyz_abs_mean':xyz.mean(0).tolist(),'correction':scalar_stats(corr)}
 for k in ('velocity_error_mean','velocity_error_p95','acceleration_error_mean','acceleration_error_p95','trajectory_p95','max_local_jump','pred_velocity_magnitude','gt_velocity_magnitude'):result[k]=float(np.mean([w[k] for w in windows])) if windows else None
 result['windows']=len(windows);return result
def main():
 started=time.time();idx=json.load(open(p2.WINDOW_CACHE/'manifest.json'));assert_validation_only('validation');thresholds=json.load(open(ROOT/'outputs/dynamesh_v2_camera_phase110_multiwindow_stable_norm/thresholds.json'))
 models={'sam':load_frozen_model(),'global':load_module(q.GlobalAffineCalibration,ROOT/'outputs/dynamesh_v2_camera_phase111_temporal_ablation/global_xyz/best.pt'),'framewise':load_module(q.FramewiseLatentAdapter,OUT/'framewise/best.pt'),'true':load_module(CameraFeatureTemporalAdapter,OUT/'temporal_true/best.pt'),'shuffled':load_module(CameraFeatureTemporalAdapter,OUT/'temporal_shuffled/best.pt')}
 rows={x:[] for x in METHODS};wg={x:defaultdict(list) for x in METHODS};window_categories={};count=0
 with torch.no_grad():
  for wi,s in enumerate(p2.load_samples(idx,'validation')):
   clean=s['target'][0];raw=s['raw'][0];rawerr=np.linalg.norm(raw['pred_cam_t'].cpu().numpy()-clean['t_all'],axis=1);rawrep=p.frame_reprojection(raw['pred_xy'].cpu(),s['meta']['gt_xyc'],s['meta']['size']);ci=8
   bucket=quality_bucket({'camera_error':rawerr[ci],'fit_residual':clean['fit'][ci],'upper_lower_disagreement':clean['ul'][ci],'reprojection_error':rawrep[ci]},thresholds);window_categories[wi]=bucket
   for c,condition in enumerate(CONDITIONS):
    for method in METHODS:
     out,used=output(method,models,s,c);meta=used['meta'];target=used['target'][c]['t_all'];pred=out['pred_cam_t'].detach().cpu().numpy();rawt=used['raw'][c]['pred_cam_t'].cpu().numpy();rep=p.frame_reprojection(out['pred_xy'].detach().cpu(),meta['gt_xyc'],meta['size'])
     order=np.argsort(meta['frame_id']);tm=trajectory_metrics(pred[order],np.asarray(target)[order]);key=(bucket,condition);wg[method][key].append(tm)
     for i in range(16):
      d=pred[i]-target[i];rows[method].append({'window':wi,'bucket':bucket,'condition':condition,'frame_id':int(meta['frame_id'][i]),'error':float(np.linalg.norm(d)),'xyz_abs':np.abs(d).tolist(),'z_error':float(abs(d[2])),'reproj':float(rep[i]),'correction':float(np.linalg.norm(pred[i]-rawt[i]))})
   count+=1
   if count%25==0:atomic_json(OUT/'final_validation_status.json',{'state':'running','windows':count,'total':614,'elapsed_seconds':time.time()-started})
 result={'protocol':{'split':'3DPW validation only','windows':count,'conditions':list(CONDITIONS),'methods':list(METHODS),'threshold_source':'fixed 3DPW train quantiles','test_used':False},'methods':{},'bucket_counts':{b:sum(v==b for v in window_categories.values()) for b in ('stable','medium','failure','other')}}
 for method in METHODS:
  result['methods'][method]={'overall':aggregate(rows[method],[x for v in wg[method].values() for x in v]),'conditions':{},'buckets':{}}
  for c in CONDITIONS:
   rr=[x for x in rows[method] if x['condition']==c];ww=[x for (b,cc),v in wg[method].items() if cc==c for x in v];result['methods'][method]['conditions'][c]=aggregate(rr,ww)
  for b in ('stable','medium','failure'):
   rr=[x for x in rows[method] if x['bucket']==b];ww=[x for (bb,c),v in wg[method].items() if bb==b for x in v];result['methods'][method]['buckets'][b]=aggregate(rr,ww) if rr else None
 # Window-level wins use absolute, velocity and acceleration on exactly matched keys.
 def wm(method):return {(b,c,i):aggregate([x for x in rows[method] if x['window']==i and x['condition']==c],wg[method][(b,c)][[j for j in range(i+1) if window_categories[j]==b].index(i)]) for i,b in window_categories.items() for c in CONDITIONS}
 # Avoid re-aggregating rows: compare stored trajectory plus per-window mean error.
 per={}
 for method in METHODS:
  per[method]={}
  counters=defaultdict(int)
  for i,b in window_categories.items():
   for c in CONDITIONS:
    j=counters[(b,c)];counters[(b,c)]+=1;rr=[x for x in rows[method] if x['window']==i and x['condition']==c];per[method][(i,c)]={'absolute':float(np.mean([x['error'] for x in rr])),**wg[method][(b,c)][j]}
 comparisons={}
 for base,name in (('raw_sam','framewise_vs_raw'),('framewise','true_vs_framewise'),('framewise_shuffled_temporal','true_vs_shuffled')):
  gains=np.asarray([per[base][k]['absolute']-per['framewise_true_temporal'][k]['absolute'] if name!='framewise_vs_raw' else per[base][k]['absolute']-per['framewise'][k]['absolute'] for k in per[base]])
  comparisons[name]={'absolute_mean_gain':float(gains.mean()),'absolute_win_count':int((gains>0).sum()),'total':len(gains)}
 comparisons['true_window_wins']={ref:{metric:int(sum(per['framewise_true_temporal'][k][metric]<per[ref][k][metric] for k in per[ref])) for metric in ('absolute','velocity_error_mean','acceleration_error_mean')} for ref in ('framewise','framewise_shuffled_temporal')}
 fw=result['methods']['framewise'];tr=result['methods']['framewise_true_temporal'];raw=result['methods']['raw_sam'];sh=result['methods']['framewise_shuffled_temporal']
 fast_keys=sorted(per['framewise'],key=lambda k:per['framewise'][k]['gt_velocity_magnitude']);fast_keys=fast_keys[int(.75*len(fast_keys)):];fast={'windows':len(fast_keys)}
 for method in ('framewise','framewise_true_temporal','framewise_shuffled_temporal'):fast[method]={'absolute':float(np.mean([per[method][k]['absolute'] for k in fast_keys])),'velocity_error':float(np.mean([per[method][k]['velocity_error_mean'] for k in fast_keys])),'velocity_amplitude_ratio':float(np.mean([per[method][k]['pred_velocity_magnitude']/max(per[method][k]['gt_velocity_magnitude'],1e-8) for k in fast_keys]))}
 stage_a='FRAMEWISE_CAMERA_GENERALIZATION_CONFIRMED' if fw['overall']['translation']['mean']<raw['overall']['translation']['mean'] else 'FRAMEWISE_CAMERA_GENERALIZATION_WEAK'
 abs_ok=tr['overall']['translation']['mean']<=1.02*fw['overall']['translation']['mean'];vel=tr['overall']['velocity_error_mean']<fw['overall']['velocity_error_mean'];acc=tr['overall']['acceleration_error_mean']<fw['overall']['acceleration_error_mean'];order=tr['overall']['velocity_error_mean']<sh['overall']['velocity_error_mean'];stable_ok=tr['buckets']['stable'] is None or tr['buckets']['stable']['translation']['mean']<=1.05*fw['buckets']['stable']['translation']['mean'];smooth=fast['framewise_true_temporal']['velocity_amplitude_ratio']<.8*fast['framewise']['velocity_amplitude_ratio']
 if stage_a.endswith('CONFIRMED') and abs_ok and vel and acc and order and stable_ok and not smooth:verdict='CAMERA_TEMPORAL_REFINEMENT_GENERALIZATION_CONFIRMED'
 elif stage_a.endswith('CONFIRMED') and abs_ok and (vel or acc):verdict='CAMERA_FRAMEWISE_CONFIRMED_TEMPORAL_AUXILIARY'
 elif stage_a.endswith('CONFIRMED'):verdict='CAMERA_FRAMEWISE_ONLY_PREFERRED'
 else:verdict='CAMERA_PHASE2_GENERALIZATION_WEAK'
 result.update({'comparisons':comparisons,'fast_motion':fast,'over_smoothing_risk':bool(smooth),'stage_a_verdict':stage_a,'stage_b_verdict':verdict,'verdict':verdict,'recommend_body_camera_integration':verdict in ('CAMERA_TEMPORAL_REFINEMENT_GENERALIZATION_CONFIRMED','CAMERA_FRAMEWISE_CONFIRMED_TEMPORAL_AUXILIARY'),'elapsed_seconds':time.time()-started})
 atomic_json(OUT/'final_validation_metrics.json',result);atomic_json(OUT/'final_summary.json',{'state':'complete',**{k:result[k] for k in ('stage_a_verdict','stage_b_verdict','verdict','recommend_body_camera_integration','comparisons','fast_motion','over_smoothing_risk','elapsed_seconds')}});atomic_json(OUT/'final_validation_status.json',{'state':'complete','windows':count,'total':614,'elapsed_seconds':time.time()-started});print(json.dumps({'state':'complete','verdict':verdict,'windows':count,'elapsed_seconds':time.time()-started}))
if __name__=='__main__':main()
