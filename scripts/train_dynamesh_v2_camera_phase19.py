"""Camera Phase-1.9 reliable multi-joint translation micro-overfit.

Train-only, one 16-frame window, frozen SAM heads, no visual inference and no
Meta conversion.  Prior Phase outputs are read-only comparison artifacts.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/root/sam-body4d-0.2.0')
OUT=ROOT/'outputs/dynamesh_v2_camera_phase19_multijoint_micro'
PHASE18=ROOT/'outputs/dynamesh_v2_camera_phase18_multijoint_gt'
CACHE=ROOT/'data/dynamesh_v2/cache_v2/train'
BODY=ROOT/'outputs/dynamesh_v2_phase2_training/best.pt'
PRIOR_OUTPUTS=(ROOT/'outputs/dynamesh_v2_camera_phase1',ROOT/'outputs/dynamesh_v2_camera_phase16_loss_rebalancing',ROOT/'outputs/dynamesh_v2_camera_phase17_supervision_audit',PHASE18)
CHECKPOINTS={'phase1':PRIOR_OUTPUTS[0]/'best.pt','phase16_c':PRIOR_OUTPUTS[1]/'exp_c/best.pt','phase17_reproj':PRIOR_OUTPUTS[2]/'reprojection_only/best.pt'}
CONDITIONS=('clean','occ15','occ35','occ55'); RATIOS=(0.,.15,.35,.55); SEED=20260902
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'models/sam_3d_body')]

from ablate_dynamesh_v2_camera_phase16 import adapter_forward, projection
from diagnose_dynamesh_v2_camera_phase15 import gradient_norm
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter
from train_dynamesh_v2_camera_phase1 import (assert_camera_optimizer_ownership,frame_reprojection,
    load_frozen_model,load_track,masked_normalized_reprojection_loss,metadata_for_track,
    decode_condition,reliability_features,set_seed,weighted_smooth_l1)
from validate_dynamesh_v2_camera_multijoint_gt import (SMPL_MODEL,fit_joint_subsets,gt_camera_joints,stats)


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); os.replace(tmp,path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def assert_train_only(split):
    if split!='train': raise ValueError('Camera Phase-1.9 accepts train split only')


def assert_loss_contract(names):
    if any('common_root' in name or 'common-root' in name for name in names): raise ValueError('common-root loss is forbidden')
    if set(names)!={'cam3d','reproj','reliability','anchor'}: raise ValueError('exact four-term loss contract required')


def assert_inference_reliability_fields(fields):
    forbidden={'gt_2d','gt_3d','t_all','fit_residual','upper_lower_disagreement','w_target'}
    leaked=forbidden.intersection(fields)
    if leaked: raise ValueError(f'GT/target leak into inference reliability: {sorted(leaked)}')


def derive_train_taus(rows):
    clean=[]
    for row in rows:
        assert_train_only(row.get('split','train'))
        if row['condition']=='clean': clean.append(row)
    if not clean: raise ValueError('train clean distribution is empty')
    fit=np.asarray([float(r['fit_residual']) for r in clean]); ul=np.asarray([float(r['upper_lower_disagreement']) for r in clean]); camera=np.asarray([float(r['camera_error']) for r in clean])
    fit=fit[np.isfinite(fit)]; ul=ul[np.isfinite(ul)];camera=camera[np.isfinite(camera)]
    if not len(fit) or not len(ul) or not len(camera): raise ValueError('finite train reliability distribution is empty')
    return {'tau_fit':float(np.median(fit)),'tau_ul':float(np.median(ul)),
            'finite_fit_count':int(len(fit)),'finite_ul_count':int(len(ul)),
            'camera_close_threshold':float(np.percentile(camera,25)),
            'source':'full 3DPW train clean median; fixed before micro-window selection'}


def target_reliability(fit,ul,tau_fit,tau_ul):
    if tau_fit<=0 or tau_ul<=0: raise ValueError('train-derived tau must be positive')
    fit,ul=np.asarray(fit,float),np.asarray(ul,float)
    return np.clip(np.exp(-fit/tau_fit)*np.exp(-ul/tau_ul),0,1)


def slice_aligned_track(track,sl):
    length=len(track['frame_id']); result={}
    for key,value in track.items():
        result[key]=value[sl] if isinstance(value,np.ndarray) and value.ndim>0 and value.shape[0]==length else value
    return result


def weighted_cam3d_loss(pred,target,weight):
    loss=F.smooth_l1_loss(pred,target.detach(),reduction='none').mean(-1)
    weight=torch.broadcast_to(weight.detach().to(loss),loss.shape)
    return (loss*weight).sum()/weight.sum().clamp_min(1e-8)


def read_phase18_rows():
    rows=[]
    with (PHASE18/'per_frame.csv').open() as f:
        for row in csv.DictReader(f):
            converted={'split':'train','sequence':row['sequence'],'condition':row['condition'],'person_id':int(row['person_id']),'frame_id':int(row['frame_id'])}
            for key,value in row.items():
                if key not in converted and key not in ('sequence','condition','person_id','frame_id'): converted[key]=float(value) if value else np.nan
            rows.append(converted)
    return rows


def select_case_a_candidates(rows,limit=10):
    summary=json.loads((PHASE18/'summary.json').read_text()); threshold=summary['joint_diagnosis']['thresholds']
    clean=[r for r in rows if r['condition']=='clean']; groups={}
    for row in clean: groups.setdefault((row['sequence'],row['person_id']),[]).append(row)
    candidates=[]
    for key,track_rows in groups.items():
        track_rows.sort(key=lambda r:r['frame_id']); path=CACHE/f'{key[0]}_person{key[1]}.npz'
        track=load_track(path); ids=track['frame_id'].astype(int); local=track['local_frame_index'].astype(int); lookup={int(v):i for i,v in enumerate(ids)}
        bbox=np.array([r['bbox_size'] for r in track_rows]); bbox_dyn=np.zeros(len(bbox)); bbox_dyn[1:]=np.abs(np.diff(np.log(np.maximum(bbox,1e-6))))
        row_lookup={r['frame_id']:i for i,r in enumerate(track_rows)}
        for row in track_rows:
            if not (row['camera_error']>=threshold['camera_error_p75'] and row['fit_residual']<=threshold['fit_p25'] and row['reprojection_error']>=threshold['reprojection_p75'] and row['body_teacher_geodesic_deg']<=threshold['body_teacher_p25']): continue
            idx=lookup[row['frame_id']]; start=max(0,min(idx-8,len(ids)-16))
            if len(ids[start:start+16])!=16 or not np.all(np.diff(local[start:start+16])==1): continue
            ri=row_lookup[row['frame_id']]; local_bbox=max(bbox_dyn[max(0,ri-1):min(len(bbox_dyn),ri+2)])
            window_errors=np.array([next(r['camera_error'] for r in track_rows if r['frame_id']==fid) for fid in ids[start:start+16]])
            contrast=row['camera_error']-min(window_errors[0],window_errors[-1])
            score=row['camera_error']/threshold['camera_error_p75']+row['reprojection_error']/threshold['reprojection_p75']-row['fit_residual']/threshold['fit_p25']-row['upper_lower_disagreement']/max(summary['conditions']['clean']['upper_lower_disagreement']['median'],1e-8)+max(contrast,0)/threshold['camera_error_p75']-5*local_bbox
            candidates.append({**{k:row[k] for k in ('sequence','person_id','frame_id','camera_error','z_abs_error','fit_residual','upper_lower_disagreement','reprojection_error','body_teacher_geodesic_deg','bbox_size')},'bbox_log_change':float(local_bbox),'window_start':int(start),'window_frame_ids':ids[start:start+16].tolist(),'score':float(score),'cache_path':str(path)})
    candidates.sort(key=lambda x:x['score'],reverse=True)
    if not candidates: raise RuntimeError('no legal clean train Case-A window')
    return candidates[:limit]


def prepare_candidate(candidate,model,smpl_model,taus):
    track=load_track(Path(candidate['cache_path'])); assert_train_only(str(track['split'].item())); full=metadata_for_track(track)
    start=int(candidate['window_start']); sl=slice(start,start+16)
    meta={k:(v[sl] if isinstance(v,np.ndarray) and len(v)==len(full['frame_id']) else v) for k,v in full.items()}; meta['sequence']=full['sequence']; meta['person']=full['person']
    if meta['frame_id'].astype(int).tolist()!=candidate['window_frame_ids']: raise ValueError('candidate/cache frame mismatch')
    tokens=torch.as_tensor(track['pose_token'][:,sl],device='cuda',dtype=torch.float32); raw=[]
    with torch.no_grad():
        for c in range(4): raw.append(decode_condition(model,track['pose_token'][c,sl],track['mhr_init_pose'][c,sl],meta))
    sliced=slice_aligned_track(track,sl)
    gt=gt_camera_joints(sliced,smpl_model,torch.device('cuda')); weights=np.clip(meta['gt_xyc'][...,2],0,None)
    target=[]
    for c in range(4):
        points=raw[c]['points3d'].detach().cpu().numpy(); t_all=[];t_up=[];t_lo=[];fit=[];ul=[]
        for i in range(16):
            fitted=fit_joint_subsets(points[i],gt[i],weights[i]); t_all.append(fitted['all']['translation']);t_up.append(fitted['upper']['translation']);t_lo.append(fitted['lower']['translation']);fit.append(fitted['all']['mean_residual']);ul.append(fitted['upper_lower_disagreement'])
        t_all,t_up,t_lo=np.asarray(t_all),np.asarray(t_up),np.asarray(t_lo); fit,ul=np.asarray(fit),np.asarray(ul)
        w=target_reliability(fit,ul,taus['tau_fit'],taus['tau_ul'])
        xyz=points+t_all[:,None]; K=np.broadcast_to(meta['K'],(16,3,3)); p=np.einsum('nij,nkj->nki',K,xyz); xy=p[...,:2]/p[...,2:]
        target_reproj=frame_reprojection(xy,meta['gt_xyc'],meta['size'])
        target.append({'t_all':t_all,'t_upper':t_up,'t_lower':t_lo,'fit':fit,'ul':ul,'weight':w,'target_reproj':target_reproj,'camera_close_threshold':taus['camera_close_threshold']})
    return track,meta,tokens,raw,target


def sanity_report(candidate,meta,raw,target):
    per_condition={}; passed=True
    for c,label in enumerate(CONDITIONS):
        t=target[c]; raw_t=raw[c]['pred_cam_t'].detach().cpu().numpy(); raw_rep=frame_reprojection(raw[c]['pred_xy'].detach().cpu(),meta['gt_xyc'],meta['size'])
        jump=np.linalg.norm(np.diff(t['t_all'],axis=0),axis=1)
        report={'target_depth':stats(t['t_all'][:,2]),'target_step':stats(jump),'fit_residual':stats(t['fit']),'upper_lower_disagreement':stats(t['ul']),'weight':stats(t['weight']),'raw_reprojection':stats(raw_rep),'target_reprojection':stats(t['target_reproj']),'raw_camera_error':stats(np.linalg.norm(raw_t-t['t_all'],axis=1))}
        report['passed']=bool(np.isfinite(t['t_all']).all() and jump.max()<.5 and np.mean(t['target_reproj'])<np.mean(raw_rep) and np.percentile(t['ul'],95)<.10)
        passed &= report['passed']; per_condition[label]=report
    return {'candidate':candidate,'sequence':meta['sequence'],'person_id':meta['person'],'frame_ids':meta['frame_id'].astype(int).tolist(),'conditions':per_condition,'passed':bool(passed),'reselection_count':0}


def phase19_forward(adapter,model,tokens,meta,raw,target,condition):
    features=reliability_features(tokens[condition][None],meta,RATIOS[condition])
    assert_inference_reliability_fields({'visibility','bbox_center_delta','bbox_scale_delta','temporal_latent_disagreement'})
    refined,reliability=adapter_forward(adapter,tokens[condition][None],features,'learned')
    cam=model.head_camera(refined.squeeze(0),model.init_camera.weight.expand(16,-1)); projected=projection(model,raw[condition]['points3d'].detach(),cam,meta)
    t=torch.as_tensor(target[condition]['t_all'],device='cuda',dtype=torch.float32); w=torch.as_tensor(target[condition]['weight'],device='cuda',dtype=torch.float32)
    gt=torch.as_tensor(meta['gt_xyc'],device='cuda'); bbox=torch.as_tensor(meta['size'],device='cuda')
    raw_t=raw[condition]['pred_cam_t'].detach(); raw_error=torch.linalg.vector_norm(raw_t-t,dim=-1); tau_cam=torch.as_tensor(target[condition]['camera_close_threshold'],device='cuda').clamp_min(1e-6)
    anchor_weight=w*torch.exp(-raw_error/tau_cam)
    terms={'cam3d':weighted_cam3d_loss(projected['pred_cam_t'],t,w),
           'reproj':masked_normalized_reprojection_loss(projected['pred_keypoints_2d'],gt,bbox),
           'reliability':F.binary_cross_entropy(reliability.reshape(-1),w.detach()),
           'anchor':weighted_smooth_l1(cam,raw[condition]['pred_cam'].detach(),anchor_weight)}
    assert_loss_contract(tuple(terms))
    return terms,{'pred_cam':cam,'pred_cam_t':projected['pred_cam_t'],'pred_xy':projected['pred_keypoints_2d'],'reliability':reliability.reshape(-1),'refined_token':refined}


def balance_gradient_lambdas(raw_norm):
    if raw_norm['cam3d']<=0 or raw_norm['reproj']<=0 or raw_norm['reliability']<=0: raise ValueError('three core gradients must be positive')
    ref=float(raw_norm['cam3d']); desired={'cam3d':ref,'reproj':ref,'reliability':ref/4}
    lambdas={name:desired[name]/float(raw_norm[name]) for name in desired}; lambdas['cam3d']=1.; lambdas['anchor']=.01
    if any(not np.isfinite(v) or v<=0 or v>=100 for v in lambdas.values()): raise ValueError(f'unsafe lambda balance: {lambdas}')
    weighted={name:float(raw_norm[name])*lambdas[name] for name in raw_norm}
    core=[weighted[name] for name in ('cam3d','reproj','reliability')]
    return {'lambdas':lambdas,'weighted_gradient_norms':weighted,'core_max_min_ratio':max(core)/min(core),'anchor_policy':'zero at exact Identity; fixed small lambda=0.01, excluded from core ratio'}


def gradient_probe(adapter,model,tokens,meta,raw,target):
    averaged={}; per=[phase19_forward(adapter,model,tokens,meta,raw,target,c)[0] for c in range(4)]
    for name in ('cam3d','reproj','reliability','anchor'): averaged[name]=torch.stack([x[name] for x in per]).mean()
    params=[p for p in adapter.parameters() if p.requires_grad]; raw_norm={name:gradient_norm(loss,params) for name,loss in averaged.items()}
    balance=balance_gradient_lambdas(raw_norm)
    return {'raw_losses':{k:float(v.detach()) for k,v in averaged.items()},'raw_gradient_norms':raw_norm,**balance,'adjustments':1}


def total_loss(terms,lambdas): return sum(lambdas[k]*terms[k] for k in terms)


def train(adapter,model,tokens,meta,raw,target,lambdas,steps=400,lr=1e-3):
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=lr,weight_decay=1e-4); assert_camera_optimizer_ownership(optimizer,adapter)
    if any(p.requires_grad for p in model.parameters()): raise AssertionError('SAM/head parameters must remain frozen')
    history=[];best=math.inf;best_step=0
    for step in range(1,steps+1):
        adapter.train(); optimizer.zero_grad(set_to_none=True); c=(step-1)%4; terms,_=phase19_forward(adapter,model,tokens,meta,raw,target,c); loss=total_loss(terms,lambdas)
        if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss step {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step()
        if step%4==0:
            adapter.eval()
            with torch.no_grad():
                vals=[]
                for condition in range(4): vals.append(float(total_loss(phase19_forward(adapter,model,tokens,meta,raw,target,condition)[0],lambdas)))
            avg=float(np.mean(vals)); history.append({'step':step,'condition_total':vals,'mean_total':avg,'train_terms':{k:float(v.detach()) for k,v in terms.items()}})
            if avg<best:
                best,best_step=avg,step;torch.save({'adapter_state_dict':adapter.state_dict(),'best_step':step,'best_total':best,'lambdas':lambdas},OUT/'best.pt')
            if step%40==0: atomic_json(OUT/'status.json',{'state':'training','step':step,'best_step':best_step,'best_total':best,'pid':os.getpid()});print(json.dumps({'step':step,'mean_total':avg,'best':best}),flush=True)
    torch.save({'adapter_state_dict':adapter.state_dict(),'final_step':steps,'lambdas':lambdas},OUT/'final.pt');return history,best_step,best


def load_adapter(path):
    adapter=CameraFeatureTemporalAdapter().cuda(); ckpt=torch.load(path,map_location='cuda',weights_only=False); adapter.load_state_dict(ckpt['adapter_state_dict'],strict=True);adapter.eval();return adapter


@torch.no_grad()
def evaluate_method(name,adapter,gate_mode,model,tokens,meta,raw,target):
    result={}
    for c,label in enumerate(CONDITIONS):
        if adapter is None: out={'pred_cam':raw[c]['pred_cam'],'pred_cam_t':raw[c]['pred_cam_t'],'pred_xy':raw[c]['pred_xy'],'reliability':torch.ones(16,device='cuda')}
        else:
            features=reliability_features(tokens[c][None],meta,RATIOS[c]);refined,reliability=adapter_forward(adapter,tokens[c][None],features,gate_mode);cam=model.head_camera(refined.squeeze(0),model.init_camera.weight.expand(16,-1));proj=projection(model,raw[c]['points3d'].detach(),cam,meta);out={'pred_cam':cam,'pred_cam_t':proj['pred_cam_t'],'pred_xy':proj['pred_keypoints_2d'],'reliability':reliability.reshape(-1)}
        t=target[c]; pred_t=out['pred_cam_t'].cpu().numpy();raw_t=raw[c]['pred_cam_t'].cpu().numpy(); cam=out['pred_cam'].cpu().numpy();raw_cam=raw[c]['pred_cam'].cpu().numpy(); reliability=out['reliability'].cpu().numpy(); rep=frame_reprojection(out['pred_xy'].cpu(),meta['gt_xyc'],meta['size']); raw_rep=frame_reprojection(raw[c]['pred_xy'].cpu(),meta['gt_xyc'],meta['size']); err=np.linalg.norm(pred_t-t['t_all'],axis=1);raw_err=np.linalg.norm(raw_t-t['t_all'],axis=1); center=int(np.argmax(raw_err)); stable=(t['weight']>=np.percentile(t['weight'],75))&(raw_err<=t['camera_close_threshold']);stable[center]=False
        rel_corr=float(np.corrcoef(reliability,t['weight'])[0,1]) if np.std(reliability)>1e-8 and np.std(t['weight'])>1e-8 else None
        neighbours=[i for i in (center-1,center+1) if 0<=i<16]
        result[label]={'center_index':center,'center_frame':int(meta['frame_id'][center]),'translation_error':stats(err),'weighted_translation_error':float(np.sum(err*t['weight'])/np.sum(t['weight'])),'xyz_abs_error_mean':np.mean(np.abs(pred_t-t['t_all']),axis=0).tolist(),'improvement_percent':float(100*(raw_err.mean()-err.mean())/raw_err.mean()),'reprojection':stats(rep),'raw_reprojection_mean':float(raw_rep.mean()),'z':{'raw_mean':float(raw_t[:,2].mean()),'refined_mean':float(pred_t[:,2].mean()),'target_mean':float(t['t_all'][:,2].mean()),'raw_range':float(np.ptp(raw_t[:,2])),'refined_range':float(np.ptp(pred_t[:,2]))},'scale':{'raw_mean':float(raw_cam[:,0].mean()),'refined_mean':float(cam[:,0].mean())},'reliability':stats(reliability),'reliability_target_correlation':rel_corr,'center':{'raw_t':raw_t[center].tolist(),'pred_t':pred_t[center].tolist(),'target_t':t['t_all'][center].tolist(),'raw_error':float(raw_err[center]),'pred_error':float(err[center]),'reprojection':float(rep[center]),'weight':float(t['weight'][center]),'reliability':float(reliability[center])},'neighbors':[{'frame_id':int(meta['frame_id'][i]),'raw_error':float(raw_err[i]),'pred_error':float(err[i]),'weight':float(t['weight'][i]),'reliability':float(reliability[i]),'reprojection':float(rep[i])} for i in neighbours],'stable_frames':{'definition':f'w_target>=window_P75 and raw_error<=train_clean_P25={t["camera_close_threshold"]:.6f}m','count':int(stable.sum()),'pred_cam_deviation':float(np.linalg.norm(cam[stable]-raw_cam[stable],axis=1).mean()) if stable.any() else None,'pred_cam_t_deviation':float(np.linalg.norm(pred_t[stable]-raw_t[stable],axis=1).mean()) if stable.any() else None,'reprojection_change':float((rep[stable]-raw_rep[stable]).mean()) if stable.any() else None,'camera_error_change':float((err[stable]-raw_err[stable]).mean()) if stable.any() else None},'per_frame':{'frame_id':meta['frame_id'].astype(int).tolist(),'raw_error':raw_err.tolist(),'pred_error':err.tolist(),'reprojection':rep.tolist(),'weight':t['weight'].tolist(),'reliability':reliability.tolist(),'raw_z':raw_t[:,2].tolist(),'pred_z':pred_t[:,2].tolist(),'target_z':t['t_all'][:,2].tolist()}}
    return result


def run(steps=400,lr=1e-3):
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);set_seed();body_hash=sha256(BODY)
    rows=read_phase18_rows();taus=derive_train_taus(rows);top=select_case_a_candidates(rows,10);atomic_json(OUT/'top_candidates.json',{'top10':top})
    model=load_frozen_model();import smplx
    smpl_model=smplx.SMPL(model_path=str(SMPL_MODEL),batch_size=1).cuda().eval();selected=None
    for attempt,candidate in enumerate(top[:2]):
        track,meta,tokens,raw,target=prepare_candidate(candidate,model,smpl_model,taus);sanity=sanity_report(candidate,meta,raw,target);sanity['reselection_count']=attempt
        if sanity['passed']: selected=(track,meta,tokens,raw,target,sanity);break
    if selected is None: raise RuntimeError('two allowed Case-A windows failed target sanity')
    track,meta,tokens,raw,target,sanity=selected;atomic_json(OUT/'target_sanity.json',sanity)
    config={'seed':SEED,'steps':steps,'lr':lr,'window_length':16,'conditions':CONDITIONS,'taus':taus,'losses':['cam3d','reproj','reliability','anchor'],'trainable':'CameraFeatureTemporalAdapter only','forbidden':'Body/SAM/MHRHead/PerspectiveHead/DuoMo/common-root/Meta converter/visual inference','body_sha256':body_hash};atomic_json(OUT/'config.json',config)
    adapter=CameraFeatureTemporalAdapter().cuda();probe=gradient_probe(adapter,model,tokens,meta,raw,target);atomic_json(OUT/'loss_scale_probe.json',probe)
    history,best_step,best=train(adapter,model,tokens,meta,raw,target,probe['lambdas'],steps,lr);atomic_json(OUT/'history.json',{'history':history})
    methods={'raw':(None,'learned'),'phase1':(load_adapter(CHECKPOINTS['phase1']),'learned'),'phase16_c':(load_adapter(CHECKPOINTS['phase16_c']),'fixed_open'),'phase17_reproj':(load_adapter(CHECKPOINTS['phase17_reproj']),'fixed_open'),'phase19':(load_adapter(OUT/'best.pt'),'learned')}
    comparison={name:evaluate_method(name,*method,model,tokens,meta,raw,target) for name,method in methods.items()}
    clean=comparison['phase19']['clean']; stable_change=clean['stable_frames']['camera_error_change']; stable_ok=stable_change is None or stable_change<=.1
    good=clean['improvement_percent']>=20 and clean['reprojection']['mean']<=1.05*clean['raw_reprojection_mean'] and stable_ok
    instability=clean['improvement_percent']>0 and (clean['reprojection']['mean']>1.1*clean['raw_reprojection_mean'] or (stable_change is not None and stable_change>.25))
    verdict='MULTI_JOINT_TARGET_LEARNS_BUT_GENERALIZATION_UNKNOWN' if good else 'MULTI_JOINT_SUPERVISION_CAUSES_INSTABILITY' if instability else 'CAMERA_RECOVERY_STILL_WEAK'
    summary={'state':'complete','verdict':verdict,'elapsed_seconds':time.time()-started,'selected_window':sanity,'taus':taus,'gradient_probe':probe,'training':{'steps':steps,'best_step':best_step,'best_total':best},'comparison':comparison,'target':{label:{'translation':target[c]['t_all'].tolist(),'reprojection':stats(target[c]['target_reproj']),'weight':stats(target[c]['weight'])} for c,label in enumerate(CONDITIONS)},'prior_checkpoints':{k:str(v) for k,v in CHECKPOINTS.items()},'body_sha256':body_hash}
    if sha256(BODY)!=body_hash: raise RuntimeError('Body checkpoint changed')
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'status.json',{'state':'complete','verdict':verdict,'best_step':best_step,'pid':os.getpid()});print(json.dumps({'verdict':verdict,'best_step':best_step,'elapsed':summary['elapsed_seconds']}),flush=True);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=400);parser.add_argument('--lr',type=float,default=1e-3);args=parser.parse_args();run(args.steps,args.lr)


if __name__=='__main__':main()
