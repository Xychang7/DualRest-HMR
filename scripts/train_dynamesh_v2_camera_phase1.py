"""DynaMesh4D-v2 Camera Phase-1 clean failure mining and micro-overfit."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, pickle, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT=Path('/root/sam-body4d-0.2.0')
DATA=Path('/root/data1/sam-body4d-master/dataset/3DPW')
CACHE=ROOT/'data/dynamesh_v2/cache_v2'
OUT=ROOT/'outputs/dynamesh_v2_camera_phase1'
SEED=20260901
CONDITIONS=('clean','occ15','occ35','occ55')
RATIOS=(0.,.15,.35,.55)
MHR_IDX=np.array([5,6,7,8,62,41,9,10,11,12,13,14])
OP_IDX=np.array([5,2,6,3,7,4,11,8,12,9,13,10])
sys.path[:0]=[str(ROOT),str(ROOT/'models/sam_3d_body'),str(ROOT/'scripts')]

from build_dynamesh_v2_cache_sanity import bbox_from_pose2d, load_pickle
from mine_dynamesh_v2_camera_failures import mine_records,is_camera_specific_candidate
from sam_3d_body import load_sam_3d_body
from sam_3d_body.models.modules.camera_temporal import CameraFeatureTemporalAdapter,CameraReliabilityFeatures


def assert_aligned_camera_fields(pose_token, init_state, frame_id, person_id, conditions):
    token,init,frames=np.asarray(pose_token),np.asarray(init_state),np.asarray(frame_id)
    if token.ndim!=3 or token.shape[-1]!=1024: raise ValueError('pose_token must be (C,T,1024)')
    if init.ndim!=3 or init.shape[:2]!=token.shape[:2]: raise ValueError('variant init state must align with token condition/frame axes')
    if len(conditions)!=token.shape[0] or frames.shape!=(token.shape[1],): raise ValueError('condition/frame metadata alignment mismatch')
    if len(np.unique(frames))!=len(frames) or np.any(np.diff(frames)<=0): raise ValueError('frame IDs must be strictly increasing and unique')
    int(person_id)


def assert_projection_equivalence(expected,actual,atol=1e-5):
    expected,actual=np.asarray(expected),np.asarray(actual)
    if expected.shape!=actual.shape or not np.isfinite(actual).all(): raise ValueError('projection equivalence shape/finite failure')
    error=float(np.max(np.abs(expected-actual)))
    if error>atol: raise ValueError(f'projection contract differs by {error}, tolerance={atol}')
    return error


def soft_reliability_target(normalized_raw_reprojection_error,scale):
    if scale<=0: raise ValueError('reliability scale must be positive')
    error=torch.as_tensor(normalized_raw_reprojection_error)
    return torch.exp(-torch.clamp(error,min=0)/float(scale)).clamp(0,1)


def teacher_stability_weight(reliability_target,observation_stability):
    reliability_target,observation_stability=torch.broadcast_tensors(reliability_target,observation_stability)
    return reliability_target.detach().clamp(0,1)*observation_stability.detach().clamp(0,1)


def assert_inference_reliability_contract(features):
    leaked={'gt_2d','gt_reprojection_error','reprojection_residual'}.intersection(features)
    if leaked: raise ValueError(f'GT/reprojection leakage into inference reliability features: {sorted(leaked)}')


def masked_normalized_reprojection_loss(pred_xy,gt_xyc,bbox_size):
    if pred_xy.shape!=gt_xyc[...,:2].shape: raise ValueError('predicted and GT 2D shapes differ')
    valid=torch.isfinite(gt_xyc).all(-1)&(gt_xyc[...,2]>0)&torch.isfinite(pred_xy).all(-1)
    if not torch.any(valid): raise ValueError('no valid confidence-masked 2D joints')
    scale=bbox_size[...,None].expand_as(valid).to(pred_xy).clamp_min(1e-8)
    error=torch.linalg.vector_norm(pred_xy-gt_xyc[...,:2],dim=-1)/scale
    return error[valid].mean()


def weighted_smooth_l1(prediction,target,weight):
    loss=F.smooth_l1_loss(prediction,target.detach(),reduction='none').mean(-1)
    weight=torch.broadcast_to(weight.to(loss),loss.shape)
    return (loss*weight).sum()/weight.sum().clamp_min(1e-8)


def assert_camera_optimizer_ownership(optimizer,camera_adapter):
    expected={id(p) for p in camera_adapter.parameters() if p.requires_grad}
    actual={id(p) for group in optimizer.param_groups for p in group['params']}
    if actual!=expected: raise AssertionError('optimizer must own exactly CameraFeatureTemporalAdapter trainable parameters')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def load_frozen_model():
    cfg=OmegaConf.load(ROOT/'configs/body4d.yaml')
    model,_=load_sam_3d_body(cfg.sam_3d_body.ckpt_path,device=torch.device('cuda'),mhr_path=cfg.sam_3d_body.mhr_path)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def load_track(path):
    with np.load(path,allow_pickle=False) as z: track={k:np.asarray(z[k]) for k in z.files}
    assert_aligned_camera_fields(track['pose_token'],track['mhr_init_pose'],track['frame_id'],track['person_id'].item(),tuple(track['condition_names'].astype(str)))
    if str(track['split'].item())!='train': raise ValueError('Camera Phase-1 accepts train cache only')
    return track


def metadata_for_track(track):
    sequence=str(track['sequence'].item()); person=int(track['person_id'].item())
    ann=load_pickle(DATA/'sequenceFiles/train'/f'{sequence}.pkl')
    selected=np.asarray(track['local_frame_index'],np.int64)
    frame_ids=np.asarray(ann['img_frame_ids'],np.int64)[selected]
    if not np.array_equal(frame_ids,track['frame_id']): raise ValueError(f'{sequence}/person{person}: annotation/cache frame mismatch')
    image_path=DATA/'imageFiles'/sequence/f'image_{selected[0]:05d}.jpg'; image=cv2.imread(str(image_path))
    if image is None: raise FileNotFoundError(image_path)
    h,w=image.shape[:2]; poses=np.asarray(ann['poses2d'][person])
    boxes=np.stack([bbox_from_pose2d(poses[i],w,h) for i in selected]).astype(np.float32)
    center=(boxes[:,:2]+boxes[:,2:])/2.; size=(boxes[:,2]-boxes[:,0])*1.25
    K=np.asarray(ann['cam_intrinsics'],np.float32)
    gt=np.transpose(poses[selected][:,:,OP_IDX],(0,2,1)).astype(np.float32)
    return {'sequence':sequence,'person':person,'selected':selected,'frame_id':frame_ids,'boxes':boxes,'center':center,'size':size,'K':K,'image_size':np.tile(np.array([[w,h]],np.float32),(len(selected),1)),'gt_xyc':gt}


def decode_condition(model,tokens,init,meta,batch_size=32,grad_camera_token=None):
    device=torch.device('cuda'); all_cam=[]; all_t=[]; all_xy=[]; all_kps=[]
    for start in range(0,len(tokens),batch_size):
        stop=min(start+batch_size,len(tokens)); raw=torch.as_tensor(tokens[start:stop],device=device,dtype=torch.float32); pose_init=torch.as_tensor(init[start:stop],device=device)
        with torch.no_grad(): pose=model.head_pose(raw,pose_init); points=pose['pred_keypoints_3d'][:,MHR_IDX].detach()
        camera_token=raw if grad_camera_token is None else grad_camera_token[start:stop]
        cam=model.head_camera(camera_token,model.init_camera.weight.expand(stop-start,-1))
        projection=model.head_camera.perspective_projection(points,cam,torch.as_tensor(meta['center'][start:stop],device=device),torch.as_tensor(meta['size'][start:stop],device=device),torch.as_tensor(meta['image_size'][start:stop],device=device),torch.as_tensor(np.broadcast_to(meta['K'],(stop-start,3,3)).copy(),device=device))
        all_cam.append(cam); all_t.append(projection['pred_cam_t']); all_xy.append(projection['pred_keypoints_2d']); all_kps.append(points)
    return {k:torch.cat(v) for k,v in [('pred_cam',all_cam),('pred_cam_t',all_t),('pred_xy',all_xy),('points3d',all_kps)]}


def frame_reprojection(pred_xy,gt_xyc,size):
    pred=np.asarray(pred_xy); gt=np.asarray(gt_xyc); valid=np.isfinite(gt).all(-1)&(gt[...,2]>0)&np.isfinite(pred).all(-1)
    values=[]
    for i in range(len(pred)):
        if not valid[i].any(): values.append(np.nan)
        else: values.append(float(np.linalg.norm(pred[i,valid[i]]-gt[i,valid[i],:2],axis=-1).mean()/size[i]))
    return np.asarray(values)


def centred_innovation(values):
    values=np.asarray(values,np.float64); out=np.zeros(len(values),np.float64)
    if len(values)>2: out[1:-1]=np.linalg.norm(values[1:-1]-.5*(values[:-2]+values[2:]),axis=-1)
    return out


def deltas(values,normalizer=1.):
    values=np.asarray(values,np.float64); out=np.zeros_like(values); normalizer=np.asarray(normalizer,np.float64)
    denominator=normalizer[1:] if normalizer.ndim>0 and normalizer.shape[0]==values.shape[0] else normalizer
    out[1:]=(values[1:]-values[:-1])/denominator
    return out


def equivalence_gate():
    raw_path=ROOT/'outputs/3dpw_protocol_isolation_50f.pkl'; csv_path=ROOT/'outputs/dynamesh_v2_camera_reprojection_gate_50f/per_frame.csv'
    with raw_path.open('rb') as f: raw=pickle.load(f,encoding='latin1')
    ann=load_pickle(DATA/'sequenceFiles/test/flat_guitar_01.pkl'); selected=np.asarray(raw['frame_indices'],np.int64)
    image=cv2.imread(str(DATA/'imageFiles/flat_guitar_01'/f'image_{selected[0]:05d}.jpg')); h,w=image.shape[:2]
    poses=np.asarray(ann['poses2d'][0]); boxes=np.stack([bbox_from_pose2d(poses[i],w,h) for i in selected])
    # Existing formal records serialize bbox coordinates as float32.  At the
    # observed ~1k-pixel magnitude, one float32 quantization step is
    # 0.0001220703125 px; use a two-step absolute tolerance, not a fitted one.
    box_error=assert_projection_equivalence(np.asarray(raw['bboxes']),boxes,atol=2e-4)
    k_error=assert_projection_equivalence(np.asarray(raw['cam_intrinsics']),np.asarray(ann['cam_intrinsics']),atol=1e-6)
    frame_ids=np.asarray(ann['img_frame_ids'],np.int64)[selected]; expected=np.arange(0,100,2)
    if not np.array_equal(frame_ids,expected): raise ValueError('formal record frame mapping mismatch')
    center=(boxes[:,:2]+boxes[:,2:])/2.; size=(boxes[:,2]-boxes[:,0])*1.25; K=np.asarray(ann['cam_intrinsics'])
    points=np.stack([x['pred_keypoints_3d'][MHR_IDX] for x in raw['dyna']]); cam_t=np.stack([x['pred_cam_t'] for x in raw['dyna']]); xyz=points+cam_t[:,None,:]
    projected=np.einsum('ij,nkj->nki',K,xyz); projected=projected[...,:2]/projected[...,2:]
    gt=np.transpose(poses[selected][:,:,OP_IDX],(0,2,1)); errors=frame_reprojection(projected,gt,size)
    with csv_path.open() as f: rows=list(csv.DictReader(f))
    saved=np.asarray([float(x['mean_normalized_error']) for x in rows])
    metric_error=assert_projection_equivalence(saved,errors,atol=1e-6)
    result={'passed':True,'reference':'existing formal 50-frame record + saved reprojection Gate','bbox_max_abs':box_error,'intrinsics_max_abs':k_error,'normalized_reprojection_max_abs':metric_error,'frames':50}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'equivalence_gate.json').write_text(json.dumps(result,indent=2)+'\n'); return result


def mine_all():
    gate=equivalence_gate(); model=load_frozen_model(); records=[]; per_track={}
    paths=sorted((CACHE/'train').glob('*.npz'))
    for number,path in enumerate(paths,1):
        track=load_track(path); meta=metadata_for_track(track)
        decoded=decode_condition(model,track['pose_token'][0],track['mhr_init_pose'][0],meta)
        cam=decoded['pred_cam'].detach().cpu().numpy(); xy=decoded['pred_xy'].detach().cpu().numpy(); reproj=frame_reprojection(xy,meta['gt_xyc'],meta['size'])
        body=centred_innovation(np.asarray(track['body_cont'][0])/math.sqrt(260)); center_delta=deltas(meta['center'],meta['size'][:,None]); scale_delta=deltas(np.log(np.maximum(meta['size'],1e-6)))
        for i,fid in enumerate(meta['frame_id']):
            records.append({'sequence':meta['sequence'],'person_id':meta['person'],'frame_id':int(fid),'condition':'clean','pred_cam':cam[i].tolist(),'reprojection_error':float(reproj[i]),'body_temporal_innovation':float(body[i]),'visibility':int(track['visibility'][i]),'bbox_center_delta':center_delta[i].tolist(),'bbox_scale_delta':float(scale_delta[i]),'mask_area_delta':None})
        per_track[f'{meta["sequence"]}/person{meta["person"]}']={'frames':len(meta['frame_id']),'mean_reprojection':float(np.nanmean(reproj))}
        print(json.dumps({'mined_track':number,'total_tracks':len(paths),'track':str(path),'frames':len(reproj)}),flush=True)
    candidates=mine_records(records,'train','clean')
    # Require an isolated positive reprojection spike and a valid 16-frame contiguous cache window.
    selected_candidate=None
    for candidate in candidates:
        path=CACHE/'train'/f'{candidate["sequence"]}_person{candidate["person_id"]}.npz'; track=load_track(path); ids=track['frame_id']; loc=track['local_frame_index']; idx=int(np.where(ids==candidate['frame_id'])[0][0])
        start=max(0,min(idx-8,len(ids)-16));
        if is_camera_specific_candidate(candidate) and np.all(np.diff(loc[start:start+16])==1):
            selected_candidate={**candidate,'cache_path':str(path),'cache_start':start,'window_frame_ids':ids[start:start+16].astype(int).tolist()}; break
    if selected_candidate is None: raise RuntimeError('no legal 16-frame clean failure window found')
    result={'gate':gate,'split':'train','condition':'clean','records':len(records),'tracks':len(paths),'top_candidates':candidates[:10],'selected':selected_candidate,'diagnostic_candidates':candidates[1:3]}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'mining.json').write_text(json.dumps(result,indent=2)+'\n'); return result


def reliability_features(tokens,meta,ratio):
    tokens=tokens.float(); disagreement=torch.zeros(tokens.shape[:2]+(1,),device=tokens.device,dtype=tokens.dtype)
    if tokens.shape[1]>2: disagreement[:,1:-1,0]=torch.linalg.vector_norm(tokens[:,1:-1]-.5*(tokens[:,:-2]+tokens[:,2:]),dim=-1)/math.sqrt(tokens.shape[-1])
    scale=torch.quantile(disagreement.detach().flatten(),.75).clamp_min(1e-6); disagreement=(disagreement/scale).clamp(0,4)
    center=torch.as_tensor(meta['center'],device=tokens.device,dtype=tokens.dtype); size=torch.as_tensor(meta['size'],device=tokens.device,dtype=tokens.dtype)
    cdelta=torch.zeros((len(size),2),device=tokens.device,dtype=tokens.dtype); cdelta[1:]=(center[1:]-center[:-1])/size[1:,None].clamp_min(1e-6)
    sdelta=torch.zeros((len(size),1),device=tokens.device,dtype=tokens.dtype); sdelta[1:,0]=torch.log(size[1:].clamp_min(1e-6)/size[:-1].clamp_min(1e-6))
    visibility=torch.full((1,len(size),1),1.-ratio,device=tokens.device,dtype=tokens.dtype)
    features=CameraReliabilityFeatures(visibility,cdelta[None],sdelta[None],None,disagreement)
    assert_inference_reliability_contract({'visibility':visibility,'bbox_center_delta':cdelta,'bbox_scale_delta':sdelta,'temporal_latent_disagreement':disagreement})
    return features


def prepare_window(model,mining):
    chosen=mining['selected']; track=load_track(Path(chosen['cache_path'])); full=metadata_for_track(track); start=int(chosen['cache_start']); sl=slice(start,start+16)
    meta={k:(v[sl] if isinstance(v,np.ndarray) and len(v)==len(full['frame_id']) else v) for k,v in full.items()}; meta['sequence']=full['sequence']; meta['person']=full['person']
    tokens=torch.as_tensor(track['pose_token'][:,sl],device='cuda',dtype=torch.float32); init=track['mhr_init_pose'][:,sl]
    raw=[]
    for condition in range(4): raw.append(decode_condition(model,track['pose_token'][condition,sl],init[condition],meta))
    scale=float(np.nanmedian(frame_reprojection(raw[0]['pred_xy'].detach().cpu(),meta['gt_xyc'],meta['size']))); scale=max(scale,.01)
    return track,meta,tokens,init,raw,scale


def camera_loss_terms(adapter,model,tokens,meta,raw,scale,condition):
    ratio=RATIOS[condition]; selected=tokens[condition][None]; features=reliability_features(selected,meta,ratio); refined,reliability=adapter(selected,features)
    points=raw[condition]['points3d'].detach(); cam=model.head_camera(refined.squeeze(0),model.init_camera.weight.expand(16,-1)); proj=model.head_camera.perspective_projection(points,cam,torch.as_tensor(meta['center'],device='cuda'),torch.as_tensor(meta['size'],device='cuda'),torch.as_tensor(meta['image_size'],device='cuda'),torch.as_tensor(np.broadcast_to(meta['K'],(16,3,3)).copy(),device='cuda'))
    gt=torch.as_tensor(meta['gt_xyc'],device='cuda'); bbox=torch.as_tensor(meta['size'],device='cuda')
    raw_err=[]
    raw_xy=raw[condition]['pred_xy'].detach()
    for t in range(16): raw_err.append(masked_normalized_reprojection_loss(raw_xy[t:t+1],gt[t:t+1],bbox[t:t+1]))
    raw_err=torch.stack(raw_err); target=soft_reliability_target(raw_err,scale)
    clean_err=[]; clean_xy=raw[0]['pred_xy'].detach()
    for t in range(16): clean_err.append(masked_normalized_reprojection_loss(clean_xy[t:t+1],gt[t:t+1],bbox[t:t+1]))
    clean_target=soft_reliability_target(torch.stack(clean_err),scale)
    cdelta=features.bbox_center_delta.squeeze(0); sdelta=features.bbox_scale_delta.squeeze(0); stability=torch.exp(-10*(torch.linalg.vector_norm(cdelta,dim=-1)+torch.abs(sdelta[:,0]))).clamp(0,1)
    teacher_weight=teacher_stability_weight(clean_target,stability)
    teacher=weighted_smooth_l1(cam,raw[0]['pred_cam'].detach(),teacher_weight)
    reproj=masked_normalized_reprojection_loss(proj['pred_keypoints_2d'],gt,bbox)
    rel=F.binary_cross_entropy(reliability.squeeze(0).squeeze(-1),target.detach())
    anchor=weighted_smooth_l1(cam,raw[condition]['pred_cam'].detach(),teacher_stability_weight(target,stability))
    return {'teacher':teacher,'reproj':reproj,'reliability':rel,'anchor':anchor}, {'refined':refined,'reliability':reliability,'pred_cam':cam,'pred_cam_t':proj['pred_cam_t'],'pred_xy':proj['pred_keypoints_2d'],'target':target,'teacher_weight':teacher_weight}


def probe():
    mining=json.loads((OUT/'mining.json').read_text()) if (OUT/'mining.json').exists() else mine_all(); model=load_frozen_model(); _,meta,tokens,_,raw,scale=prepare_window(model,mining); adapter=CameraFeatureTemporalAdapter().cuda(); values={}
    for c,name in enumerate(CONDITIONS):
        terms,_=camera_loss_terms(adapter,model,tokens,meta,raw,scale,c); values[name]={k:float(v.detach()) for k,v in terms.items()}
    means={k:float(np.mean([values[c][k] for c in CONDITIONS])) for k in ('teacher','reproj','reliability','anchor')}
    result={'reliability_scale':scale,'per_condition':values,'mean_raw_magnitudes':means,'note':'forward-only; no optimizer step'}; (OUT/'loss_scale_probe.json').write_text(json.dumps(result,indent=2)+'\n'); return result


LAMBDAS={'teacher':1000.0,'reproj':1.0,'reliability':0.1,'anchor':1000.0}


def total_camera_loss(terms):
    return sum(LAMBDAS[name]*terms[name] for name in LAMBDAS)


@torch.no_grad()
def evaluate_adapter(adapter,model,tokens,meta,raw,scale):
    adapter.eval(); result={}; center=int(np.where(meta['frame_id']==json.loads((OUT/'mining.json').read_text())['selected']['frame_id'])[0][0])
    for condition,name in enumerate(CONDITIONS):
        terms,out=camera_loss_terms(adapter,model,tokens,meta,raw,scale,condition)
        baseline_error=frame_reprojection(raw[condition]['pred_xy'].detach().cpu(),meta['gt_xyc'],meta['size'])
        refined_error=frame_reprojection(out['pred_xy'].detach().cpu(),meta['gt_xyc'],meta['size'])
        raw_cam=raw[condition]['pred_cam'].detach().cpu().numpy(); new_cam=out['pred_cam'].detach().cpu().numpy()
        raw_t=raw[condition]['pred_cam_t'].detach().cpu().numpy(); new_t=out['pred_cam_t'].detach().cpu().numpy(); reliability=out['reliability'].squeeze().detach().cpu().numpy()
        neighbours=[center-1,center+1]; stable=np.ones(16,dtype=bool); stable[center]=False
        result[name]={
            'baseline_mean_reprojection':float(np.mean(baseline_error)),
            'refined_mean_reprojection':float(np.mean(refined_error)),
            'baseline_p95_reprojection':float(np.percentile(baseline_error,95)),
            'refined_p95_reprojection':float(np.percentile(refined_error,95)),
            'failure':{
                'frame_id':int(meta['frame_id'][center]),
                'baseline_reprojection':float(baseline_error[center]),
                'refined_reprojection':float(refined_error[center]),
                'raw_pred_cam':raw_cam[center].tolist(),'refined_pred_cam':new_cam[center].tolist(),
                'raw_pred_cam_t':raw_t[center].tolist(),'refined_pred_cam_t':new_t[center].tolist(),
                'reliability':float(reliability[center]),
            },
            'neighbours':[
                {'frame_id':int(meta['frame_id'][i]),'baseline_reprojection':float(baseline_error[i]),'refined_reprojection':float(refined_error[i]),'reliability':float(reliability[i]),'pred_cam_change':float(np.linalg.norm(new_cam[i]-raw_cam[i]))}
                for i in neighbours
            ],
            'baseline_camera_innovation_mean':float(np.mean(centred_innovation(raw_cam))),
            'refined_camera_innovation_mean':float(np.mean(centred_innovation(new_cam))),
            'stable_frame_anchor_deviation':float(np.linalg.norm(new_cam[stable]-raw_cam[stable],axis=-1).mean()),
            'mean_reliability':float(np.mean(reliability)),
            'loss_terms':{k:float(v) for k,v in terms.items()},
        }
    return result


def train(steps=400,lr=1e-3):
    set_seed(); OUT.mkdir(parents=True,exist_ok=True)
    body_ckpt=ROOT/'outputs/dynamesh_v2_phase2_training/best.pt'; body_metrics=ROOT/'outputs/dynamesh_v2_phase2_validation/metrics.json'; camera_code=ROOT/'models/sam_3d_body/sam_3d_body/models/modules/camera_temporal.py'
    frozen={'body_checkpoint':str(body_ckpt),'body_checkpoint_sha256':sha256(body_ckpt),'body_metrics':str(body_metrics),'body_metrics_sha256':sha256(body_metrics),'camera_phase0_sha256':sha256(camera_code)}
    mining=json.loads((OUT/'mining.json').read_text()); gate=json.loads((OUT/'equivalence_gate.json').read_text())
    if not gate['passed'] or mining['split']!='train' or mining['condition']!='clean': raise RuntimeError('pre-training gates are not satisfied')
    model=load_frozen_model(); _,meta,tokens,_,raw,scale=prepare_window(model,mining)
    adapter=CameraFeatureTemporalAdapter().cuda(); optimizer=torch.optim.AdamW(adapter.parameters(),lr=lr,weight_decay=1e-4); assert_camera_optimizer_ownership(optimizer,adapter)
    if any(p.requires_grad for p in model.parameters()): raise AssertionError('frozen SAM model unexpectedly trainable')
    trainable=sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    config={'seed':SEED,'steps':steps,'lr':lr,'weight_decay':1e-4,'optimizer':'AdamW','window_length':16,'sequence':meta['sequence'],'person_id':meta['person'],'frame_ids':meta['frame_id'].astype(int).tolist(),'failure_frame_id':mining['selected']['frame_id'],'conditions':list(CONDITIONS),'lambdas':LAMBDAS,'reliability_scale':scale,'trainable_parameters':trainable,'optimizer_ownership':'CameraFeatureTemporalAdapter only','frozen':frozen,'common_root_loss':False,'checkpoint_selection':'lowest same-window four-condition running total loss'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    initial=evaluate_adapter(adapter,model,tokens,meta,raw,scale); history=[]; best=math.inf; started=time.time()
    for step in range(1,steps+1):
        adapter.train(); condition=(step-1)%4; optimizer.zero_grad(set_to_none=True); terms,_=camera_loss_terms(adapter,model,tokens,meta,raw,scale,condition); total=total_camera_loss(terms)
        if not torch.isfinite(total): raise FloatingPointError(f'non-finite loss at step {step}')
        total.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0); optimizer.step()
        item={'step':step,'condition':CONDITIONS[condition],'total':float(total.detach()),**{k:float(v.detach()) for k,v in terms.items()}}
        history.append(item)
        if step%20==0:
            adapter.eval()
            with torch.no_grad():
                checkpoint_total=float(np.mean([float(total_camera_loss(camera_loss_terms(adapter,model,tokens,meta,raw,scale,c)[0])) for c in range(4)]))
            item['four_condition_total']=checkpoint_total
            if checkpoint_total<best:
                best=checkpoint_total; torch.save({'adapter_state_dict':adapter.state_dict(),'config':config,'best_step':step,'best_total':best},OUT/'best.pt')
            print(json.dumps(item),flush=True)
    torch.save({'adapter_state_dict':adapter.state_dict(),'config':config,'final_step':steps},OUT/'final.pt'); (OUT/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    checkpoint=torch.load(OUT/'best.pt',map_location='cuda',weights_only=False); adapter.load_state_dict(checkpoint['adapter_state_dict']); final=evaluate_adapter(adapter,model,tokens,meta,raw,scale)
    clean=final['clean']; failure=clean['failure']; neighbour_base=np.mean([x['baseline_reprojection'] for x in clean['neighbours']]); neighbour_new=np.mean([x['refined_reprojection'] for x in clean['neighbours']]); neighbour_rel=np.mean([x['reliability'] for x in clean['neighbours']])
    failure_gain=(failure['baseline_reprojection']-failure['refined_reprojection'])/max(failure['baseline_reprojection'],1e-8); window_gain=(clean['baseline_mean_reprojection']-clean['refined_mean_reprojection'])/max(clean['baseline_mean_reprojection'],1e-8)
    aligned=failure['reliability']<neighbour_rel
    preserved=neighbour_new-neighbour_base<0.01 and clean['stable_frame_anchor_deviation']<0.05
    if failure_gain>.10 and window_gain>0 and preserved and aligned: verdict='CAMERA_TEMPORAL_ADAPTER_LEARNABLE'
    elif failure_gain>0 and window_gain>=0 and preserved and not aligned: verdict='RELIABILITY_NOT_ALIGNED'
    elif failure_gain>0: verdict='CAMERA_RECOVERY_WEAK'
    else: verdict='CAMERA_ARCHITECTURE_OR_LOSS_NEEDS_REDESIGN'
    summary={'state':'complete','frozen':frozen,'trainable_parameters':trainable,'steps':steps,'lr':lr,'best_step':checkpoint['best_step'],'best_total':checkpoint['best_total'],'elapsed_seconds':time.time()-started,'initial':initial,'final':final,'failure_relative_improvement':failure_gain,'clean_window_relative_improvement':window_gain,'failure_reliability_below_neighbours':bool(aligned),'normal_neighbours_preserved':bool(preserved),'verdict':verdict}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if sha256(body_ckpt)!=frozen['body_checkpoint_sha256']: raise RuntimeError('Body checkpoint changed during Camera training')
    print(json.dumps({'state':'complete','best_step':checkpoint['best_step'],'failure_gain':failure_gain,'window_gain':window_gain,'verdict':verdict},indent=2)); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('gate','mine','probe','train')); parser.add_argument('--steps',type=int,default=400); args=parser.parse_args(); set_seed()
    result={'gate':equivalence_gate,'mine':mine_all,'probe':probe,'train':lambda:train(args.steps)}[args.mode]();
    if args.mode!='train': print(json.dumps(result,indent=2))


if __name__=='__main__': main()

