"""Phase-1.8 strict multi-joint translation-only camera supervision Gate.

Uses cache-v2, frozen SAM heads, and native 3DPW/SMPL joints.  It does not run
the visual backbone, Meta conversion, training, or validation/test data.
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path('/root/sam-body4d-0.2.0')
CACHE = ROOT / 'data/dynamesh_v2/cache_v2/train'
OUT = ROOT / 'outputs/dynamesh_v2_camera_phase18_multijoint_gt'
SMPL_MODEL = Path('/root/DuoMo-main/data/body_models/smpl/SMPL_NEUTRAL.pkl')
CONDITIONS = ('clean', 'occ15', 'occ35', 'occ55')
MHR_NAMES = ('left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')
MHR_INDICES = (5,6,7,8,62,41,9,10,11,12,13,14)
SMPL_INDICES = (16,17,18,19,20,21,1,2,4,5,7,8)
OP_INDICES = (5,2,6,3,7,4,11,8,12,9,13,10)
JOINT_MAPPING = tuple({
    'name': name, 'mhr_source': 'MHR70 pred_keypoints_3d', 'mhr70_index': mi,
    'smpl_source': 'native SMPL joint', 'smpl_native_index': si,
    'confidence_source': '3DPW OpenPose18 confidence only', 'openpose18_index': oi,
} for name, mi, si, oi in zip(MHR_NAMES, MHR_INDICES, SMPL_INDICES, OP_INDICES))


def validate_joint_mapping():
    names = [x['name'] for x in JOINT_MAPPING]
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError('mapping must contain 12 unique joints')
    if {'pelvis','neck','head','hip_midpoint'}.intersection(names):
        raise ValueError('approximate/root joints are forbidden')
    if tuple(x['mhr70_index'] for x in JOINT_MAPPING) != MHR_INDICES:
        raise ValueError('MHR70 mapping changed')
    if tuple(x['smpl_native_index'] for x in JOINT_MAPPING) != SMPL_INDICES:
        raise ValueError('native SMPL mapping changed')


def translation_only_fit(x, y, weights):
    x, y, weights = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(weights, np.float64)
    if x.ndim != 2 or x.shape[1:] != (3,) or y.shape != x.shape or weights.shape != (len(x),):
        raise ValueError(f'shape mismatch: x={x.shape}, y={y.shape}, weights={weights.shape}')
    valid = np.isfinite(x).all(1) & np.isfinite(y).all(1) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError('no valid joint remains')
    w = weights[valid]; delta = y[valid] - x[valid]
    translation = np.sum(delta * w[:, None], axis=0) / np.sum(w)
    residual = np.linalg.norm(x[valid] + translation - y[valid], axis=1)
    # Robust diagnostic only: coordinate-wise weighted median, never used as target.
    robust = []
    for axis in range(3):
        order = np.argsort(delta[:, axis]); cumulative = np.cumsum(w[order])
        robust.append(delta[order[np.searchsorted(cumulative, .5 * cumulative[-1])], axis])
    robust = np.asarray(robust)
    mad = np.median(np.linalg.norm(delta - np.median(delta, axis=0), axis=1))
    return {'translation': translation, 'valid_joint_count': int(valid.sum()),
            'residuals': residual, 'mean_residual': float(residual.mean()),
            'median_residual': float(np.median(residual)), 'max_residual': float(residual.max()),
            'robust_translation': robust, 'mean_robust_disagreement': float(np.linalg.norm(translation-robust)),
            'translation_delta_mad': float(mad)}


def fit_joint_subsets(x, y, weights):
    all_fit = translation_only_fit(x, y, weights)
    def optional_fit(xs,ys,ws):
        try: return translation_only_fit(xs,ys,ws)
        except ValueError as error:
            if 'no valid joint' not in str(error): raise
            return {'translation':np.full(3,np.nan),'valid_joint_count':0,'residuals':np.array([]),
                    'mean_residual':np.nan,'median_residual':np.nan,'max_residual':np.nan,
                    'robust_translation':np.full(3,np.nan),'mean_robust_disagreement':np.nan,'translation_delta_mad':np.nan}
    upper = optional_fit(np.asarray(x)[:6], np.asarray(y)[:6], np.asarray(weights)[:6])
    lower = optional_fit(np.asarray(x)[6:], np.asarray(y)[6:], np.asarray(weights)[6:])
    return {'all': all_fit, 'upper': upper, 'lower': lower,
            'upper_lower_disagreement': float(np.linalg.norm(upper['translation']-lower['translation'])),
            'all_upper_disagreement': float(np.linalg.norm(all_fit['translation']-upper['translation'])),
            'all_lower_disagreement': float(np.linalg.norm(all_fit['translation']-lower['translation']))}


def stats(values):
    value = np.asarray(values, np.float64); value = value[np.isfinite(value)]
    if not len(value): return {'count': 0, 'mean': None, 'median': None, 'std': None, 'p95': None, 'max': None}
    return {'count': int(len(value)), 'mean': float(value.mean()), 'median': float(np.median(value)),
            'std': float(value.std()), 'p95': float(np.percentile(value,95)), 'max': float(value.max())}


def corr(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float); keep=np.isfinite(a)&np.isfinite(b)
    if keep.sum()<3 or np.std(a[keep])==0 or np.std(b[keep])==0: return None
    return float(np.corrcoef(a[keep],b[keep])[0,1])


def rank_corr(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); keep=np.isfinite(a)&np.isfinite(b); a,b=a[keep],b[keep]
    if len(a)<3: return None
    def rank(x):
        order=np.argsort(x,kind='mergesort'); out=np.empty(len(x),float); out[order]=np.arange(len(x)); return out
    return corr(rank(a),rank(b))


def choose_verdict(clean, all_fit, failure, heavy_body_corr):
    low_fraction=float(np.mean(np.asarray(all_fit)<=.10))
    isolated=bool(failure[522]['camera_error']>1.5*max(failure[520]['camera_error'],failure[524]['camera_error']))
    broad=bool(failure[522]['camera_error']>2*failure[524]['camera_error'] and
               np.linalg.norm(np.array([failure[522][f'target_t{a}'] for a in 'xyz'])-
                              np.array([failure[524][f'target_t{a}'] for a in 'xyz']))<.15)
    if clean['fit_residual']['p95']<=.05 and clean['upper_lower_disagreement']['p95']<=.05 and isolated and abs(clean['correlations']['fit_vs_pose_magnitude'] or 0)<.3:
        verdict='MULTI_JOINT_CAMERA_GT_VALIDATED'
    elif low_fraction>=.90 and clean['upper_lower_disagreement']['median']<=.06 and broad and (heavy_body_corr or 0)>=.5:
        verdict='MULTI_JOINT_CAMERA_GT_USABLE_WITH_RELIABILITY'
    else: verdict='MULTI_JOINT_CAMERA_GT_NOT_RELIABLE'
    return verdict, isolated, broad, low_fraction


def body_teacher_error(body_cont, device):
    import torch
    from train_dynamesh_v2_micro_overfit import decode_body_articulation_rotmats
    rotations=[]
    with torch.no_grad():
        for condition in range(4):
            rotations.append(decode_body_articulation_rotmats(torch.as_tensor(body_cont[condition],device=device,dtype=torch.float32)))
        clean=rotations[0]
        errors=[]
        for condition, rotation in enumerate(rotations):
            if condition==0: errors.append(np.zeros(len(rotation),np.float32)); continue
            relative=rotation @ clean.transpose(-1,-2)
            cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
            errors.append(torch.rad2deg(torch.acos(cosine)).mean(-1).cpu().numpy())
    return np.stack(errors)


def gt_camera_joints(track, smpl_model, device):
    import torch
    pose=torch.as_tensor(track['gt_pose'],device=device,dtype=torch.float32)
    with torch.no_grad():
        output=smpl_model(betas=torch.as_tensor(track['gt_betas'],device=device,dtype=torch.float32),
            global_orient=pose[:,:3],body_pose=pose[:,3:72],
            transl=torch.as_tensor(track['gt_transl'],device=device,dtype=torch.float32))
        world=output.joints[:,list(SMPL_INDICES)].detach().cpu().numpy()
    camera=np.asarray(track['camera_extrinsics'],np.float64)
    return np.einsum('nij,nkj->nki',camera[:,:3,:3],world)+camera[:,None,:3,3]


def process_track(path, model, smpl_model, device):
    from train_dynamesh_v2_camera_phase1 import load_track, metadata_for_track, decode_condition, frame_reprojection
    track=load_track(path); meta=metadata_for_track(track)
    if str(track['split'].item())!='train': raise ValueError('test/validation leakage')
    gt3d=gt_camera_joints(track,smpl_model,device)
    weights=np.clip(np.asarray(meta['gt_xyc'][...,2],np.float64),0,None)
    teacher=body_teacher_error(track['body_cont'],device)
    body=np.asarray(track['body_cont'],np.float64)
    magnitude=np.linalg.norm(body,axis=-1)/math.sqrt(body.shape[-1]); change=np.full(magnitude.shape,np.nan)
    contiguous=np.diff(np.asarray(track['local_frame_index']))==1
    change[:,1:]=np.linalg.norm(body[:,1:]-body[:,:-1],axis=-1)/math.sqrt(body.shape[-1]); change[:,1:][:,~contiguous]=np.nan
    rows=[]
    with __import__('torch').no_grad():
        decoded=[decode_condition(model,track['pose_token'][c],track['mhr_init_pose'][c],meta) for c in range(4)]
    for c,label in enumerate(CONDITIONS):
        points=decoded[c]['points3d'].detach().cpu().numpy(); raw_t=decoded[c]['pred_cam_t'].detach().cpu().numpy()
        reproj=frame_reprojection(decoded[c]['pred_xy'].detach().cpu().numpy(),meta['gt_xyc'],meta['size'])
        for i,fid in enumerate(meta['frame_id']):
            fit=fit_joint_subsets(points[i],gt3d[i],weights[i]); target=fit['all']['translation']; error=raw_t[i]-target
            rows.append({'sequence':meta['sequence'],'person_id':int(meta['person']),'frame_id':int(fid),'condition':label,
                'valid_joint_count':fit['all']['valid_joint_count'],
                'target_tx':target[0],'target_ty':target[1],'target_tz':target[2],
                'raw_tx':raw_t[i,0],'raw_ty':raw_t[i,1],'raw_tz':raw_t[i,2],
                'camera_error_x':error[0],'camera_error_y':error[1],'camera_error_z':error[2],
                'camera_error':np.linalg.norm(error),'z_abs_error':abs(error[2]),
                'z_ratio':raw_t[i,2]/target[2] if abs(target[2])>1e-8 else np.nan,
                'fit_residual':fit['all']['mean_residual'],'upper_residual':fit['upper']['mean_residual'],
                'lower_residual':fit['lower']['mean_residual'],
                'upper_lower_disagreement':fit['upper_lower_disagreement'],
                'all_upper_disagreement':fit['all_upper_disagreement'],'all_lower_disagreement':fit['all_lower_disagreement'],
                'robust_target_disagreement':fit['all']['mean_robust_disagreement'],
                'translation_delta_mad':fit['all']['translation_delta_mad'],
                'reprojection_error':reproj[i], 'body_teacher_geodesic_deg':teacher[c,i],
                'body_cont_magnitude':magnitude[c,i],'body_cont_change':change[c,i],
                'bbox_size':meta['size'][i], 'visibility':int(track['visibility'][i])})
    return rows


def summarize(rows, elapsed):
    by={label:[r for r in rows if r['condition']==label] for label in CONDITIONS}
    metrics=('fit_residual','upper_residual','lower_residual','upper_lower_disagreement','all_upper_disagreement',
             'all_lower_disagreement','camera_error','z_abs_error','target_tz','raw_tz','robust_target_disagreement')
    conditions={label:{name:stats([r[name] for r in subset]) for name in metrics} for label,subset in by.items()}
    for label,subset in by.items():
        conditions[label]['camera_error_signed_xyz']={axis:{'mean':float(np.mean([r[f'camera_error_{axis}'] for r in subset])),'std':float(np.std([r[f'camera_error_{axis}'] for r in subset]))} for axis in 'xyz'}
        conditions[label]['z_ratio']=stats([r['z_ratio'] for r in subset])
        conditions[label]['correlations']={
            'fit_vs_body_teacher_pearson':corr([r['fit_residual'] for r in subset],[r['body_teacher_geodesic_deg'] for r in subset]),
            'fit_vs_body_teacher_spearman':rank_corr([r['fit_residual'] for r in subset],[r['body_teacher_geodesic_deg'] for r in subset]),
            'fit_vs_pose_magnitude':corr([r['fit_residual'] for r in subset],[r['body_cont_magnitude'] for r in subset]),
            'fit_vs_pose_change':corr([r['fit_residual'] for r in subset],[r['body_cont_change'] for r in subset]),
            'fit_vs_bbox_size':corr([r['fit_residual'] for r in subset],[r['bbox_size'] for r in subset]),
            'camera_error_vs_reprojection':corr([r['camera_error'] for r in subset],[r['reprojection_error'] for r in subset]),
        }
    clean=by['clean']; failure={int(r['frame_id']):r for r in clean if r['sequence']=='courtyard_basketball_00' and r['person_id']==1 and r['frame_id'] in (520,522,524)}
    if set(failure)!={520,522,524}: raise ValueError(f'fixed failure records missing: {sorted(failure)}')
    all_fit=np.asarray([r['fit_residual'] for r in rows]); all_cam=np.asarray([r['camera_error'] for r in rows]); all_rep=np.asarray([r['reprojection_error'] for r in rows]); all_body=np.asarray([r['body_teacher_geodesic_deg'] for r in rows])
    fit_low,fit_high=np.percentile(all_fit,[25,75]); cam_high=np.percentile(all_cam,75); rep_high=np.percentile(all_rep,75); body_low,body_high=np.percentile(all_body,[25,75])
    case_a=[r for r in rows if r['camera_error']>=cam_high and r['fit_residual']<=fit_low and r['reprojection_error']>=rep_high and r['body_teacher_geodesic_deg']<=body_low]
    case_b=[r for r in rows if r['reprojection_error']>=rep_high and r['fit_residual']>=fit_high and r['body_teacher_geodesic_deg']>=body_high]
    c=conditions['clean']; verdict,failure_detected,broad_failure,low_population=choose_verdict(c,all_fit,failure,conditions['occ55']['correlations']['fit_vs_body_teacher_pearson'])
    return {'gate':'DynaMesh4D-v2 Camera Phase-1.8','verdict':verdict,'elapsed_seconds':elapsed,
        'scale_and_coordinates':{'MHR':'local camera-axis contract, metres','GT':'native SMPL world joints transformed by 3DPW cam_poses, metres','fitted_transform':'translation only; no rotation/scale/offset'},
        'data':{'split':'train','tracks':len(set((r['sequence'],r['person_id']) for r in rows)),'unique_frames':len(set((r['sequence'],r['person_id'],r['frame_id']) for r in rows)),'condition_frame_records':len(rows)},
        'conditions':conditions,'failure_frames':{str(k):{key:(float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for key,v in row.items() if key not in ('sequence','condition')} for k,row in failure.items()},
        'failure_detected':{'isolated_522_vs_both_neighbors':failure_detected,'broad_520_522_failure_vs_524':broad_failure},'joint_diagnosis':{'case_a_count':len(case_a),'case_b_count':len(case_b),'thresholds':{'camera_error_p75':float(cam_high),'fit_p25':float(fit_low),'fit_p75':float(fit_high),'reprojection_p75':float(rep_high),'body_teacher_p25':float(body_low),'body_teacher_p75':float(body_high)}},
        'reliability':{'low_fit_residual_fraction_le_100mm':low_population,'candidate':'w_target=exp(-fit_residual/tau), tau calibrated only from train residual distribution; stop-gradient target','gt_body_rotation_error':'not available without invalid MHR-vs-SMPL articulation matching'},
        'future_loss':None if verdict.endswith('NOT_RELIABLE') else 'L_cam3d = w_target * SmoothL1(pred_cam_t, stopgrad(t_all)); combine later with L_reproj + L_reliability + L_anchor'}


def atomic_text(path, text):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(text); tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--max-tracks',type=int); parser.add_argument('--track-name'); parser.add_argument('--summarize-existing',action='store_true'); parser.add_argument('--output-dir',type=Path,default=OUT); args=parser.parse_args()
    validate_joint_mapping(); start=time.time(); sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'models/sam_3d_body')]
    if args.summarize_existing:
        with (args.output_dir/'per_frame.csv').open() as f:
            rows=list(csv.DictReader(f))
        integer={'person_id','frame_id','valid_joint_count','visibility'}
        for row in rows:
            for key in integer: row[key]=int(row[key])
            for key in row.keys()-integer-{'sequence','condition'}: row[key]=float(row[key]) if row[key] else np.nan
        summary=summarize(rows,time.time()-start); atomic_text(args.output_dir/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n'); print(json.dumps({'resummarized':len(rows),'verdict':summary['verdict']})); return
    import torch, smplx
    from train_dynamesh_v2_camera_phase1 import load_frozen_model
    paths=sorted(CACHE.glob('*.npz'))
    if args.track_name: paths=[path for path in paths if path.stem==args.track_name]
    paths=paths[:args.max_tracks] if args.max_tracks else paths
    if not paths: raise FileNotFoundError(CACHE)
    model=load_frozen_model(); device=torch.device('cuda'); smpl_model=smplx.SMPL(model_path=str(SMPL_MODEL),batch_size=1).to(device).eval()
    rows=[]
    for number,path in enumerate(paths,1):
        rows.extend(process_track(path,model,smpl_model,device)); print(json.dumps({'track':number,'total':len(paths),'path':path.name,'records':len(rows)}),flush=True)
    summary=summarize(rows,time.time()-start); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    mapping={'joints':JOINT_MAPPING,'upper_indices':list(range(6)),'lower_indices':list(range(6,12)),'unit':'metres','GT_camera_transform':'R @ native_SMPL_joint_world + cam_pose_translation','confidence':'3DPW OpenPose18 confidence used only as weight/mask'}
    atomic_text(out/'joint_mapping.json',json.dumps(mapping,indent=2)+'\n')
    fields=list(rows[0]); tmp=out/'per_frame.csv.tmp'
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    tmp.replace(out/'per_frame.csv'); atomic_text(out/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),'verdict':summary['verdict'],'records':len(rows),'elapsed_seconds':summary['elapsed_seconds']}))


if __name__=='__main__': main()
