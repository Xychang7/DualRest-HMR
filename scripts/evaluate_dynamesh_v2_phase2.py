"""Resumable, scientifically independent Phase-2 geometry validation."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path("/root/sam-body4d-0.2.0"); CACHE=ROOT/"data/dynamesh_v2/cache_v2"; TRAIN_OUT=ROOT/"outputs/dynamesh_v2_phase2_training"; OUT=ROOT/"outputs/dynamesh_v2_phase2_validation"; TRACK_OUT=OUT/"tracks"
DATA=Path("/root/data1/sam-body4d-master/dataset/3DPW"); SMPL=Path("/root/DuoMo-main/data/body_models/smpl/SMPL_NEUTRAL.pkl"); REGRESSOR=Path("/root/DuoMo-main/data/body_models/J_regressor_h36m.npy")
MHR_PY=Path("/root/miniconda3/envs/mhr_clean/bin/python"); WORKER=ROOT/"adapters/mhr_to_smpl_eval_worker.py"; RATIOS=(0.0,0.15,0.35,0.55); LABELS=("clean","occ15","occ35","occ55"); SCHEMA="dynamesh_v2_phase2_track_eval_v1"
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"models/sam_3d_body")); sys.path.insert(0,str(ROOT/"scripts"))
from eval_3dpw_standard import _similarity_align, build_gt_records, regress_h36m_j14
from train_dynamesh_v2_micro_overfit import decode_body_articulation_rotmats


def canonical_hash(payload):
    clean={k:v for k,v in payload.items() if k!="checksum_sha256"}
    return hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n"); os.replace(temp,path)


def validate_track_result(path,sequence,person):
    payload=json.loads(Path(path).read_text())
    if payload.get("schema_version")!=SCHEMA or payload.get("sequence")!=sequence or payload.get("person_id")!=person or payload.get("completion") is not True: raise ValueError("track completion identity/schema mismatch")
    if payload.get("checksum_sha256")!=canonical_hash(payload): raise ValueError("track checksum mismatch")
    if set(payload.get("conditions",{}))!=set(LABELS): raise ValueError("track conditions incomplete")
    frames=int(payload["frames"])
    for label in LABELS:
        if set(payload["conditions"][label])!={"baseline","v2"}: raise ValueError(f"{label} methods incomplete")
        for method in ("baseline","v2"):
            item=payload["conditions"][label][method]
            for field in ("mpjpe_mm","pa_mpjpe_mm","converter_fitting_error_mm","body_teacher_geodesic_deg","baseline_token_l2","token_l2"):
                values=np.asarray(item[field],dtype=float)
                if values.shape!=(frames,) or not np.isfinite(values).all(): raise ValueError(f"{label}/{method}/{field} invalid")
    return payload


def convert_independent(vertices,directory,stem):
    input_path=Path(directory)/f"{stem}_mhr.npy"; output_path=Path(directory)/f"{stem}_smpl.npz"; np.save(input_path,np.asarray(vertices,dtype=np.float32))
    env=os.environ.copy(); root="/root/miniconda3/envs/mhr_clean"; env["LD_LIBRARY_PATH"]=f"{root}/lib/python3.12/site-packages/torch/lib:{root}/lib:"+env.get("LD_LIBRARY_PATH","")
    subprocess.run([str(MHR_PY),str(WORKER),"--input",str(input_path),"--output",str(output_path),"--smpl-model",str(SMPL)],env=env,cwd="/root/MHR-main/tools/mhr_smpl_conversion",check=True)
    with np.load(output_path) as result: smpl=np.asarray(result["vertices"]); errors=np.asarray(result["errors"]).reshape(-1)
    input_path.unlink(); output_path.unlink(); return smpl,errors


def geo_errors(pred_vertices,gt_vertices,regressor):
    pred=regress_h36m_j14(pred_vertices,regressor); gt=regress_h36m_j14(gt_vertices,regressor)
    mp=np.linalg.norm(pred-gt,axis=-1).mean(-1)*1000
    pa=np.asarray([np.linalg.norm(_similarity_align(p,g)-g,axis=-1).mean()*1000 for p,g in zip(pred,gt)])
    return mp,pa


def geodesic_from_cosine_deg(cosine):
    """Reduce joint rotation errors while preserving the frame dimension."""
    if cosine.ndim != 2:
        raise ValueError(f"cosine must be (frames,joints), got {tuple(cosine.shape)}")
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0,1.0))).mean(-1)


def geodesic_deg(cont,teacher):
    a=decode_body_articulation_rotmats(cont); b=decode_body_articulation_rotmats(teacher); relative=a@b.transpose(-1,-2); cosine=(relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    return geodesic_from_cosine_deg(cosine).detach().cpu().numpy()


@torch.no_grad()
def decode_variant_meshes(head,raw,refined,init,chunk=32):
    baseline,rebuilt,raw_cont,refined_cont=[],[],[],[]
    for start in range(0,len(raw),chunk):
        sl=slice(start,min(start+chunk,len(raw))); raw_out=head(raw[sl],init[sl]); refined_out=head(refined[sl],init[sl])
        vertices=head.mhr_forward(global_trans=torch.zeros_like(raw_out["global_rot"]),global_rot=raw_out["global_rot"],body_pose_params=refined_out["body_pose"],hand_pose_params=raw_out["hand"],scale_params=raw_out["scale"],shape_params=raw_out["shape"],expr_params=raw_out["face"],do_pcblend=True,return_keypoints=False,return_joint_coords=False,return_model_params=False,return_joint_rotations=False); vertices[..., [1,2]]*=-1
        baseline.append(raw_out["pred_vertices"].cpu()); rebuilt.append(vertices.cpu()); raw_cont.append(raw_out["pred_pose_raw"][:,6:].cpu()); refined_cont.append(refined_out["pred_pose_raw"][:,6:].cpu())
    return np.asarray(torch.cat(baseline)),np.asarray(torch.cat(rebuilt)),torch.cat(raw_cont),torch.cat(refined_cont)


def method_metrics(smpl,fit,gt,regressor,geo,base_dist,dist):
    mp,pa=geo_errors(smpl,gt,regressor)
    return {"mpjpe_mm":mp.tolist(),"pa_mpjpe_mm":pa.tolist(),"converter_fitting_error_mm":(fit*1000).tolist(),"body_teacher_geodesic_deg":geo.tolist(),"baseline_token_l2":base_dist.tolist(),"token_l2":dist.tolist()}


def aggregate(results,checkpoint,windows,elapsed):
    arrays={method:{label:defaultdict(list) for label in LABELS} for method in ("baseline","v2")}
    per_track=[]
    for result in results:
        compact={"sequence":result["sequence"],"person_id":result["person_id"],"frames":result["frames"],"conditions":{}}
        for label in LABELS:
            compact["conditions"][label]={}
            for method in ("baseline","v2"):
                item=result["conditions"][label][method]
                for field,values in item.items(): arrays[method][label][field].extend(values)
                compact["conditions"][label][method]={"mpjpe_mm":float(np.mean(item["mpjpe_mm"])),"pa_mpjpe_mm":float(np.mean(item["pa_mpjpe_mm"]))}
        per_track.append(compact)
    summary={method:{} for method in ("baseline","v2")}
    for method in summary:
        for label in LABELS:
            b=arrays[method][label]; summary[method][label]={"frames":len(b["mpjpe_mm"]),"mpjpe_mm":float(np.mean(b["mpjpe_mm"])),"pa_mpjpe_mm":float(np.mean(b["pa_mpjpe_mm"])),"body_teacher_geodesic_deg":float(np.mean(b["body_teacher_geodesic_deg"])),"converter_fitting_error_mm":{"mean":float(np.mean(b["converter_fitting_error_mm"])),"std":float(np.std(b["converter_fitting_error_mm"])),"min":float(np.min(b["converter_fitting_error_mm"])),"max":float(np.max(b["converter_fitting_error_mm"]))},"baseline_token_l2":float(np.mean(b["baseline_token_l2"])),"token_l2":float(np.mean(b["token_l2"]))}
    for label in LABELS:
        base=summary["baseline"][label]["baseline_token_l2"]; refined=summary["v2"][label]["token_l2"]; summary["v2"][label]["token_recovery_percent"]=0.0 if base==0 else 100*(base-refined)/base
    for label in ("occ35","occ55"):
        for metric in ("mpjpe_mm","pa_mpjpe_mm"):
            gap=summary["baseline"][label][metric]-summary["baseline"]["clean"][metric]; recovered=summary["baseline"][label][metric]-summary["v2"][label][metric]; summary["v2"][label][f"{metric}_occlusion_gap"]=gap; summary["v2"][label][f"{metric}_recovered"]=recovered; summary["v2"][label][f"{metric}_recovery_ratio"]=None if gap==0 else recovered/gap
    stability={}
    for label in LABELS:
        deltas=np.asarray([x["conditions"][label]["v2"]["pa_mpjpe_mm"]-x["conditions"][label]["baseline"]["pa_mpjpe_mm"] for x in per_track]); stability[label]={"tracks_improved_pa":int((deltas<0).sum()),"total_tracks":len(deltas),"mean_delta_pa_mpjpe_mm":float(deltas.mean()),"median_delta_pa_mpjpe_mm":float(np.median(deltas))}
    clean_delta=summary["v2"]["clean"]["pa_mpjpe_mm"]-summary["baseline"]["clean"]["pa_mpjpe_mm"]; heavy_good=all(summary["v2"][x]["pa_mpjpe_mm"]<summary["baseline"][x]["pa_mpjpe_mm"] and summary["v2"][x]["mpjpe_mm"]<summary["baseline"][x]["mpjpe_mm"] for x in ("occ35","occ55")); feature_good=all(summary["v2"][x]["token_recovery_percent"]>0 for x in ("occ35","occ55")); stable=all(stability[x]["tracks_improved_pa"]>=10 for x in ("occ35","occ55"))
    if clean_delta<=0.5 and heavy_good and feature_good and stable: judgment="GENERALIZATION_CONFIRMED"
    elif clean_delta<=2.0 and any(summary["v2"][x]["pa_mpjpe_mm"]<summary["baseline"][x]["pa_mpjpe_mm"] for x in ("occ35","occ55")): judgment="WEAK_GENERALIZATION"
    elif feature_good and not heavy_good: judgment="MICRO_OVERFIT_ONLY"
    else: judgment="ARCHITECTURE_OR_TRAINING_NEEDS_REDESIGN"
    return {"state":"complete","checkpoint":str(TRAIN_OUT/"best.pt"),"best_epoch":checkpoint["best_epoch"],"validation_windows":len(windows),"completed_tracks":len(results),"summary":summary,"per_track":per_track,"per_track_stability":stability,"clean_pa_degradation_mm":clean_delta,"judgment":judgment,"judgment_rule":{"clean_preservation_pa_mm":0.5,"stable_heavy_tracks_min":10},"elapsed_seconds":elapsed}


def main():
    from omegaconf import OmegaConf
    from sam_3d_body import SAM3DBodyEstimator,load_sam_3d_body
    from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
    OUT.mkdir(parents=True,exist_ok=True); TRACK_OUT.mkdir(parents=True,exist_ok=True); manifest=json.loads((CACHE/"manifest.json").read_text()); windows=manifest["validation_windows_length16_nonoverlap"]
    entries=[x for x in manifest["entries"] if x["split"]=="validation"]; entry_lookup={(x["sequence"],x["person_id"]):x for x in entries}; grouped=defaultdict(list)
    for window in windows: grouped[(window["sequence"],window["person_id"])].append(window)
    if len(grouped)!=16: raise RuntimeError(f"expected 16 validation tracks, got {len(grouped)}")
    completed=[]; pending=[]
    for key in sorted(grouped):
        path=TRACK_OUT/f"{key[0]}_person{key[1]}.json"
        if path.exists(): completed.append(validate_track_result(path,*key))
        else: pending.append(key)
    started=time.time(); status={"state":"running","completed_tracks":len(completed),"total_tracks":16,"current_track":pending[0] if pending else None,"elapsed_seconds":0.0,"last_track_seconds":None,"estimated_progress":len(completed)/16,"failed_tracks":[]}; atomic_json(OUT/"status.json",status)
    if pending:
        device=torch.device("cuda"); cfg=OmegaConf.load(ROOT/"configs/body4d.yaml"); model,model_cfg=load_sam_3d_body(cfg.sam_3d_body.ckpt_path,device=device,mhr_path=cfg.sam_3d_body.mhr_path); estimator=SAM3DBodyEstimator(model,model_cfg,human_detector=None,human_segmentor=None,fov_estimator=None)
        for p in estimator.model.parameters(): p.requires_grad_(False)
        estimator.model.eval(); head=estimator.model.head_pose; adapter=BodyFeatureTemporalAdapter().to(device); checkpoint=torch.load(TRAIN_OUT/"best.pt",map_location=device,weights_only=False); adapter.load_state_dict(checkpoint["adapter_state_dict"],strict=True); adapter.eval(); regressor=np.load(REGRESSOR)
        with tempfile.TemporaryDirectory(prefix="dynav2_phase2_eval_") as directory:
          for track_number,key in enumerate(pending,1):
            track_started=time.time(); sequence,person=key; status.update({"current_track":{"sequence":sequence,"person_id":person},"elapsed_seconds":time.time()-started}); atomic_json(OUT/"status.json",status)
            try:
                entry=entry_lookup[key]
                with np.load(entry["path"],allow_pickle=False) as cache: tokens=np.asarray(cache["pose_token"]); init=np.asarray(cache["mhr_init_pose"]); teacher=np.asarray(cache["body_cont"][0]); frame_ids=np.asarray(cache["frame_id"])
                track_windows=grouped[key]; indices=np.concatenate([np.arange(x["cache_start"],x["cache_stop"]) for x in track_windows]); assert len(indices)==len(np.unique(indices)); keys=[(sequence,person,int(frame_ids[i])) for i in indices]
                gt_records=build_gt_records(keys,gt_dir=DATA/"sequenceFiles/validation",smpl_model_path=SMPL,device="cpu"); gt_lookup={(x["sequence"],x["person_id"],x["frame_id"]):x["smpl_vertices"] for x in gt_records}; gt=np.stack([gt_lookup[x] for x in keys]); teacher_t=torch.as_tensor(teacher[indices])
                result={"schema_version":SCHEMA,"sequence":sequence,"person_id":person,"frames":len(indices),"windows":len(track_windows),"frame_ids":[x[2] for x in keys],"conditions":{},"completion":False}
                for condition,label in enumerate(LABELS):
                    raw_windows=[]; init_windows=[]
                    for window in track_windows:
                        sl=slice(window["cache_start"],window["cache_stop"]); raw_windows.append(tokens[condition,sl]); init_windows.append(init[condition,sl])
                    raw=torch.as_tensor(np.stack(raw_windows),device=device,dtype=torch.float32); init_t=torch.as_tensor(np.stack(init_windows),device=device)
                    with torch.no_grad(): refined=adapter(raw,torch.full(raw.shape[:2],RATIOS[condition],device=device)); flat_raw=raw.reshape(-1,1024); flat_refined=refined.reshape(-1,1024); flat_init=init_t.reshape(-1,519); clean=torch.as_tensor(tokens[0,indices],device=device,dtype=torch.float32); base_dist=torch.linalg.vector_norm(flat_raw-clean,dim=-1).cpu().numpy(); refined_dist=torch.linalg.vector_norm(flat_refined-clean,dim=-1).cpu().numpy()
                    baseline_mhr,v2_mhr,raw_cont,refined_cont=decode_variant_meshes(head,flat_raw,flat_refined,flat_init); baseline_geo=geodesic_deg(raw_cont,teacher_t); v2_geo=geodesic_deg(refined_cont,teacher_t)
                    # Strictly independent official fits: no method or condition shares identity fitting.
                    baseline_smpl,baseline_fit=convert_independent(baseline_mhr,directory,f"{sequence}_{person}_{label}_baseline"); v2_smpl,v2_fit=convert_independent(v2_mhr,directory,f"{sequence}_{person}_{label}_v2")
                    result["conditions"][label]={"baseline":method_metrics(baseline_smpl,baseline_fit,gt,regressor,baseline_geo,base_dist,base_dist),"v2":method_metrics(v2_smpl,v2_fit,gt,regressor,v2_geo,base_dist,refined_dist)}
                    del baseline_mhr,v2_mhr,baseline_smpl,v2_smpl; torch.cuda.empty_cache()
                result["elapsed_seconds"]=time.time()-track_started; result["completion"]=True; result["checksum_sha256"]=canonical_hash(result); track_path=TRACK_OUT/f"{sequence}_person{person}.json"; atomic_json(track_path,result); validate_track_result(track_path,sequence,person); completed.append(result)
                status.update({"completed_tracks":len(completed),"current_track":None,"elapsed_seconds":time.time()-started,"last_track_seconds":result["elapsed_seconds"],"estimated_progress":len(completed)/16}); atomic_json(OUT/"status.json",status)
            except Exception as exc:
                status["state"]="failed"; status["failed_tracks"].append({"sequence":sequence,"person_id":person,"error":repr(exc)}); status["elapsed_seconds"]=time.time()-started; atomic_json(OUT/"status.json",status); raise
    else:
        checkpoint=torch.load(TRAIN_OUT/"best.pt",map_location="cpu",weights_only=False)
    metrics=aggregate(completed,checkpoint,windows,time.time()-started); atomic_json(OUT/"metrics.json",metrics); atomic_json(OUT/"status.json",{"state":"complete","completed_tracks":16,"total_tracks":16,"current_track":None,"elapsed_seconds":metrics["elapsed_seconds"],"last_track_seconds":status.get("last_track_seconds"),"estimated_progress":1.0,"failed_tracks":[]}); print(json.dumps({"state":"complete","judgment":metrics["judgment"]}),flush=True)


if __name__=="__main__": main()
