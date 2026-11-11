"""Protocol B: resumable full-frame, full-metric 3DPW evaluation."""
from __future__ import annotations

import argparse, json, os, random, shutil, subprocess, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT=Path("/root/sam-body4d-0.2.0"); OLD=ROOT/"outputs/dynamesh_v2_final_test"
OUT=ROOT/"outputs/dynamesh_v2_unified_3dpw_eval"; UNITS=OUT/"tracks"; REPS=OUT/"representations"; METRICS=OUT/"metrics"
DATA=Path("/root/data1/sam-body4d-master/dataset/3DPW"); CACHE=OLD/"cache"
VAL_CACHE=ROOT/"data/dynamesh_v2/cache_v2"; BODY=ROOT/"outputs/dynamesh_v2_phase2_training/best.pt"
FRAME=ROOT/"outputs/dynamesh_v2_camera_phase2/framewise/best.pt"; TEMP=ROOT/"outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt"
SMPL=Path("/root/DuoMo-main/data/body_models/smpl/SMPL_NEUTRAL.pkl"); REG=Path("/root/DuoMo-main/data/body_models/J_regressor_h36m.npy")
WORKER=ROOT/"adapters/mhr_to_smpl_persistent_worker.py"; MHR_PY=Path("/root/miniconda3/envs/mhr_clean/bin/python")
LABELS=("clean","occ15","occ35","occ55"); RATIOS=(0.,.15,.35,.55); SCHEMA="dynav2-unified-unit-v1"; SEED=20260908
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"models/sam_3d_body")]
from dynamesh_v2_unified_eval_core import atomic_json, atomic_npz, build_temporal_coverage, canonical_hash, compute_geometry_metrics, acceleration_error, file_sha256
from eval_3dpw_standard import build_gt_records


def load_pkl(path):
 import pickle
 with open(path,"rb") as f:
  try:return pickle.load(f)
  except UnicodeDecodeError:f.seek(0);return pickle.load(f,encoding="latin1")


def npy_atomic(path,array):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+".tmp")
 with tmp.open("wb") as f:np.save(f,np.asarray(array));f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)


def hashes():
 return {"body":file_sha256(BODY),"camera_framewise":file_sha256(FRAME),"camera_temporal":file_sha256(TEMP),"evaluator":file_sha256(Path(__file__)),"core":file_sha256(ROOT/"scripts/dynamesh_v2_unified_eval_core.py"),"converter":file_sha256(WORKER),"h36m_regressor":file_sha256(REG),"smpl_model":file_sha256(SMPL)}


def create_manifest():
 entries=[]
 gt_hashes={}
 for path in sorted(CACHE.glob("*.npz")):
  with np.load(path,allow_pickle=False) as z:
   split=str(z["split"].item());sequence=str(z["sequence"].item());person=int(z["person_id"].item());frames=len(z["frame_id"])
   if split!="test":raise ValueError(f"non-test cache in Protocol B: {path}")
   if len(np.unique(z["frame_id"]))!=frames or np.any(np.diff(z["frame_id"])<=0):raise ValueError(f"bad frame IDs {path}")
  gt_path=DATA/"sequenceFiles/test"/f"{sequence}.pkl"
  if not gt_path.exists():raise FileNotFoundError(f"missing GT annotation: {gt_path}")
  gt_hashes.setdefault(str(gt_path),file_sha256(gt_path))
  entries.append({"sequence":sequence,"person_id":person,"frames":frames,"cache_path":str(path),"cache_sha256":file_sha256(path),"gt_path":str(gt_path),"gt_sha256":gt_hashes[str(gt_path)]})
 manifest={"schema":"dynav2-unified-test-manifest-v1","split":"test","sequences":len(set(x["sequence"] for x in entries)),"person_tracks":len(entries),"valid_person_frames":sum(x["frames"] for x in entries),"total_units":len(entries)*len(LABELS),"entries":entries}
 manifest["checksum_sha256"]=canonical_hash(manifest);atomic_json(OUT/"test_manifest.json",manifest);return manifest


def protocol(manifest,h):
 p={"schema":"dynav2-unified-protocol-v1","name":"Protocol B — Unified Full-Frame 3DPW Evaluation","seed":SEED,"frame_coverage":"all campose_valid person frames","temporal_inference":{"window_length":16,"stride":8,"aggregation":"prediction from window with maximum distance to either boundary; ties choose earlier window","edge_handling":"include final T=16 window; edge-replicate only for contiguous segments shorter than 16","continuity":"local_frame_index increments by exactly one"},"input":"GT poses2d-derived identity-strict person bbox","conditions":list(LABELS),"mpjpe":"pelvis-centered H36M J14, translation only, mm","pa_mpjpe":"per-frame similarity Procrustes on same J14, mm","mpvpe":"SMPL6890 after subtracting each mesh H36M pelvis, no rotation/scale, mm","acceleration":{"formula":"1000*mean_j ||(Jp[t-1]-2Jp[t]+Jp[t+1])-(Jg[t-1]-2Jg[t]+Jg[t+1])||","joints":"pelvis-centered H36M J14","unit":"mm/processed-frame^2","processed_fps":30,"fps_squared_applied":False,"triplets":"only consecutive local_frame_index within same person track"},"aggregation":"frame/triplet weighted overall; person-track paired bootstrap","bootstrap":{"unit":"person_track","resamples":10000,"seed":20260908},"model_hashes":h,"manifest_hash":manifest["checksum_sha256"],"frozen_statement":"DynaMesh4D-v2 model checkpoints were frozen before the first 3DPW Test evaluation.","selection_statement":"This unified evaluation protocol is defined for protocol alignment and metric completeness; no model parameter is selected using the resulting Test metrics."}
 p["checksum_sha256"]=canonical_hash(p);atomic_json(OUT/"protocol.json",p);return p


def write_lock(protocol,h):
 lock={"schema":"dynav2-unified-lock-v1","created_at":time.time(),"protocol_hash":protocol["checksum_sha256"],"model_and_code_hashes":h,"statement":protocol["frozen_statement"],"selection_statement":protocol["selection_statement"]}
 lock["checksum_sha256"]=canonical_hash(lock);atomic_json(OUT/"UNIFIED_EVAL_LOCK.json",lock);return lock


def load_models():
 from omegaconf import OmegaConf
 from sam_3d_body import SAM3DBodyEstimator,load_sam_3d_body
 from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
 cfg=OmegaConf.load(ROOT/"configs/body4d.yaml");model,mcfg=load_sam_3d_body(cfg.sam_3d_body.ckpt_path,device="cuda",mhr_path=cfg.sam_3d_body.mhr_path)
 est=SAM3DBodyEstimator(model,mcfg,human_detector=None,human_segmentor=None,fov_estimator=None);est.model.eval().requires_grad_(False)
 adapter=BodyFeatureTemporalAdapter().cuda();ckpt=torch.load(BODY,map_location="cuda",weights_only=False);adapter.load_state_dict(ckpt["adapter_state_dict"],strict=True);adapter.eval().requires_grad_(False)
 return est,adapter


def build_condition_tensor(reference, requested_alpha):
 alpha=float(requested_alpha)
 if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
  raise ValueError("requested alpha must be finite and in [0,1]")
 condition=torch.full(reference.shape[:2],alpha,device=reference.device,dtype=reference.dtype)
 audit={"requested_alpha":alpha,"actual_min":float(condition.min().item()),"actual_max":float(condition.max().item()),"actual_mean":float(condition.mean().item()),"shape":list(condition.shape),"dtype":str(condition.dtype),"device":str(condition.device)}
 tolerance=max(1e-7,abs(alpha)*1e-6)
 if any(abs(audit[key]-alpha)>tolerance for key in ("actual_min","actual_max","actual_mean")):
  raise RuntimeError(f"runtime alpha audit failed: {audit}")
 return condition,audit


@torch.no_grad()
def refine_all(adapter,tokens,ratio,local_indices,batch=16,condition_audit=None):
 coverage=build_temporal_coverage(local_indices,16,8);windows=coverage["windows"];result=np.empty_like(tokens,dtype=np.float32)
 audits=[]
 for start in range(0,len(windows),batch):
  group=windows[start:start+batch];x=torch.as_tensor(np.stack([tokens[w] for w in group]),device="cuda",dtype=torch.float32);r,audit=build_condition_tensor(x,ratio);audits.append(audit)
  y=adapter(x,r).cpu().numpy()
  for offset,_ in enumerate(group):
   wi=start+offset
   for index,(owner,pos) in coverage["owner"].items():
    if owner==wi:result[index]=y[offset,pos]
 if not np.isfinite(result).all():raise FloatingPointError("non-finite refined token")
 if condition_audit is not None:
  count=sum(int(np.prod(row["shape"])) for row in audits)
  condition_audit.update({"requested_alpha":float(ratio),"actual_min":min(row["actual_min"] for row in audits),"actual_max":max(row["actual_max"] for row in audits),"actual_mean":sum(row["actual_mean"]*int(np.prod(row["shape"])) for row in audits)/count,"shape":[row["shape"] for row in audits],"dtype":audits[0]["dtype"],"device":audits[0]["device"],"batch_count":len(audits),"element_count":count})
 return result,coverage


@torch.no_grad()
def decode_mhr(head,raw,refined,init,chunk=32):
 baseline=[];v2=[]
 for start in range(0,len(raw),chunk):
  sl=slice(start,min(start+chunk,len(raw)));r=torch.as_tensor(raw[sl],device="cuda",dtype=torch.float32);f=torch.as_tensor(refined[sl],device="cuda",dtype=torch.float32);i=torch.as_tensor(init[sl],device="cuda",dtype=torch.float32)
  ro=head(r,i);fo=head(f,i);verts=head.mhr_forward(global_trans=torch.zeros_like(ro["global_rot"]),global_rot=ro["global_rot"],body_pose_params=fo["body_pose"],hand_pose_params=ro["hand"],scale_params=ro["scale"],shape_params=ro["shape"],expr_params=ro["face"],do_pcblend=True,return_keypoints=False,return_joint_coords=False,return_model_params=False,return_joint_rotations=False);verts[..., [1,2]]*=-1
  baseline.append(ro["pred_vertices"].cpu().numpy());v2.append(verts.cpu().numpy())
 return np.concatenate(baseline).astype(np.float32),np.concatenate(v2).astype(np.float32)


def run_converter(mhr_path,out_path):
 env=os.environ.copy();root="/root/miniconda3/envs/mhr_clean";env["LD_LIBRARY_PATH"]=f"{root}/lib/python3.12/site-packages/torch/lib:{root}/lib:"+env.get("LD_LIBRARY_PATH","")
 subprocess.run([str(MHR_PY),str(WORKER),"--input",str(mhr_path),"--output",str(out_path),"--smpl-model",str(SMPL)],cwd="/root/MHR-main/tools/mhr_smpl_conversion",env=env,check=True)


def unit_paths(sequence,person,label,root=OUT):
 stem=f"{sequence}_person{person}_{label}";work=root/"tracks"/stem;rep=root/"representations"/stem
 return {"state":work/"state.json","raw_mhr":work/"raw_mhr.npy","v2_mhr":work/"v2_mhr.npy","raw_conv":work/"raw_converter.npz","v2_conv":work/"v2_converter.npz","raw_rep":rep/"raw.npz","v2_rep":rep/"v2.npz","metrics":root/"metrics"/f"{stem}.json"}


def state_base(entry,label,protocol_hash,h):
 return {"schema":SCHEMA,"sequence":entry["sequence"],"person_id":entry["person_id"],"condition":label,"input_cache_hash":entry["cache_sha256"],"protocol_hash":protocol_hash,"model_hashes":h,"stages":{},"completion":False}


def save_state(path,state):state["checksum_sha256"]=canonical_hash(state);atomic_json(path,state)


def validate_partial(state,entry,label,protocol_hash,h):
 if state.get("schema")!=SCHEMA or state.get("sequence")!=entry["sequence"] or state.get("person_id")!=entry["person_id"] or state.get("condition")!=label:raise ValueError("unit identity/schema mismatch")
 if state.get("protocol_hash")!=protocol_hash or state.get("input_cache_hash")!=entry["cache_sha256"] or state.get("model_hashes")!=h:raise ValueError("unit hash mismatch")
 if state.get("checksum_sha256")!=canonical_hash(state):raise ValueError("unit checksum mismatch")


def _finite_metric_tree(node):
 if isinstance(node,dict):return all(_finite_metric_tree(value) for value in node.values())
 if isinstance(node,list):return np.isfinite(np.asarray(node,dtype=np.float64)).all()
 if isinstance(node,(float,np.floating)):return np.isfinite(node)
 return True


def validate_completed_unit(state,entry,label,protocol_hash,h,paths):
 validate_partial(state,entry,label,protocol_hash,h)
 if state.get("completion") is not True:raise ValueError("unit is incomplete")
 for method in ("raw","v2"):
  path=paths[f"{method}_rep"]
  if not path.exists() or file_sha256(path)!=state["representation_checksums"][method]:raise ValueError(f"{method} representation checksum mismatch")
  with np.load(path,allow_pickle=False) as z:
   required={"frame_id","local_frame_index","h36m_joints","j14_pelvis_centered","converter_fitting_error"}
   if not required.issubset(z.files) or not any(key.startswith("param__") for key in z.files):raise ValueError(f"{method} representation schema mismatch")
   if any(not np.isfinite(np.asarray(z[key])).all() for key in z.files):raise ValueError(f"{method} representation non-finite")
 metric=json.loads(paths["metrics"].read_text())
 if metric.get("checksum_sha256")!=canonical_hash(metric) or file_sha256(paths["metrics"])!=state["metrics_sha256"]:raise ValueError("metric checksum mismatch")
 if not _finite_metric_tree({key:value for key,value in metric.items() if key!="checksum_sha256"}):raise ValueError("metric contains non-finite value")
 return state


def persist_rep(path,conversion,frame_ids,local_indices,reg):
 vertices=np.asarray(conversion["vertices"],np.float32);fit=np.asarray(conversion["errors"],np.float32)
 h36m=np.einsum("jv,tvc->tjc",reg,vertices).astype(np.float32);j14=(h36m-h36m[:,:1])[:,(6,5,4,1,2,3,16,15,14,11,12,13,8,10)]
 arrays={"frame_id":frame_ids,"local_frame_index":local_indices,"h36m_joints":h36m,"j14_pelvis_centered":j14,"converter_fitting_error":fit}
 arrays.update({k:np.asarray(conversion[k],np.float32) for k in conversion.files if k.startswith("param__")})
 atomic_npz(path,**arrays);return vertices,h36m,j14,fit


def process_unit(entry,label,ci,protocol_hash,h,est,adapter,root=OUT,limit=None):
 p=unit_paths(entry["sequence"],entry["person_id"],label,root);p["state"].parent.mkdir(parents=True,exist_ok=True)
 state=json.loads(p["state"].read_text()) if p["state"].exists() else state_base(entry,label,protocol_hash,h);validate_partial(state,entry,label,protocol_hash,h) if p["state"].exists() else None
 if state.get("completion") is True:return state
 with np.load(entry["cache_path"],allow_pickle=False) as z:
  frame_ids=np.asarray(z["frame_id"]);local=np.asarray(z["local_frame_index"]);raw=np.asarray(z["pose_token"][ci],np.float32);init=np.asarray(z["mhr_init_pose"][ci],np.float32)
 if limit is not None:frame_ids,local,raw,init=frame_ids[:limit],local[:limit],raw[:limit],init[:limit]
 if not p["raw_mhr"].exists() or not p["v2_mhr"].exists():
  refined,coverage=refine_all(adapter,raw,RATIOS[ci],local);baseline,v2=decode_mhr(est.model.head_pose,raw,refined,init);npy_atomic(p["raw_mhr"],baseline);npy_atomic(p["v2_mhr"],v2)
  state["stages"]["decode_raw_complete"]=True;state["stages"]["decode_dyna_complete"]=True;state["coverage"]={"frames":len(raw),"windows":len(coverage["windows"])};save_state(p["state"],state)
 if not p["raw_conv"].exists():run_converter(p["raw_mhr"],p["raw_conv"]);state["stages"]["raw_converter_complete"]=True;save_state(p["state"],state)
 if not p["v2_conv"].exists():run_converter(p["v2_mhr"],p["v2_conv"]);state["stages"]["dyna_converter_complete"]=True;save_state(p["state"],state)
 keys=[(entry["sequence"],entry["person_id"],int(x)) for x in frame_ids];gt_records=build_gt_records(keys,gt_dir=DATA/"sequenceFiles/test" if root==OUT else DATA/"sequenceFiles/validation",smpl_model_path=SMPL,device="cpu");gt=np.stack([x["smpl_vertices"] for x in gt_records]);reg=np.load(REG)
 with np.load(p["raw_conv"],allow_pickle=False) as rawc, np.load(p["v2_conv"],allow_pickle=False) as v2c:
  rv,_,rj,rfit=persist_rep(p["raw_rep"],rawc,frame_ids,local,reg);vv,_,vj,vfit=persist_rep(p["v2_rep"],v2c,frame_ids,local,reg)
  rm=compute_geometry_metrics(rv,gt,reg);vm=compute_geometry_metrics(vv,gt,reg);ra=acceleration_error(rj,rm["gt_j14"],local);va=acceleration_error(vj,vm["gt_j14"],local)
  result={"schema":"dynav2-unified-metrics-v1","sequence":entry["sequence"],"person_id":entry["person_id"],"condition":label,"frames":len(frame_ids),"acceleration_triplets":len(ra["per_triplet"]),"raw":{"mpjpe_mm":rm["mpjpe_mm"].tolist(),"pa_mpjpe_mm":rm["pa_mpjpe_mm"].tolist(),"mpvpe_mm":rm["mpvpe_mm"].tolist(),"accel_mm_per_processed_frame2":ra["per_triplet"].tolist(),"converter_fitting_error_mm":(rfit*1000).tolist()},"v2":{"mpjpe_mm":vm["mpjpe_mm"].tolist(),"pa_mpjpe_mm":vm["pa_mpjpe_mm"].tolist(),"mpvpe_mm":vm["mpvpe_mm"].tolist(),"accel_mm_per_processed_frame2":va["per_triplet"].tolist(),"converter_fitting_error_mm":(vfit*1000).tolist()}}
  result["checksum_sha256"]=canonical_hash(result);atomic_json(p["metrics"],result)
 state["stages"]["representation_persisted"]=True;state["representation_checksums"]={"raw":file_sha256(p["raw_rep"]),"v2":file_sha256(p["v2_rep"])};state["metrics_sha256"]=file_sha256(p["metrics"]);state["stages"]["metrics_complete"]=True;state["completion"]=True;save_state(p["state"],state)
 for key in ("raw_mhr","v2_mhr","raw_conv","v2_conv"):
  p[key].unlink(missing_ok=True)
 return state


def aggregate(manifest):
 per=[];all_values={m:{l:defaultdict(list) for l in LABELS} for m in ("raw","v2")}
 for e in manifest["entries"]:
  row={"sequence":e["sequence"],"person_id":e["person_id"],"conditions":{}}
  for label in LABELS:
   p=unit_paths(e["sequence"],e["person_id"],label);x=json.loads(p["metrics"].read_text());row["conditions"][label]={}
   for m in ("raw","v2"):
    summary={k:float(np.mean(v)) for k,v in x[m].items()};row["conditions"][label][m]=summary
    for k,v in x[m].items():all_values[m][label][k].extend(v)
  per.append(row)
 overall={m:{l:{k:float(np.mean(v)) for k,v in all_values[m][l].items()} for l in LABELS} for m in ("raw","v2")}
 atomic_json(OUT/"per_track_metrics.json",{"tracks":per});atomic_json(OUT/"body_metrics.json",overall)
 rng=np.random.default_rng(SEED);boot={}
 for label in LABELS:
  boot[label]={}
  for metric in ("mpjpe_mm","pa_mpjpe_mm","mpvpe_mm","accel_mm_per_processed_frame2"):
   delta=np.array([x["conditions"][label]["raw"][metric]-x["conditions"][label]["v2"][metric] for x in per]);samples=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
   boot[label][metric]={"mean_paired_gain":float(delta.mean()),"ci95":[float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))],"wins":int((delta>0).sum()),"tracks":len(delta)}
 atomic_json(OUT/"bootstrap_statistics.json",boot);summary={"state":"complete","protocol":"B","tracks":len(per),"frames":manifest["valid_person_frames"],"overall":overall,"bootstrap":boot};atomic_json(OUT/"summary.json",summary)


def preflight(protocol_hash,h):
 manifest=json.loads((VAL_CACHE/"manifest.json").read_text());entries=[x for x in manifest["entries"] if x["split"]=="validation"]
 byseq=defaultdict(list)
 for e in entries:byseq[e["sequence"]].append(e)
 candidate=next(group[0] for group in byseq.values() if len(group)>1 and group[0]["frames"]>32)
 entry={"sequence":candidate["sequence"],"person_id":candidate["person_id"],"frames":17,"cache_path":candidate["path"],"cache_sha256":file_sha256(candidate["path"])}
 root=OUT/"validation_preflight";shutil.rmtree(root,ignore_errors=True);est,adapter=load_models();state=process_unit(entry,"clean",0,protocol_hash,h,est,adapter,root,limit=17)
 before=file_sha256(unit_paths(entry["sequence"],entry["person_id"],"clean",root)["v2_rep"]);state2=process_unit(entry,"clean",0,protocol_hash,h,est,adapter,root,limit=17);after=file_sha256(unit_paths(entry["sequence"],entry["person_id"],"clean",root)["v2_rep"])
 if before!=after or not state2["completion"]:raise RuntimeError("preflight resume/determinism failed")
 report={"state":"passed","contract":"FULL_FRAME_TEMPORAL_CONTRACT_LOCKED","track":entry,"frames":17,"non_multiple_of_16":True,"multi_person_sequence":True,"long_track_contract_checked":candidate["frames"],"representation_sha256":after};atomic_json(OUT/"validation_preflight.json",report);return report


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--prepare",action="store_true");ap.add_argument("--preflight",action="store_true");ap.add_argument("--run",action="store_true");args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);[x.mkdir(parents=True,exist_ok=True) for x in (UNITS,REPS,METRICS)]
 h=hashes();manifest=create_manifest();proto=protocol(manifest,h);atomic_json(OUT/"config.json",{"seed":SEED,"labels":LABELS,"ratios":RATIOS,"models_eval":True,"requires_grad":False,"optimizer":None,"estimated_parameter_representation_bytes":manifest["valid_person_frames"]*8*100*4})
 if args.prepare:print(json.dumps({"state":"prepared","manifest":{k:manifest[k] for k in ("sequences","person_tracks","valid_person_frames","total_units")},"protocol_hash":proto["checksum_sha256"]}));return
 if args.preflight:print(json.dumps(preflight(proto["checksum_sha256"],h)));return
 if not args.run:raise ValueError("choose --prepare, --preflight, or --run")
 pre=json.loads((OUT/"validation_preflight.json").read_text());
 if pre.get("state")!="passed":raise RuntimeError("validation preflight not passed")
 write_lock(proto,h);completed=0;failed=[];est,adapter=load_models();started=time.time()
 for e in manifest["entries"]:
  for ci,label in enumerate(LABELS):
   p=unit_paths(e["sequence"],e["person_id"],label)
   if p["state"].exists():
    s=json.loads(p["state"].read_text());validate_partial(s,e,label,proto["checksum_sha256"],h)
    if s.get("completion") is True:validate_completed_unit(s,e,label,proto["checksum_sha256"],h,p);completed+=1;continue
   atomic_json(OUT/"status.json",{"state":"running","stage":"evaluation","completed_units":completed,"total_units":manifest["total_units"],"completed_tracks":completed//4,"total_tracks":manifest["person_tracks"],"current_sequence":e["sequence"],"current_person":e["person_id"],"current_condition":label,"current_stage":"resume_or_decode","elapsed_seconds":time.time()-started,"failed_units":failed})
   try:process_unit(e,label,ci,proto["checksum_sha256"],h,est,adapter);completed+=1
   except Exception as exc:
    failed.append({"sequence":e["sequence"],"person_id":e["person_id"],"condition":label,"error":repr(exc)});atomic_json(OUT/"status.json",{"state":"failed","completed_units":completed,"total_units":manifest["total_units"],"failed_units":failed});raise
 aggregate(manifest);atomic_json(OUT/"status.json",{"state":"complete","stage":"complete","completed_units":completed,"total_units":manifest["total_units"],"completed_tracks":manifest["person_tracks"],"total_tracks":manifest["person_tracks"],"current_sequence":None,"current_person":None,"current_condition":None,"current_stage":None,"elapsed_seconds":time.time()-started,"failed_units":[]})


if __name__=="__main__":main()
