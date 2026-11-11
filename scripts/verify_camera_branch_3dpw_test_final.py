from pathlib import Path
import json, sys
import numpy as np

ROOT=Path('/root/sam-body4d-0.2.0')
sys.path.insert(0,str(ROOT/'scripts'))
import run_camera_branch_3dpw_test_final as r

out=r.OUT
required=['protocol.json','checkpoint_audit.json','geometry_audit.json','body_identity_audit.json',
          'sequence_status.json','per_sequence_results.json','per_sequence_results.csv',
          'instance_weighted_summary.json','per_sequence_mean_summary.json','camera_ablation_summary.json',
          'motion_preservation_analysis.json','run.log']
checks={'required_files':all((out/x).is_file() and (out/x).stat().st_size>0 for x in required)}
status=json.loads((out/'sequence_status.json').read_text())
checks['four_sequences_complete']=status['state']=='complete' and status['completed_sequences']==list(r.SEQUENCES) and not status['pending_sequences']
audit=json.loads((out/'checkpoint_audit.json').read_text())
checks['checkpoint_hashes']=audit['sha256']=={'framewise':r.EXPECTED_FRAME,'temporal':r.EXPECTED_TEMP}
saved=json.loads((out/'per_sequence_results.json').read_text())
max_metric_delta=0.; total_frames=total_vel=total_acc=total_jerk=0
for seq in r.SEQUENCES:
    arrays={m:np.load(out/m/f'{seq}.npz',allow_pickle=False) for m in r.METHODS}
    ref=arrays['raw']; n=len(ref['local_frame_index']); segments=r.contiguous_segments(ref['local_frame_index'])
    total_frames+=n; total_vel+=sum(max(len(s)-1,0) for s in segments)
    total_acc+=sum(max(len(s)-2,0) for s in segments); total_jerk+=sum(max(len(s)-3,0) for s in segments)
    assert len(np.unique(np.stack((ref['local_frame_index'],ref['frame_id']),1),axis=0))==n
    for m in r.METHODS:
        z=arrays[m]
        assert np.array_equal(z['local_frame_index'],ref['local_frame_index'])
        assert np.array_equal(z['frame_id'],ref['frame_id'])
        assert np.array_equal(z['body_local_joints_m'],ref['body_local_joints_m'])
        assert np.isfinite(z['pred_world_joints_m']).all() and np.isfinite(z['pred_cam_t']).all()
        calc=r.aggregate_components(r.sequence_components(z['pred_world_joints_m'],ref['gt_world_joints_m'],ref['local_frame_index'],segments))
        for k,v in calc.items():
            if isinstance(v,(int,float)):
                max_metric_delta=max(max_metric_delta,abs(float(v)-float(saved[m][seq][k])))
checks['identity_and_finite']=True
checks['recomputed_metrics_match']=max_metric_delta<1e-9
checks['coverage_counts']={'frames':total_frames,'velocity_steps':total_vel,'acceleration_triplets':total_acc,'jitter_quadruplets':total_jerk}
checks['coverage_expected']=checks['coverage_counts']=={'frames':4157,'velocity_steps':4131,'acceleration_triplets':4111,'jitter_quadruplets':4094}
checks['body_identity']=json.loads((out/'body_identity_audit.json').read_text())['verdict']=='PASS'
checks['geometry']=json.loads((out/'geometry_audit.json').read_text())['verdict']=='PASS'
verdict='PASS' if all(v is True or isinstance(v,dict) for k,v in checks.items() if k!='coverage_counts') else 'FAIL'
result={'verdict':verdict,'checks':checks,'max_recomputed_metric_abs_delta':max_metric_delta}
r.atomic_json(out/'FINAL_VERIFICATION.json',result)
print(json.dumps(result,indent=2))
if verdict!='PASS':raise SystemExit(1)
