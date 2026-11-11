"""Phase-2 generalization training on strict per-track cache-v2 windows."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/root/sam-body4d-0.2.0")
CACHE = ROOT / "data/dynamesh_v2/cache_v2"
OUTPUT = ROOT / "outputs/dynamesh_v2_phase2_training"
SEED = 20260831
WINDOW = 16
RATIOS = (0.0, 0.15, 0.35, 0.55)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models/sam_3d_body"))

from train_dynamesh_v2_micro_overfit import (
    assert_optimizer_owns_only, body_rotation_geodesic_loss,
    decode_body_articulation_rotmats,
)


def contiguous_window_starts(local: np.ndarray, length: int = WINDOW, stride: int = 1):
    starts = []
    run = 0
    for end in range(1, len(local) + 1):
        boundary = end == len(local) or local[end] != local[end - 1] + 1
        if boundary:
            starts.extend(range(run, end - length + 1, stride)); run = end
    return starts


class CacheTracks:
    def __init__(self, entries):
        self.tracks = []
        for entry in entries:
            with np.load(entry["path"], allow_pickle=False) as source:
                self.tracks.append({key: np.asarray(source[key]) for key in (
                    "pose_token", "mhr_init_pose", "body_cont", "local_frame_index", "frame_id"
                )})


class TrainWindows(Dataset):
    def __init__(self, tracks: CacheTracks):
        self.tracks = tracks.tracks; self.windows = []
        for track_index, track in enumerate(self.tracks):
            self.windows.extend((track_index, start) for start in contiguous_window_starts(track["local_frame_index"]))
    def __len__(self): return len(self.windows)
    def __getitem__(self, index):
        track_index, start = self.windows[index]; track = self.tracks[track_index]; sl = slice(start, start + WINDOW)
        return track_index, start, track["pose_token"][:, sl], track["mhr_init_pose"][:, sl], track["body_cont"][0, sl]


class FixedValWindows(Dataset):
    def __init__(self, tracks: CacheTracks, entries, windows):
        self.tracks = tracks.tracks
        lookup = {(x["sequence"], x["person_id"]): i for i, x in enumerate(entries)}
        self.windows = [(lookup[(x["sequence"], x["person_id"])], x["cache_start"]) for x in windows]
    def __len__(self): return len(self.windows)
    def __getitem__(self, index):
        track_index, start = self.windows[index]; track = self.tracks[track_index]; sl = slice(start, start + WINDOW)
        return track["pose_token"][:, sl], track["mhr_init_pose"][:, sl], track["body_cont"][0, sl]


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def formal_body_cont(head, token, init):
    shape = token.shape
    pred = head.proj(token.reshape(-1, shape[-1])) + init.reshape(-1, init.shape[-1])
    return pred[:, 6:266].reshape(*shape[:-1], 260)


def compute_losses(adapter, head, tokens, init, teacher_cont, condition):
    # tokens/init are condition-explicit cache fields and always indexed together.
    batch = tokens.shape[0]; device = tokens.device
    selected = tokens[torch.arange(batch, device=device), condition]
    selected_init = init[torch.arange(batch, device=device), condition]
    ratio = torch.as_tensor(RATIOS, device=device)[condition][:, None].expand(batch, WINDOW)
    refined = adapter(selected, ratio)
    refined_clean = adapter(tokens[:, 0], torch.zeros((batch, WINDOW), device=device))
    refined_cont = formal_body_cont(head, refined, selected_init)
    refined_rot = decode_body_articulation_rotmats(refined_cont)
    with torch.no_grad(): teacher_rot = decode_body_articulation_rotmats(teacher_cont).detach()
    feature = F.smooth_l1_loss(refined, tokens[:, 0])
    body = body_rotation_geodesic_loss(refined_rot, teacher_rot)
    anchor = F.smooth_l1_loss(refined_clean, tokens[:, 0])
    return feature + 0.1 * body + anchor, feature, body, anchor


@torch.no_grad()
def validate(adapter, head, loader, device):
    adapter.eval(); accum = [{"total": 0., "feature": 0., "body": 0., "anchor": 0., "count": 0} for _ in range(4)]
    for tokens, init, teacher in loader:
        tokens=tokens.to(device=device,dtype=torch.float32); init=init.to(device); teacher=teacher.to(device)
        for condition in range(4):
            c = torch.full((tokens.shape[0],), condition, device=device, dtype=torch.long)
            values = compute_losses(adapter, head, tokens, init, teacher, c)
            for key, value in zip(("total","feature","body","anchor"), values): accum[condition][key] += float(value) * tokens.shape[0]
            accum[condition]["count"] += tokens.shape[0]
    for item in accum:
        for key in ("total","feature","body","anchor"): item[key] /= item["count"]
    return accum, float(np.mean([x["total"] for x in accum]))


def train(max_epochs=12, patience=3, batch_size=16, lr=1e-3):
    from omegaconf import OmegaConf
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.models.modules.dynamesh_v2_temporal import BodyFeatureTemporalAdapter
    set_seed(); device=torch.device("cuda"); OUTPUT.mkdir(parents=True, exist_ok=True)
    audit=json.loads((CACHE/"completion_audit.json").read_text()); manifest=json.loads((CACHE/"manifest.json").read_text())
    if not audit["passed"] or audit["failed_tracks"] or manifest["test_cache_files"]: raise RuntimeError("cache audit is not clean")
    train_entries=[x for x in manifest["entries"] if x["split"]=="train"]
    val_entries=[x for x in manifest["entries"] if x["split"]=="validation"]
    train_tracks=CacheTracks(train_entries); val_tracks=CacheTracks(val_entries)
    train_set=TrainWindows(train_tracks); val_set=FixedValWindows(val_tracks,val_entries,manifest["validation_windows_length16_nonoverlap"])
    generator=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0,drop_last=False)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=0)
    cfg=OmegaConf.load(ROOT/"configs/body4d.yaml")
    model,model_cfg=load_sam_3d_body(cfg.sam_3d_body.ckpt_path,device=device,mhr_path=cfg.sam_3d_body.mhr_path)
    estimator=SAM3DBodyEstimator(model,model_cfg,human_detector=None,human_segmentor=None,fov_estimator=None)
    for p in estimator.model.parameters(): p.requires_grad_(False)
    estimator.model.eval(); head=estimator.model.head_pose
    adapter=BodyFeatureTemporalAdapter().to(device)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=lr,weight_decay=1e-4)
    assert_optimizer_owns_only(optimizer,adapter)
    assert not any(p.requires_grad for p in estimator.model.parameters())
    trainable=sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    config={"seed":SEED,"window_length":WINDOW,"train_tracks":len(train_entries),"validation_tracks":len(val_entries),"train_windows":len(train_set),"validation_windows":len(val_set),"batch_size":batch_size,"max_epochs":max_epochs,"patience":patience,"optimizer":"AdamW","learning_rate":lr,"weight_decay":1e-4,"loss_weights":{"feature":1.0,"body":0.1,"anchor":1.0},"checkpoint_selection":"mean total loss across fixed clean/15/35/55 validation windows","trainable_parameters":trainable,"adapter":{"input_dim":1024,"d_model":256,"n_layers":2,"d_state":16,"d_conv":4,"expand":2}}
    (OUTPUT/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    history=[]; best=math.inf; best_epoch=0; stale=0; global_step=0; started=time.time()
    for epoch in range(1,max_epochs+1):
        adapter.train(); sums={k:0. for k in ("total","feature","body","anchor")}; count=0
        for batch_index, (_,_,tokens,init,teacher) in enumerate(train_loader):
            tokens=tokens.to(device=device,dtype=torch.float32); init=init.to(device); teacher=teacher.to(device)
            # Every consecutive four optimizer steps contain each condition once.
            block=torch.randperm(4,generator=generator).tolist() if batch_index%4==0 else block
            condition=torch.full((tokens.shape[0],),block[batch_index%4],device=device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            values=compute_losses(adapter,head,tokens,init,teacher,condition); total=values[0]
            if not torch.isfinite(total): raise FloatingPointError(f"non-finite loss epoch={epoch} step={global_step}")
            total.backward(); optimizer.step(); global_step+=1
            for key,value in zip(("total","feature","body","anchor"),values): sums[key]+=float(value.detach())*tokens.shape[0]
            count+=tokens.shape[0]
        train_metrics={key:value/count for key,value in sums.items()}
        val_conditions,val_mean=validate(adapter,head,val_loader,device)
        item={"epoch":epoch,"global_step":global_step,"train":train_metrics,"validation_conditions":val_conditions,"validation_mean_total":val_mean,"learning_rate":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.time()-started}
        history.append(item); (OUTPUT/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        improved=val_mean<best-1e-8
        if improved:
            best=val_mean; best_epoch=epoch; stale=0
            torch.save({"adapter_state_dict":adapter.state_dict(),"config":config,"best_epoch":epoch,"best_validation_mean_total":best},OUTPUT/"best.pt")
        else: stale+=1
        status={"state":"training","epoch":epoch,"global_step":global_step,"best_epoch":best_epoch,"best_validation_mean_total":best,"stale_epochs":stale,"last":item}
        (OUTPUT/"status.json").write_text(json.dumps(status,indent=2)+"\n"); print(json.dumps(status),flush=True)
        if stale>=patience: break
    final={"state":"complete","epochs":len(history),"global_steps":global_step,"best_epoch":best_epoch,"best_validation_mean_total":best,"early_stopped":len(history)<max_epochs,"elapsed_seconds":time.time()-started}
    torch.save({"adapter_state_dict":adapter.state_dict(),"config":config,"final":final},OUTPUT/"final.pt")
    (OUTPUT/"status.json").write_text(json.dumps(final,indent=2)+"\n"); print(json.dumps(final,indent=2))


def self_test():
    assert contiguous_window_starts(np.arange(20),16)==[0,1,2,3,4]
    assert contiguous_window_starts(np.array([0,1,2,20,21,22]),3)==[0,3]
    print(json.dumps({"passed":True,"window_split":"no cross-discontinuity"}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("train","self-test")); parser.add_argument("--batch-size",type=int,default=16); args=parser.parse_args()
    self_test() if args.mode=="self-test" else train(batch_size=args.batch_size)
