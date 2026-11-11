"""Explicit common-root geometry for Camera Phase-0.

No learned offsets or empirical scale factors are permitted here. All public
interfaces use metres and explicitly named coordinate systems.
"""
from __future__ import annotations

import numpy as np
import torch


def _check_unit(unit):
    if unit != "meters":
        raise ValueError("Camera root geometry accepts only explicit metre-scale inputs")


def _shape(value, suffix, name):
    if tuple(value.shape[-len(suffix):]) != tuple(suffix):
        raise ValueError(f"{name} must end in shape {suffix}, got {tuple(value.shape)}")


def get_mhr_local_root(mhr_joints, root_index: int, *, unit="meters", coordinate_system="mhr_local"):
    """Select an explicitly indexed MHR root from (..., J, 3) local joints."""
    _check_unit(unit)
    if coordinate_system != "mhr_local": raise ValueError("coordinate_system must be mhr_local")
    _shape(mhr_joints, (3,), "mhr_joints")
    if mhr_joints.ndim < 2 or not 0 <= root_index < mhr_joints.shape[-2]: raise ValueError("root_index out of range")
    return mhr_joints[..., root_index, :]


def get_pred_camera_root(mhr_local_root, pred_cam_t, *, unit="meters"):
    """Place the selected MHR local root in camera coordinates."""
    _check_unit(unit); _shape(mhr_local_root, (3,), "mhr_local_root"); _shape(pred_cam_t, (3,), "pred_cam_t")
    if tuple(mhr_local_root.shape) != tuple(pred_cam_t.shape): raise ValueError("root and pred_cam_t must align exactly")
    return mhr_local_root + pred_cam_t


def get_3dpw_gt_camera_root(smpl_vertices_world, joint_regressor, root_index: int, cam_poses, *, unit="meters", camera_convention="world_to_camera"):
    """Regress a common SMPL root then apply the homogeneous world->camera transform."""
    _check_unit(unit)
    if camera_convention != "world_to_camera": raise ValueError("camera_convention must be world_to_camera")
    _shape(smpl_vertices_world, (3,), "smpl_vertices_world"); _shape(cam_poses, (4, 4), "cam_poses")
    if smpl_vertices_world.ndim != 3 or joint_regressor.ndim != 2: raise ValueError("expected vertices (N,V,3) and regressor (J,V)")
    if len(smpl_vertices_world) != len(cam_poses) or joint_regressor.shape[1] != smpl_vertices_world.shape[1]: raise ValueError("frame/vertex alignment mismatch")
    if not 0 <= root_index < joint_regressor.shape[0]: raise ValueError("root_index out of range")
    if torch.is_tensor(smpl_vertices_world):
        reg=torch.as_tensor(joint_regressor,device=smpl_vertices_world.device,dtype=smpl_vertices_world.dtype)
        camera=torch.as_tensor(cam_poses,device=smpl_vertices_world.device,dtype=smpl_vertices_world.dtype)
        root=torch.einsum("v,nvc->nc",reg[root_index],smpl_vertices_world); ones=torch.ones((*root.shape[:-1],1),device=root.device,dtype=root.dtype)
        return (camera @ torch.cat((root,ones),-1).unsqueeze(-1)).squeeze(-1)[...,:3]
    vertices=np.asarray(smpl_vertices_world); reg=np.asarray(joint_regressor); camera=np.asarray(cam_poses)
    root=np.einsum("v,nvc->nc",reg[root_index],vertices); homogeneous=np.concatenate((root,np.ones((len(root),1),dtype=root.dtype)),axis=-1)
    return np.einsum("nij,nj->ni",camera,homogeneous)[...,:3]


def compare_common_roots(pred_camera_root, gt_camera_root, *, unit="meters"):
    """Return per-frame Euclidean error without alignment or scale correction."""
    _check_unit(unit); _shape(pred_camera_root,(3,),"pred_camera_root"); _shape(gt_camera_root,(3,),"gt_camera_root")
    if tuple(pred_camera_root.shape) != tuple(gt_camera_root.shape): raise ValueError("predicted and GT roots must align exactly")
    if torch.is_tensor(pred_camera_root):
        errors=torch.linalg.vector_norm(pred_camera_root-gt_camera_root,dim=-1); return {"per_frame":errors,"mean":errors.mean(),"unit":"meters"}
    errors=np.linalg.norm(np.asarray(pred_camera_root)-np.asarray(gt_camera_root),axis=-1); return {"per_frame":errors,"mean":float(errors.mean()),"unit":"meters"}
