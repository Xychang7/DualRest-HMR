"""Pure protocol and metric helpers for Unified 3DPW Evaluation Protocol B."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

H36M_TO_J14 = (6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10)


def canonical_hash(payload: dict) -> str:
    clean = {k: v for k, v in payload.items() if k != "checksum_sha256"}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    with temp.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: str | Path, **arrays) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    with temp.open("rb") as handle: os.fsync(handle.fileno())
    os.replace(temp, path)


def _segments(local_indices: np.ndarray) -> list[tuple[int, int]]:
    local_indices = np.asarray(local_indices, dtype=np.int64)
    if local_indices.ndim != 1 or len(local_indices) == 0:
        raise ValueError("local_indices must be a non-empty vector")
    if np.any(np.diff(local_indices) <= 0):
        raise ValueError("local_indices must be strictly increasing")
    boundaries = np.flatnonzero(np.diff(local_indices) != 1) + 1
    return list(zip(np.r_[0, boundaries], np.r_[boundaries, len(local_indices)]))


def build_temporal_coverage(local_indices, window_length=16, stride=8) -> dict:
    """Cover every frame with fixed-T windows and select the most central prediction."""
    local_indices = np.asarray(local_indices, dtype=np.int64)
    if window_length < 2 or not (1 <= stride <= window_length):
        raise ValueError("invalid temporal window/stride")
    windows: list[list[int]] = []
    for first, stop in _segments(local_indices):
        length = stop - first
        if length < window_length:
            relative = np.minimum(np.arange(window_length), length - 1)
            windows.append((first + relative).tolist())
        else:
            starts = list(range(first, stop - window_length + 1, stride))
            last = stop - window_length
            if starts[-1] != last: starts.append(last)
            windows.extend([list(range(s, s + window_length)) for s in starts])
    candidates: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(local_indices))}
    for wi, window in enumerate(windows):
        for pos, index in enumerate(window):
            centrality = min(pos, window_length - 1 - pos)
            candidates[index].append((-centrality, wi, pos))
    owner = {i: list(min(options)[1:]) for i, options in candidates.items() if options}
    if len(owner) != len(local_indices):
        raise RuntimeError("temporal coverage did not assign every frame")
    return {"windows": windows, "owner": owner}


def _similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    sm, tm = source.mean(0), target.mean(0)
    x, y = source - sm, target - tm
    variance = np.sum(x * x)
    if variance <= np.finfo(np.float64).eps: raise ValueError("degenerate joints")
    u, _, vh = np.linalg.svd(x.T @ y); v = vh.T
    correction = np.eye(3); correction[-1, -1] = np.sign(np.linalg.det(u @ v.T))
    rotation = v @ correction @ u.T
    scale = np.trace(rotation @ (x.T @ y)) / variance
    return (scale * rotation @ source.T).T + tm - scale * rotation @ sm


def regress_joints(vertices, regressor):
    vertices = np.asarray(vertices, np.float64); regressor = np.asarray(regressor, np.float64)
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3): raise ValueError("vertices must be (T,6890,3)")
    if regressor.shape != (17, 6890): raise ValueError("regressor must be (17,6890)")
    h36m = np.einsum("jv,tvc->tjc", regressor, vertices)
    return h36m, h36m - h36m[:, :1], (h36m - h36m[:, :1])[:, H36M_TO_J14]


def compute_geometry_metrics(pred_vertices, gt_vertices, regressor) -> dict:
    pred = np.asarray(pred_vertices, np.float64); gt = np.asarray(gt_vertices, np.float64)
    if pred.shape != gt.shape or pred.shape[1:] != (6890, 3): raise ValueError("prediction/GT topology mismatch")
    ph, _, pj = regress_joints(pred, regressor); gh, _, gj = regress_joints(gt, regressor)
    mp = np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000.0
    pa = np.asarray([np.linalg.norm(_similarity_align(p, g) - g, axis=-1).mean() * 1000.0 for p, g in zip(pj, gj)])
    pred_centered = pred - ph[:, :1]; gt_centered = gt - gh[:, :1]
    pve = np.linalg.norm(pred_centered - gt_centered, axis=-1).mean(-1) * 1000.0
    return {"mpjpe_mm": mp, "pa_mpjpe_mm": pa, "mpvpe_mm": pve, "pred_h36m": ph, "pred_j14": pj, "gt_h36m": gh, "gt_j14": gj}


def acceleration_error(pred_j14, gt_j14, local_indices) -> dict:
    """VIBE-style second difference: millimetres per processed-frame squared."""
    pred = np.asarray(pred_j14, np.float64); gt = np.asarray(gt_j14, np.float64)
    local = np.asarray(local_indices, np.int64)
    if pred.shape != gt.shape or pred.shape[1:] != (14, 3) or len(local) != len(pred):
        raise ValueError("acceleration inputs must align as (T,14,3)")
    centers = np.asarray([i for i in range(1, len(local)-1) if local[i]-local[i-1] == 1 and local[i+1]-local[i] == 1], np.int64)
    if not len(centers): return {"triplet_centers": centers, "per_triplet": np.empty(0)}
    ap = pred[centers-1] - 2*pred[centers] + pred[centers+1]
    ag = gt[centers-1] - 2*gt[centers] + gt[centers+1]
    return {"triplet_centers": centers, "per_triplet": np.linalg.norm(ap-ag, axis=-1).mean(-1)*1000.0}


def validate_resume_state(path, *, protocol_hash, expected_schema):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != expected_schema: raise ValueError("resume schema mismatch")
    if payload.get("protocol_hash") != protocol_hash: raise ValueError("resume protocol hash mismatch")
    if payload.get("completion") is not True: raise ValueError("resume unit incomplete")
    if payload.get("checksum_sha256") != canonical_hash(payload): raise ValueError("resume checksum mismatch")
    return payload
