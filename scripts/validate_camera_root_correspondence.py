from __future__ import annotations

import numpy as np


def _finite(name, value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite values')
    return value


def apply_surface_mapping(vertices, faces, triangle_ids, barycentric_coordinates):
    vertices = _finite('vertices', vertices)
    faces = np.asarray(faces, dtype=np.int64)
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
    barycentric_coordinates = _finite('barycentric coordinates', barycentric_coordinates)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError('vertices must have shape (N,V,3)')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('faces must have shape (F,3)')
    if triangle_ids.ndim != 1 or barycentric_coordinates.shape != (len(triangle_ids), 3):
        raise ValueError('surface mapping arrays are not aligned')
    if triangle_ids.min(initial=0) < 0 or triangle_ids.max(initial=-1) >= len(faces):
        raise ValueError('triangle id outside source topology')
    if not np.allclose(barycentric_coordinates.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError('barycentric coordinates must sum to one')
    triangles = faces[triangle_ids]
    return np.einsum('ntkc,tk->ntc', vertices[:, triangles], barycentric_coordinates)


def normalize_joint_regressor(h36m_regressor):
    regressor = _finite('H36M regressor', h36m_regressor)
    if regressor.ndim != 2:
        raise ValueError('H36M regressor must have shape (J,V)')
    row_sums = regressor.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums) < 1e-12):
        raise ValueError('H36M regressor contains a zero-sum row')
    return regressor / row_sums


def regress_common_pelvis(smpl_topology_vertices, h36m_regressor):
    vertices = _finite('SMPL-topology vertices', smpl_topology_vertices)
    regressor = _finite('H36M regressor', h36m_regressor)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError('SMPL-topology vertices must have shape (N,V,3)')
    if regressor.ndim != 2 or regressor.shape[1] != vertices.shape[1]:
        raise ValueError('H36M regressor does not match SMPL topology')
    pelvis_weights = regressor[0]
    if not np.isclose(pelvis_weights.sum(), 1.0, atol=1e-6):
        raise ValueError('pelvis regressor weights must sum to one')
    return np.einsum('v,nvc->nc', pelvis_weights, vertices)


def world_to_camera(points, rotation_cw, translation_cw):
    points = _finite('world points', points)
    rotation = _finite('world-to-camera rotation', rotation_cw)
    translation = _finite('world-to-camera translation', translation_cw)
    return np.einsum('...ij,...vj->...vi', rotation, points) + translation[..., None, :]


def camera_to_world(points, rotation_cw, translation_cw):
    points = _finite('camera points', points)
    rotation = _finite('world-to-camera rotation', rotation_cw)
    translation = _finite('world-to-camera translation', translation_cw)
    return np.einsum('...ji,...vj->...vi', rotation, points - translation[..., None, :])
