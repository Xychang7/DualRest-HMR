import json
import pickle

import numpy as np
import tempfile
from contextlib import contextmanager
from pathlib import Path

from evaluation.three_dpw_oc.manifest import build_manifest, validate_predictions, write_manifest
from evaluation.three_dpw_oc.metrics import (spatial_metrics, acceleration_metrics,
                                               audit_oc_acceleration_keys)
from evaluation.three_dpw_oc.gt import (GenderedSMPLGT, load_corrected_h36m_regressor,
                                         build_oc_gt_meshes)
from evaluation.three_dpw_oc.evaluate import evaluate_records, evaluate_from_oc_json
from evaluation.three_dpw_oc.prediction_adapter import (make_paired_prediction_records,
                                                          paired_intersection_manifest)


@contextmanager
def raises(error_type, match):
    try:
        yield
    except error_type as error:
        assert match in str(error), str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {match}")


def _geometry():
    rng = np.random.default_rng(7)
    vertices = rng.normal(0, 0.3, (5, 6890, 3))
    vertices += np.arange(5)[:, None, None] * 0.01
    h36m = rng.random((17, 6890))
    h36m /= h36m.sum(axis=1, keepdims=True)
    native = rng.random((24, 6890))
    native /= native.sum(axis=1, keepdims=True)
    return vertices, h36m, native


def test_identical_meshes_have_zero_spatial_and_acceleration_error():
    vertices, h36m, native = _geometry()
    for frame in range(5):
        result = spatial_metrics(vertices[frame], vertices[frame], h36m, native)
        assert max(result.values()) < 1e-6
    records = {("seq", t, 0): vertices[t] for t in range(5)}
    result = acceleration_metrics(records, records, {("seq", t, 0) for t in range(5)}, h36m)
    assert result["valid_triplets"] == 3
    assert result["mean_m_s2"] < 1e-10


def test_constant_translation_is_removed_from_all_spatial_metrics():
    vertices, h36m, native = _geometry()
    shifted = vertices[0] + np.array([1.0, -0.8, 2.0])
    result = spatial_metrics(shifted, vertices[0], h36m, native)
    assert max(result.values()) < 1e-5


def test_similarity_alignment_handles_scale_but_mpjpe_does_not():
    vertices, h36m, native = _geometry()
    mesh = vertices[0]
    pelvis = h36m[0] @ mesh
    scaled = pelvis + (mesh - pelvis) * 1.3
    result = spatial_metrics(scaled, mesh, h36m, native)
    assert result["pa_mpjpe_mm"] < 1e-5
    assert result["mpjpe_mm"] > 1.0


def test_acceleration_uses_only_true_adjacent_rgb_frames():
    vertices, h36m, _ = _geometry()
    records = {("seq", t, 0): vertices[i] for i, t in enumerate((0, 2, 4, 6, 8))}
    result = acceleration_metrics(records, records, set(records), h36m)
    assert result["valid_triplets"] == 0
    assert result["invalid_reasons"]["missing_adjacent_frame"] > 0


def test_acceleration_respects_sequence_timestamp_status():
    vertices, h36m, _ = _geometry()
    records = {("seq", t, 0): vertices[t] for t in range(5)}
    centers = {("seq", 1, 0), ("seq", 2, 0)}
    status = {("seq", 1, 0): "valid", ("seq", 2, 0): "irregular_timestamp"}
    result = acceleration_metrics(records, records, centers, h36m,
                                  center_status=status)
    assert result["valid_triplets"] == 1
    assert result["invalid_reasons"] == {"irregular_timestamp": 1}


def test_invalid_prediction_metadata_is_rejected():
    with raises(ValueError, "matched_actor_id"):
        validate_predictions([{"sequence": "seq", "rgb_frame_idx": 1,
                               "pred_track_id": 7, "vertices": np.zeros((6890, 3))}])
    with raises(ValueError, "duplicate"):
        row = {"sequence": "seq", "rgb_frame_idx": 1, "pred_track_id": 7,
               "matched_actor_id": 0,
               "vertices": np.zeros((6890, 3)), "valid": True}
        validate_predictions([row, row])
    with raises(ValueError, "pred_track_id"):
        validate_predictions([{"sequence": "seq", "rgb_frame_idx": 1,
                               "matched_actor_id": 0,
                               "vertices": np.zeros((6890, 3)), "valid": True}])


def test_prediction_track_order_must_be_increasing():
    records = [{"sequence": "seq", "rgb_frame_idx": t, "pred_track_id": 7,
                "matched_actor_id": 0,
                "vertices": np.zeros((6890, 3)), "valid": True} for t in (2, 1)]
    with raises(ValueError, "frame order"):
        validate_predictions(records)


def test_manifest_maps_annotation_person_to_source_actor(tmp_path):
    root = tmp_path / "3DPW"
    (root / "imageFiles" / "seq").mkdir(parents=True)
    (root / "sequenceFiles" / "test").mkdir(parents=True)
    for t in range(3):
        (root / "imageFiles" / "seq" / f"image_{t:05d}.jpg").touch()
    poses = np.zeros((2, 3, 72))
    poses[0, :, 3] = 0.3
    poses[1, :, 3] = 1.2
    with (root / "sequenceFiles" / "test" / "seq.pkl").open("wb") as file:
        pickle.dump({"poses": poses, "img_frame_ids": np.array([0, 2, 4])}, file)
    images = [{"id": t, "sequence": "seq", "frame_idx": t,
               "file_name": f"image_{t:05d}.jpg"} for t in range(3)]
    annotations = []
    for t in range(3):
        for person in range(2):
            annotations.append({"id": 2*t+person, "image_id": t, "person_id": person,
                                "smpl_param": {"pose": poses[person, t].tolist(),
                                               "shape": [0.0]*10, "trans": [0.0]*3,
                                               "gender": "male"}})
    (root / "3DPW_oc.json").write_text(json.dumps({"images": images,
                                                   "annotations": annotations}))
    manifest = build_manifest(root)
    assert len(manifest["instances"]) == 6
    assert {(x["person_id"], x["source_actor_id"]) for x in manifest["instances"]} == {(0, 0), (1, 1)}
    assert manifest["instances"][1]["mocap_sync_id"] == 0
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)
    first = output.read_bytes()
    write_manifest(manifest, output)
    assert output.read_bytes() == first
    assert json.loads(first)["schema"] == "three_dpw_oc_manifest_v1"


def test_manifest_rejects_wrong_rgb_filename(tmp_path):
    root = tmp_path / "3DPW"
    (root / "imageFiles" / "seq").mkdir(parents=True)
    (root / "sequenceFiles" / "test").mkdir(parents=True)
    (root / "imageFiles" / "seq" / "image_00002.jpg").touch()
    with (root / "sequenceFiles" / "test" / "seq.pkl").open("wb") as file:
        pickle.dump({"poses": np.zeros((1, 1, 72)), "img_frame_ids": [0]}, file)
    ann = {"id": 1, "image_id": 1, "person_id": 0,
           "smpl_param": {"pose": [0.0]*72, "shape": [0.0]*10,
                          "trans": [0.0]*3, "gender": "male"}}
    (root / "3DPW_oc.json").write_text(json.dumps({"images": [{"id": 1, "sequence": "seq",
        "frame_idx": 0, "file_name": "image_00002.jpg"}], "annotations": [ann]}))
    with raises(ValueError, "RGB frame"):
        build_manifest(root)


def test_corrected_regressor_loader_does_not_fall_back_to_old_file(tmp_path):
    np.save(tmp_path / "J_regressor_h36m.npy", np.zeros((17, 6890)))
    with raises(FileNotFoundError, "J_regressor_h36m_correct.npy"):
        load_corrected_h36m_regressor(tmp_path)


def test_gendered_gt_builder_returns_finite_standard_smpl_mesh():
    body_models = Path("/root/DuoMo-main/data/body_models/smpl")
    if not body_models.is_dir():
        return
    builder = GenderedSMPLGT(body_models)
    annotation = {"smpl_param": {"pose": [0.0]*72, "shape": [0.0]*10,
                                 "trans": [0.1, -0.2, 2.0], "gender": "male"}}
    vertices = builder.vertices(annotation)
    assert vertices.shape == (6890, 3)
    assert np.isfinite(vertices).all()
    assert builder.native_pelvis_regressor.shape == (24, 6890)
    annotation["smpl_param"]["gender"] = "female"
    female_vertices = builder.vertices(annotation)
    assert np.linalg.norm(vertices - female_vertices) > 0.01


def test_oc_gt_meshes_use_annotation_id_not_sequencefiles():
    root = Path("/root/data1/sam-body4d-master/dataset/3DPW")
    models = Path("/root/DuoMo-main/data/body_models/smpl")
    if not (root / "3DPW_oc.json").is_file() or not models.is_dir():
        return
    raw = json.loads((root / "3DPW_oc.json").read_text())
    annotation = raw["annotations"][0]
    image = next(row for row in raw["images"] if row["id"] == annotation["image_id"])
    entry = {"sequence": image["sequence"], "rgb_frame_idx": image["frame_idx"],
             "source_actor_id": annotation["person_id"], "annotation_id": annotation["id"]}
    builder = GenderedSMPLGT(models)
    expected = builder.vertices(annotation)
    actual = build_oc_gt_meshes(root, [entry], builder)
    key = (entry["sequence"], entry["rgb_frame_idx"], entry["source_actor_id"])
    assert np.array_equal(actual[key], expected)


def test_unified_evaluator_filters_spatial_by_oc_and_accel_by_oc_centers():
    vertices, h36m, native = _geometry()
    all_pred = [{"sequence": "seq", "rgb_frame_idx": t, "pred_track_id": 7,
                 "matched_actor_id": 0,
                 "vertices": vertices[t], "valid": True} for t in range(5)]
    gt_all = {("seq", t, 0): vertices[t] for t in range(5)}
    manifest = {"instances": [{"sequence": "seq", "rgb_frame_idx": t,
                               "source_actor_id": 0, "annotation_id": t} for t in (1, 3)]}
    summary = evaluate_records(manifest, all_pred, gt_all, h36m, native)
    assert summary["spatial_instances"] == 2
    assert summary["acceleration"]["valid_triplets"] == 2
    assert summary["mpjpe_mm"] < 1e-6
    assert summary["mpvpe_mm"] < 1e-6


def test_unified_evaluator_rejects_missing_oc_prediction():
    vertices, h36m, native = _geometry()
    manifest = {"instances": [{"sequence": "seq", "rgb_frame_idx": 1,
                               "source_actor_id": 0, "annotation_id": 1}]}
    with raises(ValueError, "missing OC prediction"):
        evaluate_records(manifest, [], {("seq", 1, 0): vertices[1]}, h36m, native)


def test_high_level_jotr_entrypoint_requires_corrected_regressor(tmp_path):
    with raises(FileNotFoundError, "J_regressor_h36m_correct.npy"):
        evaluate_from_oc_json({"instances": []}, [], tmp_path, tmp_path, tmp_path)


def test_acceleration_key_audit_does_not_join_sparse_oc_frames():
    manifest = {"instances": [
        {"sequence": "seq", "rgb_frame_idx": frame, "source_actor_id": 0}
        for frame in (0, 1, 2, 4)
    ]}
    result = audit_oc_acceleration_keys(manifest)
    assert result["candidate_centers"] == 4
    assert result["valid_triplets"] == 1
    assert result["invalid_reasons"] == {"boundary": 1,
                                           "missing_oc_gt_neighbor": 2}


def test_paired_prediction_adapter_has_identical_keys_and_intersection():
    # Ensure a plausible non-degenerate human-size mesh for contract testing.
    mesh = np.repeat(np.linspace(-0.8, 0.8, 6890)[:, None], 3, axis=1)[None].repeat(3, 0)
    sam, dyna = make_paired_prediction_records("seq", [0, 1, 2], 0, mesh, mesh.copy())
    assert [(x["rgb_frame_idx"], x["matched_actor_id"]) for x in sam] == [
        (x["rgb_frame_idx"], x["matched_actor_id"]) for x in dyna]
    manifest = {"instances": [{"sequence": "seq", "rgb_frame_idx": t,
                                "source_actor_id": 0} for t in (0, 1, 2, 3)]}
    paired, stats = paired_intersection_manifest(manifest, sam, dyna)
    assert len(paired["instances"]) == 3
    assert stats["missing_prediction_instances"] == 1


def test_paired_prediction_adapter_rejects_key_or_topology_mismatch():
    mesh = np.repeat(np.linspace(-0.8, 0.8, 6890)[:, None], 3, axis=1)[None].repeat(2, 0)
    with raises(ValueError, "lengths or topology"):
        make_paired_prediction_records("seq", [0, 1], 0, mesh, mesh[:1])


def test_manifest_checks_even_unannotated_rgb_images(tmp_path):
    root = tmp_path / "3DPW"
    (root / "imageFiles" / "seq").mkdir(parents=True)
    (root / "sequenceFiles" / "test").mkdir(parents=True)
    (root / "imageFiles" / "seq" / "image_00000.jpg").touch()
    with (root / "sequenceFiles" / "test" / "seq.pkl").open("wb") as file:
        pickle.dump({"poses": np.zeros((1, 2, 72)), "img_frame_ids": [0, 2]}, file)
    images = [{"id": t, "sequence": "seq", "frame_idx": t,
               "file_name": f"image_{t:05d}.jpg"} for t in range(2)]
    annotation = {"id": 0, "image_id": 0, "person_id": 0,
                  "smpl_param": {"pose": [0.0]*72, "shape": [0.0]*10,
                                 "trans": [0.0]*3, "gender": "male"}}
    (root / "3DPW_oc.json").write_text(json.dumps({"images": images,
                                                   "annotations": [annotation]}))
    with raises(ValueError, "Missing referenced RGB image"):
        build_manifest(root)


if __name__ == "__main__":
    for test in (test_identical_meshes_have_zero_spatial_and_acceleration_error,
                 test_constant_translation_is_removed_from_all_spatial_metrics,
                 test_similarity_alignment_handles_scale_but_mpjpe_does_not,
                 test_acceleration_uses_only_true_adjacent_rgb_frames,
                 test_acceleration_respects_sequence_timestamp_status,
                 test_invalid_prediction_metadata_is_rejected,
                 test_prediction_track_order_must_be_increasing,
                 test_unified_evaluator_filters_spatial_by_oc_and_accel_by_oc_centers,
                 test_unified_evaluator_rejects_missing_oc_prediction,
                 test_acceleration_key_audit_does_not_join_sparse_oc_frames,
                 test_paired_prediction_adapter_has_identical_keys_and_intersection,
                 test_paired_prediction_adapter_rejects_key_or_topology_mismatch):
        test()
        print("PASS", test.__name__)
    with tempfile.TemporaryDirectory() as temp:
        test_high_level_jotr_entrypoint_requires_corrected_regressor(Path(temp))
        print("PASS test_high_level_jotr_entrypoint_requires_corrected_regressor")
    with tempfile.TemporaryDirectory() as temp:
        test_manifest_maps_annotation_person_to_source_actor(Path(temp))
        print("PASS test_manifest_maps_annotation_person_to_source_actor")
    with tempfile.TemporaryDirectory() as temp:
        test_manifest_rejects_wrong_rgb_filename(Path(temp))
        print("PASS test_manifest_rejects_wrong_rgb_filename")
    with tempfile.TemporaryDirectory() as temp:
        test_manifest_checks_even_unannotated_rgb_images(Path(temp))
        print("PASS test_manifest_checks_even_unannotated_rgb_images")
    with tempfile.TemporaryDirectory() as temp:
        test_corrected_regressor_loader_does_not_fall_back_to_old_file(Path(temp))
        print("PASS test_corrected_regressor_loader_does_not_fall_back_to_old_file")
    test_gendered_gt_builder_returns_finite_standard_smpl_mesh()
    print("PASS test_gendered_gt_builder_returns_finite_standard_smpl_mesh")
    test_oc_gt_meshes_use_annotation_id_not_sequencefiles()
    print("PASS test_oc_gt_meshes_use_annotation_id_not_sequencefiles")
