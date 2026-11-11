from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/sam-body4d-0.2.0")
CACHE = ROOT / "data/dynamesh_v2/cache_v2"
contract_path = ROOT / "models/sam_3d_body/sam_3d_body/data/dynamesh_v2_cache_v2.py"
spec = importlib.util.spec_from_file_location("dynamesh_v2_cache_contract", contract_path)
contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
file_sha256 = contract.file_sha256
validate_track_payload = contract.validate_track_payload


def main():
    status = json.loads((CACHE / "cache_build_status.json").read_text())
    manifest = json.loads((CACHE / "manifest.json").read_text())
    manifest_root = ROOT / "data/dynamesh_v2/manifests"
    train = json.loads((manifest_root / "3dpw_train_manifest.json").read_text())
    val = json.loads((manifest_root / "3dpw_validation_manifest.json").read_text())
    audit = json.loads((manifest_root / "3dpw_split_audit.json").read_text())
    assert status["state"] == "complete" and status["completed_tracks"] == status["total_tracks"] == 50
    entries = manifest["entries"]
    assert len(entries) == 50 and manifest["test_cache_files"] == 0
    train_entries = [x for x in entries if x["split"] == "train"]
    val_entries = [x for x in entries if x["split"] == "validation"]
    assert len(train_entries) == 34 and len(val_entries) == 16
    assert not (set(train["sequences"]) & set(val["sequences"]))
    assert not audit["train_test_overlap"] and not audit["validation_test_overlap"] and not audit["train_validation_overlap"]
    failures, seen, total_bytes = [], set(), 0
    details = []
    for entry in entries:
        path = Path(entry["path"]); total_bytes += path.stat().st_size
        key = (entry["split"], entry["sequence"], entry["person_id"])
        try:
            assert key not in seen; seen.add(key)
            assert file_sha256(path) == entry["sha256"]
            with np.load(path, allow_pickle=False) as payload:
                validate_track_payload(payload)
                assert str(payload["split"].item()) == entry["split"]
                assert str(payload["sequence"].item()) == entry["sequence"]
                assert int(payload["person_id"].item()) == entry["person_id"]
                assert payload["pose_token"].dtype == np.float16
                assert payload["mhr_init_pose"].dtype == np.float32
                assert payload["body_cont"].dtype == np.float32
                assert payload["pose_token"].shape[1] == entry["frames"]
                details.append({"key": key, "frames": entry["frames"], "token": list(payload["pose_token"].shape), "init": list(payload["mhr_init_pose"].shape), "body": list(payload["body_cont"].shape)})
        except Exception as exc:
            failures.append({"key": key, "error": repr(exc)})
    windows = manifest["validation_windows_length16_nonoverlap"]
    occupied = set()
    val_keys = {(x["sequence"], x["person_id"]): x for x in val_entries}
    for window in windows:
        key = (window["sequence"], window["person_id"])
        assert key in val_keys and window["cache_stop"] - window["cache_start"] == 16 and len(window["frame_ids"]) == 16
        for offset in range(window["cache_start"], window["cache_stop"]):
            slot = key + (offset,); assert slot not in occupied; occupied.add(slot)
        with np.load(val_keys[key]["path"], allow_pickle=False) as payload:
            actual = payload["frame_id"][window["cache_start"]:window["cache_stop"]].tolist()
            local = payload["local_frame_index"][window["cache_start"]:window["cache_stop"]]
            assert actual == window["frame_ids"] and np.all(np.diff(local) == 1)
    rng = np.random.default_rng(20260831)
    sample = [details[i] for group in (train_entries, val_entries) for i in []]
    by_key = {tuple(x["key"]): x for x in details}
    random_audit = []
    for group in (train_entries, val_entries):
        for index in rng.choice(len(group), size=3, replace=False):
            item = group[int(index)]; random_audit.append(by_key[(item["split"], item["sequence"], item["person_id"])])
    result = {
        "passed": not failures, "state": status["state"], "completed_tracks": len(entries),
        "train_tracks": len(train_entries), "validation_tracks": len(val_entries),
        "failed_tracks": len(failures), "failures": failures, "test_cache_files": manifest["test_cache_files"],
        "train_frames": manifest["train_frames"], "validation_frames": manifest["validation_frames"],
        "validation_windows": len(windows), "cache_bytes": total_bytes,
        "random_six_track_audit": random_audit,
    }
    output = CACHE / "completion_audit.json"; output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
