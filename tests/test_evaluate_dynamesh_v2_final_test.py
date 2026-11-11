import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import unittest

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_dynamesh_v2_final_test as ft


class FinalTestContracts(unittest.TestCase):
    def test_only_test_split_is_accepted(self):
        ft.assert_final_test_split('test')
        for split in ('train', 'validation'):
            with self.assertRaises(ValueError): ft.assert_final_test_split(split)

    def test_contiguous_windows_never_cross_discontinuity(self):
        local = np.array([0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(ft.contiguous_windows(local, length=4), [(0, 4), (4, 8)])

    def test_track_checksum_detects_mutation(self):
        import tempfile
        payload = {'schema_version': ft.TRACK_SCHEMA, 'sequence': 's', 'person_id': 0, 'frames': 1,
                   'conditions': {c: {'baseline': {}, 'v2': {}} for c in ft.LABELS},
                   'finite': True, 'completion': True}
        payload['checksum_sha256'] = ft.canonical_hash(payload)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'track.json'; path.write_text(json.dumps(payload))
            self.assertTrue(ft.validate_completed_track(path)['completion'])
            payload['frames'] = 2; path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'checksum'): ft.validate_completed_track(path)

    def test_lock_sentence_is_exact(self):
        self.assertEqual(ft.FREEZE_SENTENCE, 'DynaMesh4D-v2 architecture and checkpoints were frozen before first access to the 3DPW test split.')


if __name__ == '__main__': unittest.main()
