import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate_dynamesh_v2_phase2 import (
    LABELS, SCHEMA, canonical_hash, geodesic_from_cosine_deg, validate_track_result,
)
from repair_dynamesh_v2_phase2_track import repair_legacy_geodesic_scalars


def valid_payload(frames=3):
    fields = ('mpjpe_mm','pa_mpjpe_mm','converter_fitting_error_mm',
              'body_teacher_geodesic_deg','baseline_token_l2','token_l2')
    conditions = {}
    for label in LABELS:
        conditions[label] = {}
        for method in ('baseline','v2'):
            conditions[label][method] = {field: [0.0] * frames for field in fields}
    payload = {'schema_version':SCHEMA,'sequence':'seq','person_id':0,'frames':frames,
               'conditions':conditions,'completion':True}
    payload['checksum_sha256'] = canonical_hash(payload)
    return payload


class MetricContractTests(unittest.TestCase):
    def test_geodesic_reduction_is_per_frame_and_identical_is_exact_zero(self):
        cosine = torch.ones(4, 23)
        result = geodesic_from_cosine_deg(cosine)
        self.assertEqual(tuple(result.shape), (4,))
        self.assertTrue(torch.equal(result, torch.zeros(4)))

    def test_validator_covers_every_condition_and_method(self):
        payload = valid_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'track.json'
            path.write_text(json.dumps(payload))
            self.assertEqual(validate_track_result(path, 'seq', 0)['frames'], 3)
            for label in LABELS:
                for method in ('baseline','v2'):
                    broken = json.loads(json.dumps(payload))
                    broken['conditions'][label][method]['body_teacher_geodesic_deg'] = 0.5
                    broken['checksum_sha256'] = canonical_hash(broken)
                    path.write_text(json.dumps(broken))
                    with self.assertRaisesRegex(ValueError, f'{label}/{method}/body_teacher_geodesic_deg invalid'):
                        validate_track_result(path, 'seq', 0)

    def test_offline_repair_broadcasts_legacy_scalars_and_zeroes_clean_baseline(self):
        payload = valid_payload(frames=4)
        for label in LABELS:
            for method in ('baseline','v2'):
                payload['conditions'][label][method]['body_teacher_geodesic_deg'] = 1.25
        repaired = repair_legacy_geodesic_scalars(payload)
        self.assertEqual(repaired['conditions']['clean']['baseline']['body_teacher_geodesic_deg'], [0.0] * 4)
        self.assertEqual(repaired['conditions']['occ55']['v2']['body_teacher_geodesic_deg'], [1.25] * 4)
        self.assertEqual(repaired['metric_contract_repair']['strategy'], 'broadcast_legacy_track_mean')
        self.assertEqual(repaired['checksum_sha256'], canonical_hash(repaired))


if __name__ == '__main__':
    unittest.main(verbosity=2)
