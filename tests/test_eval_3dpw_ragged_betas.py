import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path('/root/sam-body4d-0.2.0')
sys.path.insert(0, str(ROOT))

from eval_3dpw_standard import select_person_betas
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_dynamesh_v2_final_test as ft


class RaggedBetasContract(unittest.TestCase):
    def test_selects_person_before_array_conversion(self):
        betas = [np.arange(10, dtype=np.float32), np.arange(300, dtype=np.float32)]
        np.testing.assert_array_equal(select_person_betas(betas, 0), betas[0])
        np.testing.assert_array_equal(select_person_betas(betas, 1), betas[1])

    def test_rejects_non_vector_person_betas(self):
        with self.assertRaisesRegex(ValueError, '1D'):
            select_person_betas([np.zeros((2, 10), dtype=np.float32)], 0)

    def test_non_divisible_track_uses_only_complete_windows(self):
        groups = []
        offset = 0
        for i in range(46):
            length = 18 if i < 7 else 17
            groups.append(np.arange(offset, offset + length))
            offset += length + 1
        local = np.concatenate(groups)
        self.assertEqual(len(local), 789)
        windows = ft.contiguous_windows(local, length=16)
        self.assertEqual(len(windows), 46)
        self.assertEqual(sum(b - a for a, b in windows), 736)


if __name__ == '__main__':
    unittest.main()
