import unittest

import run_three_dpw_oc_alpha015_final as runner


class Alpha015FinalContractTest(unittest.TestCase):
    def test_frozen_alpha_and_output_are_distinct(self):
        self.assertEqual(runner.ALPHA, 0.15)
        self.assertNotEqual(runner.OUT, runner.OLD)
        self.assertEqual(runner.CHECKPOINT_SHA256, "f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329")


if __name__ == "__main__":
    unittest.main()
