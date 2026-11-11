import unittest


class FullBenchmarkHelpersTest(unittest.TestCase):
    def test_metric_comparison_uses_error_sign_convention(self):
        from scripts.run_three_dpw_oc_full import metric_comparison

        sam = {"mpjpe_mm": 30.0, "pa_mpjpe_mm": 20.0,
               "mpvpe_mm": 40.0, "accel_m_s2": 10.0}
        dyna = {"mpjpe_mm": 28.0, "pa_mpjpe_mm": 19.0,
                "mpvpe_mm": 39.5, "accel_m_s2": 10.5}
        result = metric_comparison(sam, dyna)
        self.assertAlmostEqual(result["dyna_minus_sam"]["mpjpe_mm"], -2.0)
        self.assertAlmostEqual(result["sam_minus_dyna_improvement"]["mpjpe_mm"], 2.0)
        self.assertAlmostEqual(result["sam_minus_dyna_improvement"]["accel_m_s2"], -0.5)


    def test_weighted_aggregate_uses_spatial_and_temporal_denominators(self):
        from scripts.run_three_dpw_oc_full import aggregate_reports

        reports = {
        "short": {
            "sam": {"evaluated_oc_instances": 1, "mpjpe_mm": 10.0,
                    "pa_mpjpe_mm": 20.0, "mpvpe_mm": 30.0,
                    "accel_m_s2": 2.0, "valid_accel_triplets": 1},
            "dyna": {"evaluated_oc_instances": 1, "mpjpe_mm": 8.0,
                     "pa_mpjpe_mm": 18.0, "mpvpe_mm": 28.0,
                     "accel_m_s2": 1.0, "valid_accel_triplets": 1},
        },
        "long": {
            "sam": {"evaluated_oc_instances": 3, "mpjpe_mm": 30.0,
                    "pa_mpjpe_mm": 40.0, "mpvpe_mm": 50.0,
                    "accel_m_s2": 6.0, "valid_accel_triplets": 3},
            "dyna": {"evaluated_oc_instances": 3, "mpjpe_mm": 28.0,
                     "pa_mpjpe_mm": 38.0, "mpvpe_mm": 48.0,
                     "accel_m_s2": 5.0, "valid_accel_triplets": 3},
        },
        }
        result = aggregate_reports(reports)
        self.assertAlmostEqual(result["instance_weighted"]["sam"]["mpjpe_mm"], 25.0)
        self.assertAlmostEqual(result["instance_weighted"]["sam"]["accel_m_s2"], 5.0)
        self.assertAlmostEqual(result["per_sequence_mean"]["sam"]["mpjpe_mm"], 20.0)


if __name__ == "__main__":
    unittest.main()
