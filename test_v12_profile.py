import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hn_profile_v12 import cohort_profiles, chronological_drift, export_v12_profile


class V12ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(120)
        n = 100
        times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        cls.frame = pd.DataFrame({
            "title": [f"Research story {i}" for i in range(n)],
            "url": [f"https://domain{i % 5}.example/{i}" for i in range(n)],
            "by": [f"author{i % 10}" for i in range(n)],
            "time": (times.astype("int64") // 10**9).astype(int),
            "score": np.r_[rng.integers(1, 30, 50), rng.integers(30, 500, 50)],
            "descendants": rng.integers(0, 80, n),
        })

    def test_cohorts_filter_and_confidence_bounds(self):
        authors, domains = cohort_profiles(self.frame, min_stories=3)
        self.assertEqual(len(authors), 10)
        self.assertEqual(len(domains), 5)
        self.assertTrue((authors.top_decile_rate_ci95_low <= authors.top_decile_rate_ci95_high).all())
        self.assertIn("median_score", domains.columns)

    def test_drift_is_chronological_and_bounded_labels(self):
        report, meta = chronological_drift(self.frame)
        self.assertEqual(meta["ordering"], "created_at_utc")
        self.assertEqual(meta["reference_rows"] + meta["current_rows"], len(self.frame))
        self.assertTrue(set(report.drift_screen).issubset({"low", "moderate", "high", "insufficient_data"}))
        self.assertTrue(report.psi.notna().all())

    def test_export_writes_four_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v12_profile(self.frame, tmp, min_stories=2)
            for key in ("authors_file", "domains_file", "drift_file", "metadata_file"):
                self.assertTrue(Path(result[key]).exists())
            self.assertEqual(json.loads(Path(result["metadata_file"]).read_text())["version"], "12.0.0")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError): cohort_profiles(self.frame, min_stories=0)
        with self.assertRaises(ValueError): chronological_drift(self.frame, reference_fraction=.95)


if __name__ == "__main__":
    unittest.main()
