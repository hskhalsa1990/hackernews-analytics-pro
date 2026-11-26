import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hn_walkforward_v11 import walk_forward_benchmark, export_v11_walkforward


class V11WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        n = 120
        times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        # A mixed, non-monotonic outcome distribution avoids a trivial one-class
        # target while preserving a chronological ordering for evaluation.
        scores = rng.integers(1, 500, size=n) + (np.arange(n) % 9) * 7
        cls.frame = pd.DataFrame({
            "title": [f"{'Python machine learning' if i % 3 else 'Rust systems research'} story {i}" for i in range(n)],
            "url": [f"https://domain{i % 8}.example/post/{i}" for i in range(n)],
            "type": ["story"] * n, "by": [f"author{i % 15}" for i in range(n)],
            "time": (times.astype("int64") // 10**9).astype(int),
            "score": scores, "descendants": rng.integers(0, 150, size=n),
        })

    def test_walk_forward_has_three_models_and_disjoint_folds(self):
        summary, folds, predictions, metadata = walk_forward_benchmark(
            self.frame, n_splits=3, test_fraction=0.15, bootstrap_repeats=60)
        self.assertEqual(set(summary.model), {"logistic_regression", "random_forest", "extra_trees"})
        self.assertEqual(len(folds), 9)
        self.assertEqual(folds.groupby("model").fold.nunique().min(), 3)
        self.assertEqual(predictions.groupby("model").size().nunique(), 1)
        self.assertEqual(metadata["version"], "11.0.0")
        self.assertNotIn("score", metadata["predictors"])
        self.assertTrue((folds.train_rows > folds.test_rows).all())

    def test_calibration_metrics_are_bounded(self):
        summary, folds, _, _ = walk_forward_benchmark(
            self.frame, n_splits=2, test_fraction=0.15, bootstrap_repeats=0)
        for column in ["brier_score_mean", "expected_calibration_error_mean"]:
            self.assertTrue(summary[column].between(0, 1).all())
        self.assertTrue((folds.log_loss >= 0).all())

    def test_export_writes_four_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v11_walkforward(self.frame, tmp, n_splits=2,
                                            test_fraction=0.15, bootstrap_repeats=30)
            for key in ["summary_file", "folds_file", "predictions_file", "metadata_file"]:
                self.assertTrue(Path(result[key]).exists())
            metadata = json.loads(Path(result["metadata_file"]).read_text())
            self.assertEqual(metadata["version"], "11.0.0")

    def test_rejects_invalid_fold_count(self):
        with self.assertRaises(ValueError):
            walk_forward_benchmark(self.frame, n_splits=1)


if __name__ == "__main__":
    unittest.main()
