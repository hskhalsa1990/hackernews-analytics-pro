import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from hn_benchmark_v10 import benchmark_virality_models, export_v10_benchmark

class V10BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        n = 120
        times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        cls.frame = pd.DataFrame({
            "title": [f"{'Python machine learning' if i % 3 else 'Rust systems research'} release story number {i}" for i in range(n)],
            "url": [f"https://domain{i % 7}.example/post/{i}" for i in range(n)],
            "type": ["story"] * n, "by": [f"author{i % 13}" for i in range(n)],
            "time": (times.astype("int64") // 10**9).astype(int),
            "score": (rng.integers(1, 400, size=n) + np.arange(n) * 2).astype(int),
            "descendants": rng.integers(0, 100, size=n),
        })

    def test_benchmark_same_holdout_for_three_models(self):
        board, predictions, meta = benchmark_virality_models(self.frame)
        self.assertEqual(set(board.model), {"logistic_regression", "random_forest", "extra_trees"})
        self.assertEqual(board.test_rows.nunique(), 1)
        self.assertEqual(predictions.groupby("model").size().nunique(), 1)
        self.assertEqual(meta["version"], "10.0.0")
        self.assertNotIn("score", meta["predictors"])

    def test_export_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v10_benchmark(self.frame, tmp)
            self.assertTrue(Path(result["leaderboard_file"]).exists())
            self.assertTrue(Path(result["predictions_file"]).exists())
            self.assertTrue(Path(result["summary_file"]).exists())

    def test_rejects_invalid_split(self):
        with self.assertRaises(ValueError):
            benchmark_virality_models(self.frame, test_fraction=0.8)

if __name__ == "__main__":
    unittest.main()
