"""v18.9 tests: holdout label-permutation test for virality ROC-AUC."""
import unittest

import numpy as np
import pandas as pd

from hn_predictive import _permutation_auc, train_virality_model


class PermutationAucTests(unittest.TestCase):
    def test_perfect_signal_has_small_p_value(self):
        y = np.array([0] * 20 + [1] * 20)
        scores = y + np.random.default_rng(0).normal(0, 0.01, len(y))
        auc, p, null95 = _permutation_auc(y, scores, permutations=200, seed=1)
        self.assertAlmostEqual(auc, 1.0, places=3)
        self.assertLessEqual(p, 1 / 201 + 1e-12)
        self.assertLess(null95, 0.8)

    def test_random_scores_have_large_p_value(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 200)
        scores = rng.random(200)
        _, p, _ = _permutation_auc(y, scores, permutations=200, seed=2)
        self.assertGreater(p, 0.05)

    def test_deterministic_with_seed(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        s = np.array([0.1, 0.8, 0.4, 0.6, 0.7, 0.2, 0.9, 0.3])
        self.assertEqual(_permutation_auc(y, s, 50, 9), _permutation_auc(y, s, 50, 9))

    def test_invalid_permutations(self):
        with self.assertRaises(ValueError):
            _permutation_auc(np.array([0, 1]), np.array([0.1, 0.9]), 0, 1)

    def test_training_reports_p_value(self):
        rng = np.random.default_rng(3)
        n = 120
        titles = ["rust tooling release" if i % 2 else "weekly notes" for i in range(n)]
        scores = np.array([300 if i % 2 else 10 for i in range(n)]) + rng.integers(0, 5, n)
        df = pd.DataFrame({
            "id": range(n), "title": titles, "score": scores, "time": 1700000000 + np.arange(n) * 3600,
            "by": ["a"] * n, "type": ["story"] * n, "domain": ["x.example"] * n,
            "url": [""] * n, "hour_utc": np.arange(n) % 24, "weekday_utc": ["Mon"] * n,
            "title_word_count": [3] * n, "title_char_count": [len(t) for t in titles],
        })
        _, metrics, _ = train_virality_model(df, permutations=50)
        self.assertIn("roc_auc_permutation_p_value", metrics)
        self.assertLessEqual(metrics["roc_auc_permutation_p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
