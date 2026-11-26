"""v18.16 compare tests."""
import json
import os
import subprocess
import sys
import unittest

import pandas as pd

from hn_cleaning import clean_stories
from hn_compare import compare_datasets, format_comparison

SAMPLE = "demo_stories.jsonl"


def make(rows):
    base = {"type": "story", "time": 1767225600, "by": "u"}
    return clean_stories(pd.DataFrame([{**base, "id": i, **r} for i, r in enumerate(rows, start=1)]))


BASE = make([
    {"title": "a", "score": 10, "descendants": 2, "url": "https://old.example/1"},
    {"title": "b", "score": 20, "descendants": 4, "url": "https://old.example/2"},
    {"title": "c", "score": 30, "descendants": 6, "url": "https://same.example/3"},
])
CAND = make([
    {"title": "d", "score": 100, "descendants": 40, "url": "https://new.example/4"},
    {"title": "e", "score": 200, "descendants": 60, "url": "https://new.example/5"},
    {"title": "f", "score": 300, "descendants": 80, "url": "https://same.example/6"},
    {"title": "g", "score": 400, "descendants": 100, "url": "https://new.example/7"},
])


class CompareTests(unittest.TestCase):
    def test_counts_and_median_changes(self):
        r = compare_datasets(BASE, CAND)
        self.assertEqual((r["baseline_stories"], r["candidate_stories"]), (3, 4))
        self.assertEqual(r["median_score"]["baseline"], 20)
        self.assertEqual(r["median_score"]["candidate"], 250)
        self.assertAlmostEqual(r["median_score"]["change"], 230)

    def test_largest_domain_shift_is_identified(self):
        r = compare_datasets(BASE, CAND, top_n=1)
        top = r["domain_shifts"][0]
        self.assertEqual(top["domain"], "new.example")
        self.assertGreater(top["share_change"], 0)

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            compare_datasets(BASE.iloc[0:0], CAND)

    def test_invalid_top_n(self):
        with self.assertRaises(ValueError):
            compare_datasets(BASE, CAND, top_n=0)

    def test_text_mentions_limitations(self):
        text = format_comparison(compare_datasets(BASE, CAND))
        self.assertIn("Median score: 20.0 -> 250.0", text)
        self.assertIn("no significance test", text)


class CompareCliTests(unittest.TestCase):
    def test_compare_sample_against_itself_has_zero_change(self):
        out = subprocess.run([sys.executable, "-m", "hn_cli", "compare", "--baseline", SAMPLE,
                              "--candidate", SAMPLE, "--format", "json"], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": "."}, check=True).stdout
        data = json.loads(out)
        self.assertEqual(data["median_score"]["change"], 0)
        self.assertTrue(all(abs(row["share_change"]) < 1e-9 for row in data["domain_shifts"]))

    def test_missing_file_clean_error(self):
        r = subprocess.run([sys.executable, "-m", "hn_cli", "compare", "--baseline", SAMPLE,
                            "--candidate", "nope.jsonl"], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": "."})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dataset not found", r.stderr)


if __name__ == "__main__":
    unittest.main()
