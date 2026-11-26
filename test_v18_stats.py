"""v18.15 stats tests."""
import json
import os
import subprocess
import sys
import unittest

import pandas as pd

from hn_cleaning import clean_stories
from hn_stats import format_text, summarize

SAMPLE = "demo_stories.jsonl"


def frame():
    return clean_stories(pd.DataFrame([
        {"id": 1, "title": "A", "score": 10, "time": 1767225600, "by": "ann", "type": "story", "url": "https://x.example/1", "descendants": 4},
        {"id": 2, "title": "B", "score": 30, "time": 1767312000, "by": "ann", "type": "story", "url": "https://x.example/2", "descendants": 0},
        {"id": 3, "title": "C", "score": None, "time": 1767398400, "by": "bob", "type": "story", "url": "", "descendants": None},
    ]))


class SummaryTests(unittest.TestCase):
    def test_counts_and_quantiles(self):
        s = summarize(frame())
        self.assertEqual(s["stories"], 3)
        self.assertEqual(s["scored_stories"], 2)
        self.assertEqual(s["score"]["min"], 10)
        self.assertEqual(s["score"]["max"], 30)
        self.assertEqual(s["score"]["median"], 20)

    def test_top_authors_and_domains(self):
        s = summarize(frame())
        self.assertEqual(list(s["top_authors"].items())[0], ("ann", 2))
        self.assertEqual(s["top_domains"], {"x.example": 2})  # empty domain excluded

    def test_empty_comment_column_gives_none(self):
        df = clean_stories(pd.DataFrame([{"id": 9, "title": "Z", "score": 1, "time": 1767225600, "by": "q", "type": "story", "url": ""}]))
        summary = summarize(df)
        self.assertIsNone(summary["comments"])
        self.assertEqual(summary["scored_stories"], 1)

    def test_invalid_top_n(self):
        with self.assertRaises(ValueError):
            summarize(frame(), top_n=0)

    def test_text_format_mentions_counts(self):
        text = format_text(summarize(frame()))
        self.assertIn("Stories: 3 (2 with a score)", text)


class StatsCliTests(unittest.TestCase):
    def test_json_output(self):
        out = subprocess.run([sys.executable, "-m", "hn_cli", "stats", "--input", SAMPLE, "--format", "json"],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."}, check=True).stdout
        data = json.loads(out)
        self.assertEqual(data["stories"], 72)

    def test_missing_input_is_clean_error(self):
        r = subprocess.run([sys.executable, "-m", "hn_cli", "stats", "--input", "nope.jsonl"],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dataset not found", r.stderr)


if __name__ == "__main__":
    unittest.main()
