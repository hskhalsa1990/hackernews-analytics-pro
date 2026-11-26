"""v18.11 tests: query sort options and digest 'Most discussed' section."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_cleaning import clean_stories
from hn_digest import build_digest
from hn_query import filter_stories


def data():
    raw = pd.DataFrame([
        {"id": 1, "title": "Alpha", "score": 10, "time": 1767225600, "by": "a", "type": "story", "url": "https://x.example/1", "descendants": 500},
        {"id": 2, "title": "Beta", "score": 900, "time": 1764547200, "by": "b", "type": "story", "url": "https://x.example/2", "descendants": 3},
        {"id": 3, "title": "Gamma", "score": 50, "time": 1767225600, "by": "c", "type": "story", "url": "https://x.example/3", "descendants": 0},
    ])
    return clean_stories(raw)


class SortTests(unittest.TestCase):
    def test_sort_by_score_default(self):
        self.assertEqual(list(filter_stories(data())["id"]), [2, 3, 1])

    def test_sort_by_comments(self):
        self.assertEqual(list(filter_stories(data(), sort="comments")["id"]), [1, 2, 3])

    def test_sort_by_time_newest_first(self):
        self.assertEqual(list(filter_stories(data(), sort="time")["id"])[0], 1)

    def test_invalid_sort_rejected(self):
        with self.assertRaises(ValueError):
            filter_stories(data(), sort="karma")


class DigestDiscussedTests(unittest.TestCase):
    def test_most_discussed_section(self):
        md = build_digest(data(), None, top_n=2)
        section = md.split("## Most discussed stories")[1].split("## Fastest")[0]
        self.assertIn("[Alpha](https://x.example/1) — 500 comments", section)
        self.assertNotIn("Gamma", section)  # zero comments excluded

    def test_missing_comment_column(self):
        md = build_digest(pd.DataFrame([{"title": "T", "score": 1, "url": ""}]), None)
        self.assertIn("No comment data available.", md)


class QueryCliTests(unittest.TestCase):
    def test_json_output(self):
        out = subprocess.run([sys.executable, "-m", "hn_cli", "query", "--input",
                              "demo_stories.jsonl", "--keyword", "rust", "--sort", "comments",
                              "--limit", "3", "--format", "json"], capture_output=True, text=True, check=True).stdout
        rows = json.loads(out)
        self.assertLessEqual(len(rows), 3)
        self.assertTrue(all("rust" in r["title"].lower() for r in rows))
        self.assertTrue(all(isinstance(r["score"], (int, float)) for r in rows))  # numbers stay numeric


if __name__ == "__main__":
    unittest.main()
