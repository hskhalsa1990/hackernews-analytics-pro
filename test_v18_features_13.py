"""v18.13 tests: query --exclude-keyword and --count."""
import json
import os
import subprocess
import sys
import unittest

import pandas as pd

from hn_cleaning import clean_stories
from hn_query import filter_stories

SAMPLE = "demo_stories.jsonl"


def cli(*args):
    return subprocess.run([sys.executable, "-m", "hn_cli", *args], capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": "."}, check=True).stdout


def data():
    return clean_stories(pd.DataFrame([
        {"id": 1, "title": "Rust async tips", "score": 10, "time": 1767225600, "by": "a", "type": "story", "url": "", "descendants": 0},
        {"id": 2, "title": "Rust and Python interop", "score": 20, "time": 1767225600, "by": "b", "type": "story", "url": "", "descendants": 0},
        {"id": 3, "title": "Go concurrency", "score": 30, "time": 1767225600, "by": "c", "type": "story", "url": "", "descendants": 0},
    ]))


class ExcludeTests(unittest.TestCase):
    def test_exclude_removes_matching_titles(self):
        out = filter_stories(data(), exclude=["python"])
        self.assertEqual(sorted(out["id"]), [1, 3])

    def test_exclude_combined_with_keyword(self):
        out = filter_stories(data(), keywords=["rust"], exclude=["python"])
        self.assertEqual(list(out["id"]), [1])

    def test_exclude_is_literal(self):
        out = filter_stories(data(), exclude=["c++"])
        self.assertEqual(len(out), 3)


class CountCliTests(unittest.TestCase):
    def test_count_ignores_default_limit(self):
        total = int(cli("query", "--input", SAMPLE, "--keyword", "python", "--count").strip())
        self.assertGreater(total, 0)
        self.assertEqual(int(cli("query", "--input", SAMPLE, "--keyword", "python", "--count", "--limit", "1").strip()), total)

    def test_exclude_flag_reduces_count(self):
        all_rust = int(cli("query", "--input", SAMPLE, "--keyword", "rust", "--count").strip())
        excluded = int(cli("query", "--input", SAMPLE, "--keyword", "rust", "--exclude-keyword", "patterns", "--count").strip())
        self.assertLess(excluded, all_rust)


if __name__ == "__main__":
    unittest.main()
