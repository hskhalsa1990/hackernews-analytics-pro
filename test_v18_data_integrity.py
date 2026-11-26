"""v18 regression tests: missing-value handling and id-safe deduplication."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hn_cleaning import clean_stories, text_or_empty
from hn_live import save_snapshot


class DeduplicationTests(unittest.TestCase):
    def test_rows_without_id_are_not_collapsed(self):
        raw = pd.DataFrame([
            {"id": 1, "title": "a", "score": 10, "time": 1700000000, "by": "x", "type": "story", "url": "https://a.com", "descendants": 1},
            {"title": "b", "score": 5, "time": 1700000001, "by": "y", "type": "story", "url": "https://b.com", "descendants": 0},
            {"title": "c", "score": 7, "time": 1700000002, "by": "z", "type": "story", "url": "https://c.com", "descendants": 0},
        ])
        out = clean_stories(raw)
        self.assertEqual(sorted(out["title"]), ["a", "b", "c"])

    def test_duplicate_ids_still_removed(self):
        base = {"id": 7, "title": "dup", "score": 1, "time": 1700000000, "by": "x", "type": "story", "url": "", "descendants": 0}
        out = clean_stories(pd.DataFrame([base, dict(base)]))
        self.assertEqual(len(out), 1)


class MissingValueTests(unittest.TestCase):
    def test_text_or_empty(self):
        self.assertEqual(text_or_empty(np.nan), "")
        self.assertEqual(text_or_empty(None), "")
        self.assertEqual(text_or_empty(pd.NA), "")
        self.assertEqual(text_or_empty("hello"), "hello")
        self.assertEqual(text_or_empty(0), "0")

    def test_snapshot_never_stores_literal_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            df = pd.DataFrame([{"id": 5, "score": 3, "descendants": np.nan, "time": 1700000000,
                                "title": np.nan, "url": np.nan, "by": np.nan}])
            save_snapshot(db, df, "2026-01-01T00:00:00Z")
            row = sqlite3.connect(db).execute("SELECT title, url, by_author FROM story_snapshots").fetchone()
            self.assertEqual(row, ("", "", ""))


if __name__ == "__main__":
    unittest.main()
