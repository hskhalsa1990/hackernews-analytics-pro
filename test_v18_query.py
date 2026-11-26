"""v18.3 query and watch tests."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_cleaning import clean_stories
from hn_live import save_snapshot
from hn_query import filter_stories, run_query, watch_history


def sample():
    raw = pd.DataFrame([
        {"id": 1, "title": "Rust async runtime", "score": 300, "time": 1767225600, "by": "ann", "type": "story", "url": "https://www.github.com/a", "descendants": 5},
        {"id": 2, "title": "Python packaging news", "score": 120, "time": 1767225600, "by": "bob", "type": "story", "url": "https://docs.python.org/b", "descendants": 9},
        {"id": 3, "title": "RUST compiler update (c++ rival?)", "score": 50, "time": 1764547200, "by": "ann", "type": "story", "url": "https://github.com/c", "descendants": 1},
        {"id": 4, "title": "Unrelated post", "score": 999, "time": 1767225600, "by": "cat", "type": "story", "url": "", "descendants": 0},
    ])
    return clean_stories(raw)


class FilterTests(unittest.TestCase):
    def test_keyword_is_case_insensitive_and_literal(self):
        out = filter_stories(sample(), keywords=["rust"])
        self.assertEqual(list(out["id"]), [1, 3])
        literal = filter_stories(sample(), keywords=["c++"])
        self.assertEqual(list(literal["id"]), [3])  # '+' is not a regex operator

    def test_all_keywords_must_match(self):
        self.assertEqual(list(filter_stories(sample(), keywords=["rust", "async"])["id"]), [1])

    def test_domain_and_author_filters(self):
        self.assertEqual(list(filter_stories(sample(), domain="www.github.com")["id"]), [1, 3])
        self.assertEqual(list(filter_stories(sample(), author="ANN")["id"]), [1, 3])

    def test_min_score_and_since(self):
        self.assertEqual(list(filter_stories(sample(), min_score=100)["id"]), [4, 1, 2])
        self.assertEqual(list(filter_stories(sample(), since="2026-01-01")["id"]), [4, 1, 2])

    def test_limit_and_sort(self):
        out = filter_stories(sample(), limit=2)
        self.assertEqual(list(out["id"]), [4, 1])
        with self.assertRaises(ValueError):
            filter_stories(sample(), limit=0)

    def test_run_query_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            run_query("/nonexistent/x.jsonl")


class WatchTests(unittest.TestCase):
    def test_watch_matches_latest_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "h.sqlite"
            save_snapshot(db, pd.DataFrame([{"id": 1, "score": 10, "descendants": 0, "time": 1700000000,
                                             "title": "Rust release", "url": "", "by": "x"}]), "2026-01-01T00:00:00Z")
            save_snapshot(db, pd.DataFrame([{"id": 1, "score": 40, "descendants": 3, "time": 1700000000,
                                             "title": "Rust release", "url": "", "by": "x"},
                                            {"id": 2, "score": 99, "descendants": 0, "time": 1700000000,
                                             "title": "Go release", "url": "", "by": "y"}]), "2026-01-01T01:00:00Z")
            out = watch_history(db, ["rust"], lookback_hours=24)
            self.assertEqual(list(out["id"]), [1])
            self.assertEqual(int(out.iloc[0]["score"]), 40)
            self.assertEqual(int(out.iloc[0]["comments"]), 3)

    def test_watch_missing_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(watch_history(Path(tmp) / "none.sqlite", ["x"]).empty)


if __name__ == "__main__":
    unittest.main()
