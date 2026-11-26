"""v18.19 tests for the dashboard helper functions (no Streamlit needed)."""
import unittest

import pandas as pd

from hn_cleaning import clean_stories
from hn_dashboard_v18 import (compare_halves, digest_preview, search_stories,
                                        split_halves, summary_tables)
from hn_data_loader import load_stories

SAMPLE = "demo_stories.jsonl"
DAY = 86400
NEWEST = 1767225600


def frame():
    rows = [
        {"id": 1, "title": "Rust async runtime", "score": 300, "time": NEWEST, "by": "ann", "type": "story", "url": "https://a.example/1", "descendants": 40},
        {"id": 2, "title": "Python packaging", "score": 120, "time": NEWEST - 2 * DAY, "by": "bob", "type": "story", "url": "https://b.example/2", "descendants": 9},
        {"id": 3, "title": "Rust compiler news", "score": 50, "time": NEWEST - 20 * DAY, "by": "ann", "type": "story", "url": "https://a.example/3", "descendants": 0},
        {"id": 4, "title": "Go concurrency", "score": 80, "time": NEWEST - 25 * DAY, "by": "cy", "type": "story", "url": "", "descendants": 5},
    ]
    return clean_stories(pd.DataFrame(rows))


class SearchTests(unittest.TestCase):
    def test_keyword_and_min_score(self):
        out = search_stories(frame(), keyword="rust", min_score=100)
        self.assertEqual(list(out["id"]), [1])

    def test_days_window_applied(self):
        out = search_stories(frame(), days=5)
        self.assertEqual(sorted(out["id"]), [1, 2])

    def test_sort_by_comments(self):
        out = search_stories(frame(), sort="comments")
        self.assertEqual(list(out["id"])[0], 1)

    def test_blank_keyword_matches_everything(self):
        self.assertEqual(len(search_stories(frame(), keyword="   ")), 4)


class DigestPreviewTests(unittest.TestCase):
    def test_keyword_limits_digest(self):
        md = digest_preview(frame(), keyword="rust")
        self.assertIn("Rust async runtime", md)
        self.assertNotIn("Python packaging", md)

    def test_keyword_with_regex_characters_is_literal(self):
        md = digest_preview(frame(), keyword="(rust")
        self.assertIn("No story data available.", md)  # no crash, no match


class SummaryAndCompareTests(unittest.TestCase):
    def test_summary_tables_shapes(self):
        tables = summary_tables(frame(), top_n=2)
        self.assertEqual(tables["summary"]["stories"], 4)
        self.assertEqual(list(tables["authors"].columns), ["author", "stories"])
        self.assertLessEqual(len(tables["domains"]), 2)

    def test_split_halves_are_chronological(self):
        earlier, later = split_halves(frame())
        self.assertLess(earlier["created_at"].max(), later["created_at"].min())
        self.assertEqual(len(earlier) + len(later), 4)

    def test_split_needs_two_dated_stories(self):
        one = frame().iloc[:1]
        with self.assertRaises(ValueError):
            split_halves(one)

    def test_compare_halves_on_sample(self):
        result = compare_halves(clean_stories(load_stories(SAMPLE)))
        self.assertEqual(result["baseline_stories"] + result["candidate_stories"], 72)


if __name__ == "__main__":
    unittest.main()
