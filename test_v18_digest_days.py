"""v18.18 tests: digest --days window shares the query window logic."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_cleaning import clean_stories
from hn_digest import export_digest
from hn_query import window_last_days

DAY = 86400
NEWEST = 1767225600


def frame():
    return clean_stories(pd.DataFrame([
        {"id": 1, "title": "Fresh story", "score": 5, "time": NEWEST, "by": "a", "type": "story", "url": "https://f.example/1", "descendants": 1},
        {"id": 2, "title": "Old story", "score": 999, "time": NEWEST - 30 * DAY, "by": "b", "type": "story", "url": "https://o.example/2", "descendants": 1},
    ]))


class WindowTests(unittest.TestCase):
    def test_window_keeps_recent_rows(self):
        self.assertEqual(list(window_last_days(frame(), 7)["id"]), [1])

    def test_window_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            window_last_days(frame(), 0)


class DigestDaysTests(unittest.TestCase):
    def test_digest_excludes_stories_outside_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "s.jsonl"
            frame().to_json(data, orient="records", lines=True)
            result = export_digest(data, Path(tmp) / "none.sqlite", tmp, days=7)
            text = Path(result["path"]).read_text(encoding="utf-8")
            self.assertIn("Fresh story", text)
            self.assertNotIn("Old story", text)  # 999 points, but outside the window
            self.assertEqual(result["stories"], 1)


if __name__ == "__main__":
    unittest.main()
