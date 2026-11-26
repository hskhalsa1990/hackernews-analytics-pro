"""v18.17 tests: query --days relative window anchored to the newest story."""
import os
import subprocess
import sys
import unittest

import pandas as pd

from hn_cleaning import clean_stories
from hn_query import filter_stories

DAY = 86400
NEWEST = 1767225600  # 2026-01-01 00:00 UTC


def data():
    rows = [
        {"id": 1, "title": "newest", "score": 1, "time": NEWEST, "by": "a", "type": "story", "url": "", "descendants": 0},
        {"id": 2, "title": "three days", "score": 2, "time": NEWEST - 3 * DAY, "by": "a", "type": "story", "url": "", "descendants": 0},
        {"id": 3, "title": "ten days", "score": 3, "time": NEWEST - 10 * DAY, "by": "a", "type": "story", "url": "", "descendants": 0},
    ]
    return clean_stories(pd.DataFrame(rows))


class DaysTests(unittest.TestCase):
    def test_window_keeps_recent_stories(self):
        self.assertEqual(sorted(filter_stories(data(), days=7)["id"]), [1, 2])

    def test_window_is_anchored_to_newest_story_not_clock(self):
        # Shifting every timestamp back 1000 days must not change which stories fall in the window.
        shifted = data().copy()
        shifted["created_at"] = shifted["created_at"] - pd.Timedelta(days=1000)
        self.assertEqual(sorted(filter_stories(shifted, days=7)["id"]), [1, 2])

    def test_non_positive_days_rejected(self):
        with self.assertRaises(ValueError):
            filter_stories(data(), days=0)

    def test_empty_timestamps_return_empty(self):
        df = clean_stories(pd.DataFrame([{"id": 9, "title": "x", "score": 1, "by": "a", "type": "story", "url": ""}]))
        self.assertTrue(filter_stories(df, days=7).empty)

    def test_cli_days_flag_narrows_results(self):
        def count(*extra):
            out = subprocess.run([sys.executable, "-m", "hn_cli", "query", "--input",
                                  "demo_stories.jsonl", "--count", *extra],
                                 capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."}, check=True).stdout
            return int(out.strip())
        total = count()
        windowed = count("--days", "3")
        self.assertGreater(windowed, 0)
        self.assertLess(windowed, total)


if __name__ == "__main__":
    unittest.main()
