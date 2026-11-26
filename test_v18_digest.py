"""v18.2 digest tests: pure renderer, empty-data handling, and export round trip."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hn_digest import build_digest, export_digest
from hn_live import save_snapshot

NOW = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)


def stories():
    return pd.DataFrame([
        {"title": "Low", "url": "https://low.example/x", "score": 5, "by": "a", "domain": "low.example"},
        {"title": "Top", "url": "https://top.example/y", "score": 900, "by": "b", "domain": "top.example"},
        {"title": "Mid", "url": "javascript:alert(1)", "score": 300, "by": "c", "domain": ""},
        {"title": "NoScore", "url": "", "score": np.nan, "by": "d", "domain": ""},
    ])


class RendererTests(unittest.TestCase):
    def test_top_stories_sorted_and_linked(self):
        md = build_digest(stories(), None, top_n=2, generated_at=NOW)
        self.assertIn("1. [Top](https://top.example/y) — 900 points, by b (top.example)", md)
        self.assertIn("2. Mid — 300 points", md)  # non-http URL is not linked
        self.assertNotIn("NoScore", md)           # unscored rows excluded
        self.assertIn("2026-01-02 03:04 UTC", md)

    def test_rising_section_uses_only_rising_status(self):
        momentum = pd.DataFrame([
            {"title": "Rising", "url": "https://r.example", "score_gain": 40, "score_per_hour": 20.0,
             "elapsed_hours": 2.0, "momentum_status": "Rising"},
            {"title": "Cooling", "url": "", "score_gain": -3, "score_per_hour": -1.0,
             "elapsed_hours": 2.0, "momentum_status": "Cooling"},
        ])
        md = build_digest(None, momentum, generated_at=NOW)
        self.assertIn("[Rising](https://r.example) — +40 points (20.0/hour over 2.0 h)", md)
        self.assertNotIn("Cooling", md)

    def test_empty_inputs_render_messages(self):
        md = build_digest(None, None, generated_at=NOW)
        self.assertIn("No story data available.", md)
        self.assertIn("No snapshot history available", md)

    def test_invalid_top_n(self):
        with self.assertRaises(ValueError):
            build_digest(None, None, top_n=0)


class ExportTests(unittest.TestCase):
    def test_export_round_trip_with_db_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "s.jsonl"
            pd.DataFrame([{"id": 1, "title": "A", "score": 50, "time": 1700000000, "by": "x",
                           "type": "story", "url": "https://a.example", "descendants": 1}]
                         ).to_json(data, orient="records", lines=True)
            db = tmp / "h.sqlite"
            save_snapshot(db, pd.DataFrame([{"id": 1, "score": 10, "descendants": 0, "time": 1700000000,
                                             "title": "A", "url": "https://a.example", "by": "x"}]),
                          "2026-01-01T00:00:00Z")
            save_snapshot(db, pd.DataFrame([{"id": 1, "score": 30, "descendants": 2, "time": 1700000000,
                                             "title": "A", "url": "https://a.example", "by": "x"}]),
                          "2026-01-01T02:00:00Z")
            result = export_digest(data, db, tmp / "out")
            self.assertEqual(result["stories"], 1)
            self.assertEqual(result["momentum_rows"], 1)
            text = Path(result["path"]).read_text(encoding="utf-8")
            self.assertIn("[A](https://a.example) — 50 points", text)
            self.assertIn("+20 points", text)

    def test_export_with_no_sources_still_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_digest(None, Path(tmp) / "missing.sqlite", tmp)
            self.assertTrue(Path(result["path"]).exists())
            self.assertEqual(result["stories"], 0)


if __name__ == "__main__":
    unittest.main()
