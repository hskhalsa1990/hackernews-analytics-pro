import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from hn_temporal_v9 import build_topic_evolution, forecast_snapshot_engagement, backtest_snapshot_trends, export_v9_temporal

class V9TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_json(Path(__file__).parent / "demo_stories.jsonl", lines=True)

    def test_topic_evolution_groups_by_time(self):
        sample = pd.DataFrame({"title": ["Python data tools", "Python data science", "Rust systems"],
                               "time": [1704067200, 1704153600, 1706745600]})
        result = build_topic_evolution(sample, freq="D", top_n=5)
        self.assertEqual(result["documents"], 3)
        self.assertIn("python", result["terms"])
        self.assertGreaterEqual(len(result["periods"]), 2)

    def _make_db(self, path):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE story_snapshots (id INTEGER, observed_at TEXT, score INTEGER, descendants INTEGER, title TEXT, url TEXT, by_author TEXT, created_at INTEGER, rank INTEGER, PRIMARY KEY(id, observed_at))")
            for story_id in range(3):
                for step in range(5):
                    conn.execute("INSERT INTO story_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                                 (story_id, (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=step)).isoformat(),
                                  10 + story_id * 3 + step * (story_id + 1), 2 + step, f"Story {story_id}", "", "author", 1700000000, 10-step))

    def test_forecast_and_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "snapshots.sqlite"
            self._make_db(db)
            forecast = forecast_snapshot_engagement(db, horizon_hours=2)
            self.assertEqual(len(forecast), 3)
            self.assertIn("score_projected", forecast)
            metrics = backtest_snapshot_trends(db)
            self.assertEqual(metrics["test_cases"], 3)
            self.assertIn("mae", metrics["linear_trend"])

    def test_export_topic_evolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v9_temporal(self.data, tmp)
            self.assertEqual(result["version"], "9.0.0")
            self.assertTrue(Path(result["artifacts"]["topic_evolution"]).exists())
            payload = json.loads(Path(result["artifacts"]["topic_evolution"]).read_text())
            self.assertTrue(payload["periods"])

    def test_invalid_frequency_column(self):
        with self.assertRaises(ValueError):
            build_topic_evolution(pd.DataFrame({"title": ["hello world"]}))

if __name__ == "__main__":
    unittest.main()
