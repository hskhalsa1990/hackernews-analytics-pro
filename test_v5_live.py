import tempfile
import unittest
from pathlib import Path
import pandas as pd
from hn_live import save_snapshot, growth_alerts

class LiveMonitorTests(unittest.TestCase):
    def test_snapshots_detect_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "monitor.sqlite"
            first = pd.DataFrame([{"id": 123, "score": 10, "descendants": 2, "title": "Test story", "url": "", "by": "alice", "time": 1700000000}])
            second = pd.DataFrame([{"id": 123, "score": 45, "descendants": 18, "title": "Test story", "url": "", "by": "alice", "time": 1700000000}])
            save_snapshot(db, first, "2026-01-01T00:00:00Z")
            save_snapshot(db, second, "2026-01-01T01:00:00Z")
            alerts = growth_alerts(db, min_score_gain=20, min_comment_gain=10)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(int(alerts.iloc[0].score_gain), 35)
            self.assertEqual(int(alerts.iloc[0].comment_gain), 16)

    def test_empty_db_returns_empty_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(growth_alerts(Path(tmp)/"empty.sqlite").empty)

if __name__ == "__main__": unittest.main()
