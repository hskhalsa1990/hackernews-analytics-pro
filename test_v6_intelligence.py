import tempfile
import unittest
from pathlib import Path
import pandas as pd
from hn_live import save_snapshot
from hn_intelligence import story_momentum, export_momentum

class StoryIntelligenceTests(unittest.TestCase):
    def test_rank_and_velocity(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/"hn.sqlite"
            a=pd.DataFrame([{"id":1,"score":10,"descendants":2,"title":"Story","url":"https://example.com","by":"alice"},
                            {"id":2,"score":20,"descendants":4,"title":"Other","url":"","by":"bob"}])
            b=pd.DataFrame([{"id":1,"score":30,"descendants":8,"title":"Story","url":"https://example.com","by":"alice"},
                            {"id":2,"score":18,"descendants":4,"title":"Other","url":"","by":"bob"}])
            save_snapshot(db,a,"2026-01-01T00:00:00Z")
            save_snapshot(db,b,"2026-01-01T02:00:00Z")
            out=story_momentum(db,24).set_index("id")
            self.assertEqual(int(out.loc[1,"rank_improvement"]),0)
            self.assertAlmostEqual(float(out.loc[1,"score_per_hour"]),10.0)
            self.assertEqual(out.loc[1,"momentum_status"],"Rising")
            self.assertEqual(out.loc[2,"momentum_status"],"Cooling")
    def test_legacy_schema_migrates(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/"legacy.sqlite"
            with sqlite3.connect(db) as c:
                c.execute("CREATE TABLE story_snapshots (id INTEGER NOT NULL, observed_at TEXT NOT NULL, score INTEGER, descendants INTEGER, title TEXT, url TEXT, by_author TEXT, created_at INTEGER, PRIMARY KEY(id, observed_at))")
            save_snapshot(db,pd.DataFrame([{"id":3,"score":1,"title":"Legacy"}]),"2026-01-01T00:00:00Z")
            with sqlite3.connect(db) as c:
                cols=[r[1] for r in c.execute("PRAGMA table_info(story_snapshots)")]
            self.assertIn("rank",cols)
    def test_export_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=export_momentum(Path(tmp)/"missing.sqlite",tmp)
            self.assertEqual(result["rows"],0)
            self.assertTrue(Path(result["csv"]).exists())

if __name__=="__main__": unittest.main()
