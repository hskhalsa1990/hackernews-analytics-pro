"""v18.12 tests: watch JSON/CSV output and doctor --json."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_live import save_snapshot


def run(args):
    return subprocess.run([sys.executable, "-m", "hn_cli", *args],
                          capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "."})


class WatchOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "h.sqlite"
        save_snapshot(self.db, pd.DataFrame([{"id": 1, "score": 40, "descendants": 3, "time": 1700000000,
                                              "title": "Rust release", "url": "https://r.example", "by": "x"}]),
                      "2026-01-01T00:00:00Z")

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_output_keeps_numbers(self):
        out = run(["watch", "--db", str(self.db), "--keyword", "rust", "--format", "json"])
        self.assertEqual(out.returncode, 0, out.stderr)
        rows = json.loads(out.stdout)
        self.assertEqual(rows[0]["score"], 40)
        self.assertIsInstance(rows[0]["comments"], int)

    def test_csv_output(self):
        target = Path(self.tmp.name) / "out" / "w.csv"
        out = run(["watch", "--db", str(self.db), "--keyword", "rust", "--output", str(target)])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(len(pd.read_csv(target)), 1)


class DoctorJsonTests(unittest.TestCase):
    def test_doctor_json_is_valid(self):
        out = run(["doctor", "--json", "--db", "none.sqlite"])
        payload = json.loads(out.stdout)
        self.assertIn("ok", payload)
        self.assertTrue(any(c["name"] == "python" for c in payload["checks"]))


if __name__ == "__main__":
    unittest.main()
