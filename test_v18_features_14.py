"""v18.14 tests: digest keyword filter and query CSV to stdout."""
import os
import subprocess
import sys
import unittest

import pandas as pd

import tempfile
from pathlib import Path

from hn_digest import export_digest

SAMPLE = "demo_stories.jsonl"


def cli(*args):
    return subprocess.run([sys.executable, "-m", "hn_cli", *args], capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": "."}, check=True).stdout


class DigestKeywordTests(unittest.TestCase):
    def test_keyword_limits_digest_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_digest(SAMPLE, Path(tmp) / "none.sqlite", tmp, keywords=["rust"])
            text = Path(result["path"]).read_text(encoding="utf-8")
            self.assertIn("Rust", text)
            self.assertNotIn("New release: faster Python", text)
            self.assertGreater(result["stories"], 0)
            self.assertLess(result["stories"], 72)

    def test_no_matches_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_digest(SAMPLE, Path(tmp) / "none.sqlite", tmp, keywords=["zzz-no-such-term"])
            self.assertEqual(result["stories"], 0)
            self.assertIn("No story data available.", Path(result["path"]).read_text(encoding="utf-8"))


class QueryCsvTests(unittest.TestCase):
    def test_csv_to_stdout_parses(self):
        out = cli("query", "--input", SAMPLE, "--keyword", "rust", "--limit", "3", "--format", "csv")
        import io
        frame = pd.read_csv(io.StringIO(out))
        self.assertLessEqual(len(frame), 3)
        self.assertIn("title", frame.columns)
        self.assertTrue(frame["title"].str.lower().str.contains("rust").all())


if __name__ == "__main__":
    unittest.main()
