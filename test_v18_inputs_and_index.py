"""v18.8 tests: compressed inputs and filtered home page with escaped output."""
import gzip
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

from hn_data_loader import load_stories
from hn_server import make_server

SAMPLE = Path("demo_stories.jsonl")


class CompressedInputTests(unittest.TestCase):
    def test_gzip_jsonl_matches_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz = Path(tmp) / "s.jsonl.gz"
            with open(SAMPLE, "rb") as src, gzip.open(gz, "wb") as dst:
                dst.write(src.read())
            self.assertEqual(len(load_stories(gz)), len(load_stories(SAMPLE)))

    def test_gzip_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            gz = Path(tmp) / "s.csv.gz"
            with gzip.open(gz, "wt", encoding="utf-8") as f:
                f.write("id,title,score\n1,hi,5\n")
            self.assertEqual(list(load_stories(gz)["title"]), ["hi"])

    def test_unsupported_gz_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.txt.gz"
            p.write_bytes(gzip.compress(b"x"))
            with self.assertRaises(ValueError):
                load_stories(p)


class IndexFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0, str(SAMPLE), "none.sqlite")
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()

    def page(self, **params):
        url = self.base + "/?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8")

    def test_form_present_without_keyword(self):
        self.assertIn('name="keyword"', self.page())

    def test_keyword_results_rendered(self):
        self.assertIn("Matches for “rust”", self.page(keyword="rust"))

    def test_keyword_is_escaped(self):
        body = self.page(keyword="<script>alert(1)</script>")
        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertIn("&lt;script&gt;", body)

    def test_bad_min_score_message(self):
        self.assertIn("must be a number", self.page(keyword="rust", min_score="high"))


if __name__ == "__main__":
    unittest.main()
