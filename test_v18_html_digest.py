"""v18.5 HTML digest tests: escaping, link safety, structure, CLI-level export."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_digest import build_digest, export_digest, markdown_to_html


class HtmlDigestTests(unittest.TestCase):
    def test_malicious_title_is_escaped(self):
        md = "1. [<img src=x onerror=alert(1)>](https://ok.example/?a=1&b=2) — 5 points"
        out = markdown_to_html(md)
        self.assertNotIn("<img", out)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", out)
        self.assertIn('href="https://ok.example/?a=1&amp;b=2"', out)

    def test_non_http_links_not_rendered_as_anchors(self):
        out = markdown_to_html("1. [click](javascript:alert(1)) — 3 points")
        self.assertNotIn("<a ", out)
        self.assertIn("javascript:alert(1)", out)  # shown as escaped text only

    def test_quote_in_url_cannot_break_attribute(self):
        out = markdown_to_html('1. [t](https://x.example/"onmouseover="a) — 1 points')
        self.assertNotIn('"onmouseover', out.split("<body>")[1].split("</a>")[0].replace("&quot;", ""))

    def test_structure_headings_lists_and_meta(self):
        stories = pd.DataFrame([{"title": "A", "url": "https://a.example", "score": 9, "by": "u", "domain": "a.example"}])
        out = markdown_to_html(build_digest(stories, None))
        self.assertIn("<h1>Hacker News digest</h1>", out)
        self.assertIn("<h2>Top stories by score</h2>", out)
        self.assertIn("<ol>", out)
        self.assertIn('<p class="meta">Generated', out)
        self.assertTrue(out.rstrip().endswith("</html>"))

    def test_export_html_writes_file_and_rejects_bad_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_digest(None, Path(tmp) / "none.sqlite", tmp, fmt="html")
            self.assertTrue(result["path"].endswith("digest.html"))
            self.assertIn("<!doctype html>", Path(result["path"]).read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            export_digest(None, None, "x", fmt="pdf")


if __name__ == "__main__":
    unittest.main()
