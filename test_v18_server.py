"""v18.6 local server tests: real HTTP requests against an ephemeral loopback port."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from hn_server import make_server

SAMPLE = "demo_stories.jsonl"


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0, SAMPLE, "none.sqlite")
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.headers, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read().decode("utf-8")

    def test_index_serves_html_with_security_headers(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("Top stories by score", body)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])

    def test_query_api(self):
        status, _, body = self.get("/api/query?keyword=rust&min_score=100&limit=2")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertLessEqual(data["count"], 2)
        self.assertTrue(all("rust" in r["title"].lower() for r in data["results"]))

    def test_query_api_validation(self):
        status, _, body = self.get("/api/query?limit=0")
        self.assertEqual(status, 400)
        self.assertIn("limit", json.loads(body)["error"])

    def test_query_api_rejects_unbounded_and_non_finite_input(self):
        for path, fragment in [("/api/query?limit=100000", "limit must be between"),
                               ("/api/query?limit=abc", "limit must be an integer"),
                               ("/api/query?min_score=nan", "finite"),
                               ("/api/query?min_score=inf", "finite"),
                               ("/api/query?min_score=high", "must be a number")]:
            status, _, body = self.get(path)
            self.assertEqual(status, 400, path)
            self.assertIn(fragment, json.loads(body)["error"])

    def test_doctor_api_and_unknown_route(self):
        status, _, body = self.get("/api/doctor")
        self.assertEqual(status, 200)
        self.assertIn("checks", json.loads(body))
        status, _, _ = self.get("/nope")
        self.assertEqual(status, 404)

    def test_refuses_non_loopback_bind(self):
        with self.assertRaises(ValueError):
            make_server("0.0.0.0", 0, SAMPLE, "none.sqlite")


if __name__ == "__main__":
    unittest.main()
