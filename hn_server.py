"""v18.6 local web server: digest page and JSON API, standard library only.

Binds to 127.0.0.1 by default. There is no authentication, so do not expose it
on a public interface. Endpoints:
  GET /                      HTML digest of the sample dataset and optional history
  GET /api/query?keyword=&domain=&author=&min_score=&limit=
  GET /api/doctor            environment checks as JSON
"""
from __future__ import annotations

import html
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from hn_digest import build_digest, markdown_to_html
from hn_doctor import run_checks
from hn_query import run_query

MAX_API_LIMIT = 500


def make_handler(input_path: str, db_path: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, default=str).encode("utf-8"), "application/json")

        def do_GET(self):
            parsed = urlparse(self.path)
            params = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/":
                    self._index()
                elif parsed.path == "/api/query":
                    self._query(params)
                elif parsed.path == "/api/doctor":
                    results, failed = run_checks(db_path, input_path)
                    self._json(200, {"ok": not failed, "checks": [
                        {"name": n, "status": s, "detail": d} for n, s, d in results]})
                else:
                    self._json(404, {"error": "not found"})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})

        def _index(self):
            from hn_cleaning import clean_stories
            from hn_data_loader import load_stories
            from hn_intelligence import story_momentum
            from pathlib import Path
            stories = clean_stories(load_stories(input_path)) if Path(input_path).exists() else None
            momentum = story_momentum(db_path) if Path(db_path).exists() else None
            page = markdown_to_html(build_digest(stories, momentum))
            page += self._filter_section(self.path)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

        def _filter_section(self, raw_path):
            params = {k: v[-1] for k, v in parse_qs(urlparse(raw_path).query).items()}
            keyword = params.get("keyword", "").strip()
            if not keyword:
                return ('<h2>Filter stories</h2><form method="get" action="/">'
                        '<label>Keyword <input name="keyword"></label> '
                        '<label>Min score <input name="min_score" inputmode="decimal"></label> '
                        '<button type="submit">Search</button></form>')
            min_score = params.get("min_score", "").strip() or None
            try:
                result = run_query(input_path, keywords=[keyword],
                                   min_score=float(min_score) if min_score else None, limit=20)
            except ValueError:
                return "<p>Minimum score must be a number.</p>"
            rows = "".join(
                f"<li>{html.escape(str(r['title']) or '(untitled)')} — "
                f"{html.escape(str(int(r['score'])) if r['score'] == r['score'] else 'n/a')} points</li>"
                for r in result.to_dict(orient="records"))
            return (f"<h2>Matches for “{html.escape(keyword)}”</h2>"
                    + (f"<ol>{rows}</ol>" if rows else "<p>No matching stories.</p>"))

        def _query(self, params):
            try:
                limit = int(params.get("limit", 20))
            except ValueError:
                raise ValueError("limit must be an integer") from None
            if not 1 <= limit <= MAX_API_LIMIT:
                raise ValueError(f"limit must be between 1 and {MAX_API_LIMIT}")
            min_score = params.get("min_score")
            if min_score not in (None, ""):
                try:
                    min_score = float(min_score)
                except ValueError:
                    raise ValueError("min_score must be a number") from None
                if not math.isfinite(min_score):
                    raise ValueError("min_score must be a finite number")
            else:
                min_score = None
            result = run_query(
                input_path,
                keywords=[k for k in params.get("keyword", "").split(",") if k.strip()],
                domain=params.get("domain") or None,
                author=params.get("author") or None,
                min_score=min_score,
                limit=limit)
            self._json(200, {"count": len(result), "results": result.to_dict(orient="records")})

        def log_message(self, fmt, *args):  # keep test and CLI output quiet
            pass

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8000, input_path: str = "demo_stories.jsonl",
                db_path: str = "hn_monitor.sqlite") -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("serve binds to loopback only; use a reverse proxy for remote access")
    return ThreadingHTTPServer((host, port), make_handler(input_path, db_path))
