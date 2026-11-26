"""Live Hacker News ingestion and SQLite-backed story growth monitoring.

Uses the public Hacker News Firebase API. Polling is intentionally bounded;
respect the service and avoid aggressive request intervals.
"""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from typing import Iterable
import pandas as pd
import requests

from hn_cleaning import text_or_empty

BASE = "https://hacker-news.firebaseio.com/v0"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class HackerNewsClient:
    """Bounded, retrying client for the public HN Firebase API.

    Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried with
    exponential backoff. A story that still fails is skipped and recorded in
    ``self.failures`` instead of aborting the whole feed.
    """

    def __init__(self, timeout: int = 15, session=None, sleep=time.sleep,
                 max_retries: int = 3, backoff: float = 0.5):
        if max_retries < 0 or backoff < 0:
            raise ValueError("max_retries and backoff must be non-negative")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff = backoff
        self.failures: list[tuple[str, str]] = []

    def _get(self, endpoint: str):
        url = f"{BASE}/{endpoint}.json"
        attempt = 0
        while True:
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                retryable = not isinstance(exc, requests.HTTPError) or (
                    exc.response is not None and exc.response.status_code in RETRYABLE_STATUS)
                if not retryable or attempt >= self.max_retries:
                    raise
                self.sleep(self.backoff * (2 ** attempt))
                attempt += 1

    def fetch_feed(self, feed: str = "top", limit: int = 100, pause: float = 0.05,
                   max_workers: int = 4) -> pd.DataFrame:
        if feed not in {"top", "new", "best", "ask", "show", "job"}:
            raise ValueError("feed must be top, new, best, ask, show or job")
        if not 1 <= int(limit) <= 500:
            raise ValueError("limit must be between 1 and 500")
        if not 1 <= int(max_workers) <= 16:
            raise ValueError("max_workers must be between 1 and 16")
        self.failures = []
        ids = self._get(f"{feed}stories") or []
        selected = list(ids[:int(limit)])

        def fetch_item(story_id):
            if pause:
                self.sleep(pause)
            try:
                return story_id, self._get(f"item/{story_id}"), None
            except Exception as exc:  # isolate per-item failure
                return story_id, None, f"{type(exc).__name__}: {exc}"

        # ThreadPoolExecutor.map preserves input order, so output order matches the feed.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            results = list(pool.map(fetch_item, selected))

        rows = []
        for story_id, item, error in results:
            if error is not None:
                self.failures.append((str(story_id), error))
                continue
            if item and item.get("type") in {"story", "job"}:
                rows.append(item)
        return pd.DataFrame(rows)


def save_snapshot(db_path, stories: pd.DataFrame, observed_at=None) -> int:
    """Persist observations; repeated observations of an item are retained."""
    if stories.empty:
        return 0
    observed = pd.Timestamp(observed_at or pd.Timestamp.now(tz="UTC"))
    if observed.tzinfo is None:
        observed = observed.tz_localize("UTC")
    observed = observed.tz_convert("UTC").isoformat()
    path = Path(db_path); path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS story_snapshots (
            id INTEGER NOT NULL, observed_at TEXT NOT NULL, score INTEGER,
            descendants INTEGER, title TEXT, url TEXT, by_author TEXT, created_at INTEGER,
            rank INTEGER,
            PRIMARY KEY (id, observed_at))""")
        # Backward-compatible migration for databases created by v5.0.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(story_snapshots)")}
        if "rank" not in existing:
            conn.execute("ALTER TABLE story_snapshots ADD COLUMN rank INTEGER")
        rows = []
        for position, (_, r) in enumerate(stories.iterrows(), start=1):
            if pd.isna(r.get("id")): continue
            rows.append((int(r["id"]), observed, _int_or_none(r.get("score")),
                         _int_or_none(r.get("descendants")), text_or_empty(r.get("title")),
                         text_or_empty(r.get("url")), text_or_empty(r.get("by")), _int_or_none(r.get("time")), position))
        conn.executemany("""INSERT OR REPLACE INTO story_snapshots
            (id, observed_at, score, descendants, title, url, by_author, created_at, rank)
            VALUES (?,?,?,?,?,?,?,?,?)""", rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _int_or_none(value):
    try:
        return None if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def growth_alerts(db_path, min_score_gain=20, min_comment_gain=10, lookback_hours=24) -> pd.DataFrame:
    """Compare each story's newest snapshot with its earliest snapshot in window."""
    columns = ["id", "title", "score_before", "score_now", "score_gain", "comments_before", "comments_now", "comment_gain", "observed_at"]
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='story_snapshots'").fetchone()
        if not exists:
            return pd.DataFrame(columns=columns)
        data = pd.read_sql_query("SELECT * FROM story_snapshots ORDER BY observed_at", conn)
    finally:
        conn.close()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["observed_at"] = pd.to_datetime(data.observed_at, utc=True)
    cutoff = data.observed_at.max() - pd.Timedelta(hours=lookback_hours)
    data = data[data.observed_at >= cutoff]
    first = data.groupby("id", as_index=False).first()
    last = data.groupby("id", as_index=False).last()
    merged = last[["id", "title", "score", "descendants", "observed_at"]].merge(
        first[["id", "score", "descendants"]], on="id", suffixes=("_now", "_before"))
    merged["score_gain"] = merged.score_now.fillna(0) - merged.score_before.fillna(0)
    merged["comment_gain"] = merged.descendants_now.fillna(0) - merged.descendants_before.fillna(0)
    alerts = merged[(merged.score_gain >= min_score_gain) | (merged.comment_gain >= min_comment_gain)].copy()
    return alerts.rename(columns={"score_before": "score_before", "score_now": "score_now",
        "descendants_before": "comments_before", "descendants_now": "comments_now"})[
        ["id", "title", "score_before", "score_now", "score_gain", "comments_before", "comments_now", "comment_gain", "observed_at"]].reset_index(drop=True)
