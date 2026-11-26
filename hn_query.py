"""v18.3 query and watch: filter stories by keyword, domain, author, score and date.

Matching is case-insensitive and literal (no regex). A story matches when every
keyword appears in its title. Inputs are cleaned stories (see ``clean_stories``).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from hn_cleaning import clean_stories, text_or_empty

RESULT_COLUMNS = ["id", "title", "score", "by", "domain", "created_at", "url"]


SORT_FIELDS = {"score": "score", "comments": "descendants", "time": "created_at"}


def filter_stories(df: pd.DataFrame, keywords=(), domain: str | None = None, author: str | None = None,
                   min_score: float | None = None, since: str | None = None,
                   limit: int = 20, sort: str = "score", exclude=(), days: float | None = None) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if sort not in SORT_FIELDS:
        raise ValueError(f"sort must be one of {sorted(SORT_FIELDS)}")
    out = df.copy()
    for kw in keywords:
        kw = kw.strip()
        if kw:
            out = out[out["title"].astype(str).str.contains(re.escape(kw), case=False, regex=True)]
    for kw in exclude:
        kw = kw.strip()
        if kw:
            out = out[~out["title"].astype(str).str.contains(re.escape(kw), case=False, regex=True)]
    if domain:
        d = domain.lower().removeprefix("www.")
        out = out[out["domain"].astype(str).str.lower() == d]
    if author:
        out = out[out["by"].astype(str).str.lower() == author.lower()]
    if min_score is not None:
        out = out[pd.to_numeric(out["score"], errors="coerce") >= float(min_score)]
    if days is not None:
        if days <= 0:
            raise ValueError("days must be positive")
        created_all = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        if created_all.dropna().empty:
            return out.iloc[0:0][[c for c in RESULT_COLUMNS if c in out.columns]].reset_index(drop=True)
        # Window is anchored to the newest story in the dataset, not the wall clock,
        # so the same file always gives the same result.
        anchor = created_all.max()
        created = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
        out = out[created >= anchor - pd.Timedelta(days=float(days))]
    if since:
        cutoff = pd.Timestamp(since, tz="UTC")
        created = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
        out = out[created >= cutoff]
    out = out.assign(score=pd.to_numeric(out["score"], errors="coerce"),
                     descendants=pd.to_numeric(out["descendants"], errors="coerce"))
    key = SORT_FIELDS[sort]
    out = out.sort_values(key, ascending=False, na_position="last").head(limit)
    cols = [c for c in RESULT_COLUMNS + ["descendants"] if c in out.columns]
    return out[cols].reset_index(drop=True)


def window_last_days(df: pd.DataFrame, days: float) -> pd.DataFrame:
    """Keep rows from the last ``days`` ending at the newest story (not the wall clock)."""
    if days <= 0:
        raise ValueError("days must be positive")
    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    if created.dropna().empty:
        return df.iloc[0:0]
    return df[created >= created.max() - pd.Timedelta(days=float(days))]


def run_query(input_path, **filters) -> pd.DataFrame:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    from hn_data_loader import load_stories
    return filter_stories(clean_stories(load_stories(input_path)), **filters)


def watch_history(db_path, keywords, lookback_hours: float = 24) -> pd.DataFrame:
    """Latest observation of each stored story whose title matches every keyword."""
    columns = ["id", "title", "score", "comments", "observed_at", "url"]
    if not Path(db_path).exists():
        return pd.DataFrame(columns=columns)
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='story_snapshots'").fetchone()
        if not exists:
            return pd.DataFrame(columns=columns)
        data = pd.read_sql_query("SELECT * FROM story_snapshots", conn)
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["observed_at"] = pd.to_datetime(data["observed_at"], utc=True, errors="coerce")
    data = data.dropna(subset=["observed_at"])
    if data.empty:
        return pd.DataFrame(columns=columns)
    cutoff = data["observed_at"].max() - pd.Timedelta(hours=float(lookback_hours))
    data = data[data["observed_at"] >= cutoff].sort_values("observed_at")
    latest = data.groupby("id", as_index=False).last()
    for kw in (k.strip() for k in keywords):
        if kw:
            latest = latest[latest["title"].map(text_or_empty).str.contains(re.escape(kw), case=False, regex=True)]
    latest = latest.assign(comments=latest["descendants"])
    return latest[columns].sort_values("score", ascending=False).reset_index(drop=True)
