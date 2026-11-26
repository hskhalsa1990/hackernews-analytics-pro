"""v18.19 dashboard helpers: pure functions behind the 'Search & compare' tab.

Kept free of Streamlit imports so the logic is unit-tested without the optional
dashboard dependency. The Streamlit tab only lays out widgets and calls these.
"""
from __future__ import annotations

import pandas as pd

from hn_compare import compare_datasets
from hn_digest import build_digest
from hn_query import filter_stories, window_last_days
from hn_stats import summarize

SEARCH_COLUMNS = ["id", "title", "score", "descendants", "by", "domain", "created_at", "url"]


def search_stories(df: pd.DataFrame, keyword: str = "", min_score: float | None = None,
                   sort: str = "score", days: float | None = None, limit: int = 50) -> pd.DataFrame:
    """Same filtering as the `query` command, applied to an already-cleaned frame."""
    if days is not None:
        df = window_last_days(df, days)
    keywords = [keyword] if keyword.strip() else []
    return filter_stories(df, keywords=keywords, min_score=min_score, sort=sort, limit=limit)


def digest_preview(df: pd.DataFrame, keyword: str = "", days: float | None = None, top_n: int = 10) -> str:
    """Markdown digest of top and most-discussed stories (no momentum history in the dashboard)."""
    frame = df
    if days is not None:
        frame = window_last_days(frame, days)
    if keyword.strip():
        mask = frame["title"].astype(str).str.contains(keyword.strip(), case=False, regex=False)
        frame = frame[mask]
    return build_digest(frame, None, top_n=top_n)


def summary_tables(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Stats summary plus the top-domain and top-author tables for display."""
    summary = summarize(df, top_n=top_n)
    domains = pd.DataFrame(list(summary["top_domains"].items()), columns=["domain", "stories"])
    authors = pd.DataFrame(list(summary["top_authors"].items()), columns=["author", "stories"])
    return {"summary": summary, "domains": domains, "authors": authors}


def split_halves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into earlier and later halves by creation time, for in-app comparison."""
    ordered = df.dropna(subset=["created_at"]).sort_values("created_at")
    if len(ordered) < 2:
        raise ValueError("need at least two dated stories to split into halves")
    mid = len(ordered) // 2
    return ordered.iloc[:mid], ordered.iloc[mid:]


def compare_halves(df: pd.DataFrame, top_n: int = 5) -> dict:
    earlier, later = split_halves(df)
    return compare_datasets(earlier, later, top_n=top_n)
