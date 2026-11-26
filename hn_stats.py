"""v18.15 stats: quick descriptive summary of a story dataset.

Descriptive only. Scores and comment counts depend on story age and exposure;
these summaries do not measure author or domain quality.
"""
from __future__ import annotations

import pandas as pd

from hn_cleaning import clean_stories
from hn_data_loader import load_stories


def summarize(df: pd.DataFrame, top_n: int = 5) -> dict:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    score = pd.to_numeric(df["score"], errors="coerce").dropna()
    comments = pd.to_numeric(df["descendants"], errors="coerce").dropna()
    authors = df["by"].astype(str)
    domains = df["domain"].astype(str)
    named_domains = domains[domains != ""]

    def quantiles(series):
        if series.empty:
            return None
        return {q: float(series.quantile(p)) for q, p in
                [("min", 0), ("median", 0.5), ("p90", 0.9), ("max", 1)]}

    return {
        "stories": int(len(df)),
        "scored_stories": int(len(score)),
        "score": quantiles(score),
        "comments": quantiles(comments),
        "top_domains": {k: int(v) for k, v in named_domains.value_counts().head(top_n).items()},
        "top_authors": {k: int(v) for k, v in authors[authors != "unknown"].value_counts().head(top_n).items()},
        "time_range_utc": None if df["created_at"].dropna().empty else [
            df["created_at"].min().isoformat(), df["created_at"].max().isoformat()],
    }


def stats_for_file(path, top_n: int = 5) -> dict:
    return summarize(clean_stories(load_stories(path)), top_n=top_n)


def format_text(summary: dict) -> str:
    lines = [f"Stories: {summary['stories']} ({summary['scored_stories']} with a score)"]
    if summary["time_range_utc"]:
        lines.append(f"Time range (UTC): {summary['time_range_utc'][0]} to {summary['time_range_utc'][1]}")
    for label, key in [("Score", "score"), ("Comments", "comments")]:
        q = summary[key]
        if q:
            lines.append(f"{label}: min {q['min']:.0f}, median {q['median']:.0f}, p90 {q['p90']:.0f}, max {q['max']:.0f}")
    for label, key in [("Top domains", "top_domains"), ("Top authors", "top_authors")]:
        if summary[key]:
            lines.append(f"{label}: " + ", ".join(f"{k} ({v})" for k, v in summary[key].items()))
    return "\n".join(lines)
