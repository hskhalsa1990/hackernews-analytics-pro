"""Story momentum analytics derived from recorded Hacker News snapshots."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

COLUMNS = ["id", "title", "author", "url", "first_seen", "last_seen", "elapsed_hours",
           "score_before", "score_now", "score_gain", "score_per_hour", "comments_before",
           "comments_now", "comment_gain", "comments_per_hour", "rank_before", "rank_now",
           "rank_improvement", "momentum_status"]

def story_momentum(db_path, lookback_hours=24) -> pd.DataFrame:
    """Compare each story's first and latest observations in a UTC lookback window.

    Positive rank_improvement means the story moved closer to rank 1. Velocity
    fields are per elapsed hour; missing/zero elapsed time produces NaN.
    """
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    with sqlite3.connect(path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='story_snapshots'").fetchone()
        if not exists:
            return pd.DataFrame(columns=COLUMNS)
        data = pd.read_sql_query("SELECT * FROM story_snapshots ORDER BY observed_at", conn)
    if data.empty:
        return pd.DataFrame(columns=COLUMNS)
    data["observed_at"] = pd.to_datetime(data["observed_at"], utc=True, errors="coerce")
    data = data.dropna(subset=["observed_at"])
    if data.empty:
        return pd.DataFrame(columns=COLUMNS)
    cutoff = data["observed_at"].max() - pd.Timedelta(hours=float(lookback_hours))
    data = data[data["observed_at"] >= cutoff].sort_values(["id", "observed_at"], kind="stable")
    if data.empty:
        return pd.DataFrame(columns=COLUMNS)
    first = data.groupby("id", as_index=False).first()
    last = data.groupby("id", as_index=False).last()
    keep = ["id", "title", "by_author", "url", "score", "descendants", "rank", "observed_at"]
    merged = first[keep].merge(last[keep], on="id", suffixes=("_before", "_now"))
    merged["elapsed_hours"] = (merged.observed_at_now - merged.observed_at_before).dt.total_seconds() / 3600
    merged["score_gain"] = merged.score_now.fillna(0) - merged.score_before.fillna(0)
    merged["comment_gain"] = merged.descendants_now.fillna(0) - merged.descendants_before.fillna(0)
    safe_hours = merged.elapsed_hours.where(merged.elapsed_hours > 0)
    merged["score_per_hour"] = merged.score_gain / safe_hours
    merged["comments_per_hour"] = merged.comment_gain / safe_hours
    merged["rank_improvement"] = merged.rank_before - merged.rank_now
    def status(row):
        if row["score_gain"] > 0 or row["comment_gain"] > 0 or (pd.notna(row["rank_improvement"]) and row["rank_improvement"] > 0):
            return "Rising"
        if row["score_gain"] < 0 or row["comment_gain"] < 0 or (pd.notna(row["rank_improvement"]) and row["rank_improvement"] < 0):
            return "Cooling"
        return "Flat"
    merged["momentum_status"] = merged.apply(status, axis=1)
    out = pd.DataFrame({
        "id": merged.id, "title": merged.title_now.fillna(merged.title_before),
        "author": merged.by_author_now.fillna(merged.by_author_before), "url": merged.url_now.fillna(merged.url_before),
        "first_seen": merged.observed_at_before, "last_seen": merged.observed_at_now,
        "elapsed_hours": merged.elapsed_hours, "score_before": merged.score_before,
        "score_now": merged.score_now, "score_gain": merged.score_gain,
        "score_per_hour": merged.score_per_hour, "comments_before": merged.descendants_before,
        "comments_now": merged.descendants_now, "comment_gain": merged.comment_gain,
        "comments_per_hour": merged.comments_per_hour, "rank_before": merged.rank_before,
        "rank_now": merged.rank_now, "rank_improvement": merged.rank_improvement,
        "momentum_status": merged.momentum_status})
    return out[COLUMNS].sort_values(["score_per_hour", "comments_per_hour"], ascending=False, na_position="last").reset_index(drop=True)

def export_momentum(db_path, output_dir="reports", lookback_hours=24):
    """Export momentum results to CSV and JSON and return their paths."""
    from pathlib import Path
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    result = story_momentum(db_path, lookback_hours)
    csv_path, json_path = out / "story_momentum.csv", out / "story_momentum.json"
    result.to_csv(csv_path, index=False)
    result.to_json(json_path, orient="records", date_format="iso", indent=2)
    return {"rows": len(result), "csv": str(csv_path), "json": str(json_path)}
