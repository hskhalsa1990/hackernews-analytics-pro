"""v9 temporal intelligence: topic evolution and transparent snapshot trend baselines."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import sqlite3

import numpy as np
import pandas as pd

from hn_cleaning import text_or_empty

_STOP = set("a an and are as at be by for from has have how i in is it its of on or our that the their this to was we what when where who why will with you your hn http https com www".split())


def _time_series(df: pd.DataFrame, column: str | None = None) -> pd.Series:
    candidates = [column] if column else ["created_at", "time", "timestamp"]
    for name in candidates:
        if name and name in df.columns:
            values = df[name]
            if name in {"time", "timestamp"} and pd.api.types.is_numeric_dtype(values):
                return pd.to_datetime(values, unit="s", utc=True, errors="coerce")
            parsed = pd.to_datetime(values, utc=True, errors="coerce")
            if parsed.notna().any():
                return parsed
    raise ValueError("A usable created_at, time or timestamp column is required")


def build_topic_evolution(df: pd.DataFrame, freq: str = "W", top_n: int = 12,
                          time_column: str | None = None) -> dict:
    """Count title terms by UTC period; this is lexical trend analysis, not semantic NLP."""
    if "title" not in df.columns:
        raise ValueError("Dataset must include title")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    times = _time_series(df, time_column)
    work = pd.DataFrame({"title": df.title.fillna("").astype(str), "time": times}).dropna(subset=["time"])
    if work.empty:
        raise ValueError("No valid timestamps found")
    work["period"] = work.time.dt.tz_convert(None).dt.to_period(freq).astype(str)
    period_counts: dict[str, Counter] = {}
    overall = Counter()
    for period, group in work.groupby("period", sort=True):
        counts = Counter()
        for title in group.title:
            terms = set(re.findall(r"[a-z][a-z0-9_-]{2,}", title.lower())) - _STOP
            counts.update(terms)
        period_counts[period] = counts
        overall.update(counts)
    selected = [term for term, _ in overall.most_common(top_n)]
    rows = [{"period": period, "stories": int((work.period == period).sum()),
             **{term: int(counts.get(term, 0)) for term in selected}}
            for period, counts in period_counts.items()]
    return {"method": "UTC period-bucketed title-token counts", "frequency": freq,
            "documents": int(len(work)), "terms": selected, "periods": rows,
            "note": "Counts describe title-word frequency, not topic meaning or causal trends."}


def _read_snapshots(db_path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot database not found: {path}")
    with sqlite3.connect(path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='story_snapshots'").fetchone()
        if not exists:
            raise ValueError("Database has no story_snapshots table; run hn-analytics monitor first")
        data = pd.read_sql_query("SELECT * FROM story_snapshots", conn)
    if data.empty:
        raise ValueError("No snapshot observations are available")
    data["observed_at"] = pd.to_datetime(data.observed_at, utc=True, errors="coerce")
    data["score"] = pd.to_numeric(data.score, errors="coerce")
    data["descendants"] = pd.to_numeric(data.descendants, errors="coerce")
    return data.dropna(subset=["id", "observed_at"]).sort_values(["id", "observed_at"])


def _linear_projection(group: pd.DataFrame, horizon_hours: float) -> dict:
    """Fit a simple least-squares line against elapsed hours; return projection + residual spread."""
    g = group.dropna(subset=["observed_at"]).sort_values("observed_at")
    origin = g.observed_at.iloc[0]
    x = (g.observed_at - origin).dt.total_seconds().to_numpy(dtype=float) / 3600
    result = {}
    for col, key in (("score", "score"), ("descendants", "comments")):
        y = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            result[key] = None
            result[f"{key}_slope_per_hour"] = None
            result[f"{key}_residual_std"] = None
            continue
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        fitted = intercept + slope * x[valid]
        residual = y[valid] - fitted
        prediction = max(0.0, float(intercept + slope * (x[valid][-1] + horizon_hours)))
        result[key] = round(prediction, 2)
        result[f"{key}_slope_per_hour"] = round(float(slope), 4)
        result[f"{key}_residual_std"] = round(float(np.std(residual, ddof=1)), 2) if len(residual) > 1 else 0.0
    return result


def forecast_snapshot_engagement(db_path, horizon_hours: float = 6.0, min_observations: int = 3) -> pd.DataFrame:
    """Project observed score/comment trajectories; not a trained ML forecast model."""
    if horizon_hours <= 0 or min_observations < 3:
        raise ValueError("horizon_hours must be >0 and min_observations must be >=3")
    data = _read_snapshots(db_path)
    rows = []
    for story_id, group in data.groupby("id", sort=False):
        if len(group) < min_observations:
            continue
        projection = _linear_projection(group, horizon_hours)
        latest = group.iloc[-1]
        rows.append({"id": int(story_id), "title": text_or_empty(latest.get("title")),
                     "observations": int(len(group)), "last_observed_at": latest.observed_at.isoformat(),
                     "horizon_hours": float(horizon_hours), "score_now": latest.score,
                     "score_projected": projection["score"], "score_slope_per_hour": projection["score_slope_per_hour"],
                     "score_residual_std": projection["score_residual_std"], "comments_now": latest.descendants,
                     "comments_projected": projection["comments"], "comments_slope_per_hour": projection["comments_slope_per_hour"],
                     "comments_residual_std": projection["comments_residual_std"]})
    return pd.DataFrame(rows).sort_values("score_projected", ascending=False, na_position="last").reset_index(drop=True) if rows else pd.DataFrame()


def backtest_snapshot_trends(db_path, min_observations: int = 4) -> dict:
    """One-step chronological backtest versus persistence, using each story's last point as test."""
    if min_observations < 4:
        raise ValueError("min_observations must be >=4")
    data = _read_snapshots(db_path)
    actuals, trend_preds, naive_preds = [], [], []
    for _, group in data.groupby("id", sort=False):
        g = group.dropna(subset=["score"]).sort_values("observed_at")
        if len(g) < min_observations:
            continue
        train, test = g.iloc[:-1], g.iloc[-1]
        horizon = (test.observed_at - train.observed_at.iloc[-1]).total_seconds() / 3600
        if horizon <= 0:
            continue
        projection = _linear_projection(train, horizon)
        if projection["score"] is None:
            continue
        actuals.append(float(test.score)); trend_preds.append(float(projection["score"]))
        naive_preds.append(float(train.score.iloc[-1]))
    if not actuals:
        raise ValueError("Not enough stories with 4+ valid score snapshots for backtesting")
    y = np.asarray(actuals); p = np.asarray(trend_preds); n = np.asarray(naive_preds)
    metrics = lambda pred: {"mae": round(float(np.mean(np.abs(y - pred))), 3),
                            "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 3)}
    return {"method": "per-story one-step chronological holdout", "test_cases": len(y),
            "linear_trend": metrics(p), "persistence_baseline": metrics(n),
            "interpretation": "Small observational backtest only; accuracy is dataset- and horizon-dependent."}


def export_v9_temporal(df: pd.DataFrame, output_dir="reports", db_path=None,
                       freq="W", horizon_hours=6.0) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    evolution = build_topic_evolution(df, freq=freq)
    evolution_path = out / "v9_topic_evolution.json"
    evolution_path.write_text(json.dumps(evolution, indent=2), encoding="utf-8")
    artifacts = {"topic_evolution": str(evolution_path)}
    summary = {"version": "9.0.0", "topic_periods": len(evolution["periods"]),
               "topic_terms": len(evolution["terms"]), "artifacts": artifacts}
    if db_path:
        forecasts = forecast_snapshot_engagement(db_path, horizon_hours=horizon_hours)
        forecast_path = out / "v9_engagement_forecasts.csv"
        forecasts.to_csv(forecast_path, index=False)
        summary["forecast_rows"] = len(forecasts)
        artifacts["engagement_forecasts"] = str(forecast_path)
        try:
            backtest = backtest_snapshot_trends(db_path)
            backtest_path = out / "v9_forecast_backtest.json"
            backtest_path.write_text(json.dumps(backtest, indent=2), encoding="utf-8")
            summary["backtest"] = backtest
            artifacts["forecast_backtest"] = str(backtest_path)
        except ValueError as exc:
            summary["backtest_note"] = str(exc)
    summary_path = out / "v9_temporal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
