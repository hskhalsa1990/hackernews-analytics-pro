"""Advanced Hacker News analytics: engagement, cohorts, trends and anomaly flags."""
import numpy as np
import pandas as pd

def engagement_analysis(df):
    out = df.copy()
    out["comments_per_point"] = np.where(out["score"].fillna(0) > 0,
                                          out["descendants"] / out["score"], np.nan)
    out["engagement_total"] = out["score"].fillna(0) + out["descendants"].fillna(0)
    by_type = out.groupby("type", dropna=False).agg(
        stories=("type", "size"), mean_score=("score", "mean"),
        median_comments=("descendants", "median"),
        mean_engagement=("engagement_total", "mean")).reset_index()
    authors = out.groupby("by", dropna=False).agg(
        stories=("id", "size"), total_points=("score", "sum"),
        mean_score=("score", "mean"), total_comments=("descendants", "sum")
    ).reset_index().sort_values(["stories", "total_points"], ascending=False)
    return {"by_type": by_type, "authors": authors, "records": out}

def trend_analysis(df, window=7):
    if window < 2:
        raise ValueError("window must be at least 2")
    daily = (df.dropna(subset=["created_at"]).set_index("created_at")
             .resample("D").agg(stories=("id", "size"), points=("score", "sum"),
                                comments=("descendants", "sum")).fillna(0))
    daily["stories_rolling_mean"] = daily["stories"].rolling(window, min_periods=1).mean()
    daily["stories_rolling_std"] = daily["stories"].rolling(window, min_periods=2).std().fillna(0)
    # Descriptive anomaly score; not a causal or predictive claim.
    denom = daily["stories_rolling_std"].replace(0, np.nan)
    daily["volume_zscore"] = ((daily["stories"] - daily["stories_rolling_mean"]) / denom).fillna(0)
    daily["volume_anomaly"] = daily["volume_zscore"].abs() >= 2.5
    return daily.reset_index()

def compare_segments(df, segment="type", metric="score"):
    if segment not in df.columns or metric not in df.columns:
        raise ValueError("segment and metric must be existing columns")
    return (df.groupby(segment, dropna=False)[metric]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index().sort_values("count", ascending=False))

def statistical_summary(df):
    scores = pd.to_numeric(df["score"], errors="coerce").dropna()
    comments = pd.to_numeric(df["descendants"], errors="coerce").dropna()
    corr = df[["score", "descendants", "title_word_count"]].corr(numeric_only=True)
    return {
        "score_iqr": float(scores.quantile(.75) - scores.quantile(.25)) if len(scores) else None,
        "score_std": float(scores.std()) if len(scores) > 1 else None,
        "comment_median": float(comments.median()) if len(comments) else None,
        "score_comment_correlation": float(corr.loc["score", "descendants"])
            if "score" in corr and "descendants" in corr and pd.notna(corr.loc["score", "descendants"]) else None
    }
