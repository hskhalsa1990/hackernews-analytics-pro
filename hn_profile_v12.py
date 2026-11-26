"""v12 author/domain cohort profiling and chronological data-drift diagnostics.

These are descriptive analytics, not causal estimates or individual quality scores.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

VERSION = "12.0.0"


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("The input dataset contains no records.")
    df = raw.copy()
    aliases = {"by": ("by", "author"), "url": ("url",), "score": ("score", "points"),
               "descendants": ("descendants", "comments"), "time": ("time", "created_at_i")}
    for target, choices in aliases.items():
        source = next((c for c in choices if c in df.columns), None)
        if source and source != target:
            df[target] = df[source]
        elif target not in df:
            df[target] = np.nan if target in {"score", "descendants", "time"} else ""
    df["by"] = df["by"].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
    df["url"] = df["url"].fillna("").astype(str)
    df["domain"] = df["url"].map(lambda u: (urlparse(u).netloc or "").lower().removeprefix("www.") or "unknown")
    for col in ("score", "descendants", "time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["created_at"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["title"] = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    df["title_length"] = df["title"].str.len()
    return df


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denom
    margin = z * np.sqrt((p*(1-p) + z*z/(4*total))/total) / denom
    return max(0.0, centre-margin), min(1.0, centre+margin)


def cohort_profiles(raw: pd.DataFrame, min_stories: int = 3, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return author and domain cohorts with robust engagement summaries.

    High-score rate uses the dataset's 90th percentile as a descriptive threshold.
    Wilson intervals communicate uncertainty for that rate; they do not adjust for
    confounding, exposure, story age, or selection effects.
    """
    if min_stories < 1 or top_n < 1:
        raise ValueError("min_stories and top_n must be positive.")
    df = _normalise(raw)
    scores = df["score"].dropna()
    if scores.empty:
        raise ValueError("At least one numeric score is required for cohort profiling.")
    threshold = float(scores.quantile(.90))

    def aggregate(key: str) -> pd.DataFrame:
        rows = []
        for name, group in df.groupby(key, dropna=False):
            valid = group["score"].dropna()
            n = len(group)
            if n < min_stories:
                continue
            successes = int(valid.ge(threshold).sum())
            denominator = int(valid.notna().sum())
            low, high = _wilson(successes, denominator)
            rows.append({
                key: str(name), "stories": n, "scored_stories": denominator,
                "total_points": float(valid.sum()) if denominator else 0.0,
                "mean_score": float(valid.mean()) if denominator else np.nan,
                "median_score": float(valid.median()) if denominator else np.nan,
                "mean_comments": float(group["descendants"].mean()),
                "high_score_threshold": threshold,
                "top_decile_rate": successes / denominator if denominator else np.nan,
                "top_decile_rate_ci95_low": low, "top_decile_rate_ci95_high": high,
                "first_seen_utc": group["created_at"].min(), "last_seen_utc": group["created_at"].max(),
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["stories", "median_score"], ascending=[False, False]).head(top_n).reset_index(drop=True)
        return result

    return aggregate("by"), aggregate("domain")


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 2 or len(cur) < 2:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(np.nanmean(ref), np.nanmean(cur)) else np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    eps = 1e-6
    p = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
    q = np.clip(cur_counts / max(cur_counts.sum(), 1), eps, None)
    return float(np.sum((q-p) * np.log(q/p)))


def chronological_drift(raw: pd.DataFrame, reference_fraction: float = .5, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Compare earlier and later records using PSI on numeric features.

    Rows are ordered by timestamp when available, otherwise original row order.
    PSI is a screening statistic; thresholds are heuristic and should be contextualised.
    """
    if not .2 <= reference_fraction <= .8:
        raise ValueError("reference_fraction must be between 0.2 and 0.8.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    df = _normalise(raw)
    if df["created_at"].notna().sum() >= 4:
        df = df.sort_values("created_at", kind="stable", na_position="last")
        ordering = "created_at_utc"
    else:
        ordering = "input_row_order"
    cut = int(len(df) * reference_fraction)
    if cut < 2 or len(df)-cut < 2:
        raise ValueError("At least four records are required, with two in each period.")
    ref, cur = df.iloc[:cut], df.iloc[cut:]
    feature_map = {"score": "score", "comments": "descendants", "title_length": "title_length"}
    rows = []
    for label, col in feature_map.items():
        value = _psi(ref[col], cur[col], bins=bins)
        rows.append({"feature": label, "psi": value,
                     "drift_screen": ("high" if value >= .25 else "moderate" if value >= .10 else "low") if np.isfinite(value) else "insufficient_data",
                     "reference_non_null": int(ref[col].notna().sum()),
                     "current_non_null": int(cur[col].notna().sum()),
                     "reference_mean": float(ref[col].mean()) if ref[col].notna().any() else np.nan,
                     "current_mean": float(cur[col].mean()) if cur[col].notna().any() else np.nan})
    report = pd.DataFrame(rows)
    metadata = {"version": VERSION, "ordering": ordering, "reference_rows": len(ref), "current_rows": len(cur),
                "reference_fraction": reference_fraction, "bins": bins,
                "reference_start": str(ref["created_at"].min()), "reference_end": str(ref["created_at"].max()),
                "current_start": str(cur["created_at"].min()), "current_end": str(cur["created_at"].max()),
                "interpretation": "PSI is a descriptive drift screen, not a hypothesis test or proof of model degradation."}
    return report, metadata


def export_v12_profile(raw: pd.DataFrame, output_dir: str | Path, min_stories: int = 3,
                       reference_fraction: float = .5, bins: int = 10) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    authors, domains = cohort_profiles(raw, min_stories=min_stories)
    drift, metadata = chronological_drift(raw, reference_fraction=reference_fraction, bins=bins)
    files = {
        "authors": out / "v12_author_cohorts.csv", "domains": out / "v12_domain_cohorts.csv",
        "drift": out / "v12_drift_report.csv", "metadata": out / "v12_profile_metadata.json",
    }
    authors.to_csv(files["authors"], index=False)
    domains.to_csv(files["domains"], index=False)
    drift.to_csv(files["drift"], index=False)
    files["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"version": VERSION, "authors": len(authors), "domains": len(domains), "drift_features": len(drift),
            **{f"{k}_file": str(v) for k, v in files.items()}}
