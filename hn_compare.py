"""v18.16 compare: side-by-side descriptive comparison of two story datasets.

Shows how counts, median engagement and domain shares differ between a baseline
and a candidate dataset (for example two time windows). These are descriptive
differences only: they do not test significance or control for story age,
exposure or feed selection.
"""
from __future__ import annotations

import pandas as pd

from hn_cleaning import clean_stories
from hn_data_loader import load_stories


def _domain_shares(df: pd.DataFrame) -> pd.Series:
    domains = df["domain"].astype(str)
    named = domains[domains != ""]
    if named.empty:
        return pd.Series(dtype=float)
    return named.value_counts(normalize=True)


def compare_datasets(baseline: pd.DataFrame, candidate: pd.DataFrame, top_n: int = 5) -> dict:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if baseline.empty or candidate.empty:
        raise ValueError("both datasets must contain stories")

    def medians(df):
        score = pd.to_numeric(df["score"], errors="coerce")
        comments = pd.to_numeric(df["descendants"], errors="coerce")
        return (None if score.dropna().empty else float(score.median()),
                None if comments.dropna().empty else float(comments.median()))

    b_score, b_comments = medians(baseline)
    c_score, c_comments = medians(candidate)
    shares_b, shares_c = _domain_shares(baseline), _domain_shares(candidate)
    domains = sorted(set(shares_b.index) | set(shares_c.index))
    shift = pd.DataFrame({
        "domain": domains,
        "baseline_share": [float(shares_b.get(d, 0.0)) for d in domains],
        "candidate_share": [float(shares_c.get(d, 0.0)) for d in domains],
    })
    shift["share_change"] = shift["candidate_share"] - shift["baseline_share"]
    shift = shift.reindex(shift["share_change"].abs().sort_values(ascending=False).index).head(top_n)

    def delta(a, b):
        return None if a is None or b is None else b - a

    return {
        "baseline_stories": int(len(baseline)),
        "candidate_stories": int(len(candidate)),
        "median_score": {"baseline": b_score, "candidate": c_score, "change": delta(b_score, c_score)},
        "median_comments": {"baseline": b_comments, "candidate": c_comments, "change": delta(b_comments, c_comments)},
        "domain_shifts": shift.round(4).to_dict(orient="records"),
    }


def compare_files(baseline_path, candidate_path, top_n: int = 5) -> dict:
    return compare_datasets(clean_stories(load_stories(baseline_path)),
                            clean_stories(load_stories(candidate_path)), top_n=top_n)


def format_comparison(result: dict) -> str:
    lines = [f"Stories: baseline {result['baseline_stories']}, candidate {result['candidate_stories']}"]
    for label, key in [("Median score", "median_score"), ("Median comments", "median_comments")]:
        m = result[key]
        if m["change"] is None:
            lines.append(f"{label}: not available")
        else:
            lines.append(f"{label}: {m['baseline']:.1f} -> {m['candidate']:.1f} ({m['change']:+.1f})")
    if result["domain_shifts"]:
        lines.append("Largest domain share shifts:")
        for row in result["domain_shifts"]:
            lines.append(f"  {row['domain']}: {row['baseline_share']:.1%} -> {row['candidate_share']:.1%} "
                         f"({row['share_change']:+.1%})")
    lines.append("Descriptive differences only; no significance test or adjustment for story age or exposure.")
    return "\n".join(lines)
