def build_kpis(df, threshold=200):
    scores = df["score"].dropna()
    return {
        "total_stories": int(len(df)),
        "stories_with_scores": int(scores.size),
        "unique_domains": int(df.loc[df["domain"].ne(""), "domain"].nunique()),
        "mean_score": round(float(scores.mean()), 2) if not scores.empty else None,
        "median_score": round(float(scores.median()), 2) if not scores.empty else None,
        "p90_score": round(float(scores.quantile(.90)), 2) if not scores.empty else None,
        "high_performing_threshold": int(threshold),
        "high_performing_stories": int(df["score"].gt(threshold).sum()),
        "high_performing_rate_pct": round(float(df["score"].gt(threshold).mean() * 100), 2) if len(df) else 0.0,
        "first_story_utc": str(df["created_at"].min()) if df["created_at"].notna().any() else None,
        "last_story_utc": str(df["created_at"].max()) if df["created_at"].notna().any() else None
    }

def analyze_stories(df):
    scored = df.dropna(subset=["score"]).copy()
    hourly = scored.groupby("hour_utc").agg(
        story_count=("score", "size"), mean_score=("score", "mean"),
        median_score=("score", "median"), high_performing_rate=("is_high_performing", "mean")
    ).reset_index()
    domain_source = scored[scored["domain"].ne("")]
    domains = domain_source.groupby("domain").agg(
        story_count=("score", "size"), mean_score=("score", "mean"),
        median_score=("score", "median"), high_performing_rate=("is_high_performing", "mean")
    ).reset_index().sort_values(["story_count", "mean_score"], ascending=False)
    title_lengths = scored.groupby("title_word_count").agg(
        story_count=("score", "size"), mean_score=("score", "mean"),
        high_performing_rate=("is_high_performing", "mean")
    ).reset_index().sort_values("title_word_count")
    daily = df.dropna(subset=["created_at"]).set_index("created_at").resample("D").size().rename("story_count").reset_index()
    return {"hourly": hourly, "domains": domains, "title_lengths": title_lengths, "daily": daily}
