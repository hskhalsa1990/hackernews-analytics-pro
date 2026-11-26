from urllib.parse import urlparse
import pandas as pd

ALIASES = {
    "score": ["score", "points"], "time": ["time", "created_at_i", "created_at"],
    "title": ["title"], "url": ["url"], "type": ["type"],
    "by": ["by", "author"], "id": ["id", "objectID"],
    "descendants": ["descendants", "comments"]
}

def text_or_empty(value) -> str:
    """Return a string, mapping None/NaN/NA to '' (never the literal 'nan')."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass  # non-scalar value: keep its string form
    return str(value)

def clean_stories(raw):
    if raw.empty:
        raise ValueError("The input dataset contains no records.")
    df = raw.copy()
    for target, candidates in ALIASES.items():
        source = next((c for c in candidates if c in df.columns), None)
        if source and source != target:
            df[target] = df[source]
        elif target not in df.columns:
            df[target] = pd.NA
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["descendants"] = pd.to_numeric(df["descendants"], errors="coerce")
    df["by"] = df["by"].fillna("unknown").astype(str)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["type"] = df["type"].fillna("unknown").astype(str).str.lower()
    df["url"] = df["url"].fillna("").astype(str)
    df["domain"] = df["url"].map(lambda u: (urlparse(u).netloc or "").lower().removeprefix("www."))
    df["created_at"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df["date"] = df["created_at"].dt.date
    df["hour_utc"] = df["created_at"].dt.hour
    df["weekday_utc"] = df["created_at"].dt.day_name()
    df["title_word_count"] = df["title"].str.findall(r"\b[\w'-]+\b").str.len()
    df["title_char_count"] = df["title"].str.len()
    df["is_high_performing"] = df["score"].gt(200).fillna(False)
    # Deduplicate only on real ids. Rows with a missing id are kept (exact duplicates
    # among them are dropped); previously every id-less row collapsed into one.
    has_id = df["id"].notna()
    with_id = df[has_id].drop_duplicates(subset=["id"], keep="first")
    without_id = df[~has_id].drop_duplicates()
    df = pd.concat([with_id, without_id]).sort_index()
    return df.reset_index(drop=True)

def quality_report(df):
    return pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str)
    }).sort_values("missing_pct", ascending=False)
