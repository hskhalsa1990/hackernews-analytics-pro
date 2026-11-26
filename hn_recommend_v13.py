"""Content-based story recommendations and near-duplicate discovery (v13)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _story_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
    # Use body/text when present, but cap it so a long article body cannot swamp its title.
    body_col = next((c for c in ("text", "description", "selftext") if c in frame.columns), None)
    body = frame[body_col].fillna("").astype(str).str.slice(0, 2500) if body_col else ""
    if isinstance(body, str):
        return title.str.strip()
    return (title.str.strip() + " " + body.str.strip()).str.strip()


def build_story_recommender(stories: pd.DataFrame, max_features: int = 12000):
    """Return TF-IDF matrix and vectorizer; rows retain the input frame's order."""
    if stories is None or stories.empty:
        raise ValueError("At least one story is required.")
    if max_features < 100:
        raise ValueError("max_features must be at least 100.")
    texts = _story_text(stories)
    if not texts.str.len().gt(0).any():
        raise ValueError("No usable story titles or text were found.")
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1,
        max_features=max_features, sublinear_tf=True, strip_accents="unicode"
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def recommend_stories(stories: pd.DataFrame, seed_index: int = 0, top_k: int = 10,
                      min_similarity: float = 0.05) -> pd.DataFrame:
    """Recommend stories similar to one row, using cosine similarity of TF-IDF text."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1.")
    if seed_index < 0 or seed_index >= len(stories):
        raise ValueError(f"seed_index must be between 0 and {max(len(stories)-1, 0)}.")
    _, matrix = build_story_recommender(stories)
    similarities = cosine_similarity(matrix[seed_index], matrix).ravel()
    similarities[seed_index] = -1.0
    order = np.argsort(-similarities)
    order = [int(i) for i in order if similarities[i] >= min_similarity][:top_k]
    cols = [c for c in ("id", "title", "url", "by", "score", "descendants", "time") if c in stories.columns]
    result = stories.iloc[order][cols].copy().reset_index(drop=True)
    result.insert(0, "source_row", order)
    result.insert(1, "similarity", [float(similarities[i]) for i in order])
    return result


def find_near_duplicates(stories: pd.DataFrame, threshold: float = 0.75,
                         max_pairs: int = 5000) -> pd.DataFrame:
    """Find lexical near-duplicate title/text pairs above a cosine threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive.")
    _, matrix = build_story_recommender(stories)
    # Keep the pairwise matrix sparse to avoid allocating an N x N dense array.
    sim = cosine_similarity(matrix, dense_output=False).tocoo()
    candidates = [(int(i), int(j), float(value))
                  for i, j, value in zip(sim.row, sim.col, sim.data)
                  if i < j and value >= threshold]
    candidates.sort(key=lambda item: item[2], reverse=True)
    rows = [{"left_row": left, "right_row": right, "similarity": value,
             "left_title": str(stories.iloc[left].get("title", "")),
             "right_title": str(stories.iloc[right].get("title", ""))}
            for left, right, value in candidates[:max_pairs]]
    return pd.DataFrame(rows, columns=["left_row", "right_row", "similarity", "left_title", "right_title"])


def export_v13_recommendations(stories: pd.DataFrame, output_dir: str | Path,
                               seed_index: int = 0, top_k: int = 10,
                               min_similarity: float = 0.05,
                               duplicate_threshold: float = 0.75) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = recommend_stories(stories, seed_index, top_k, min_similarity)
    duplicates = find_near_duplicates(stories, duplicate_threshold)
    seed = stories.iloc[seed_index]
    summary = {
        "version": "13.0.0", "method": "TF-IDF word/bigram cosine similarity",
        "seed_index": int(seed_index), "seed_title": str(seed.get("title", "")),
        "recommendations": int(len(recs)), "near_duplicate_pairs": int(len(duplicates)),
        "min_similarity": float(min_similarity), "duplicate_threshold": float(duplicate_threshold),
        "limitations": "Lexical similarity is not contextual semantic understanding; inspect results before treating items as duplicates."
    }
    rec_file, dup_file, meta_file = out / "v13_story_recommendations.csv", out / "v13_near_duplicates.csv", out / "v13_recommendation_summary.json"
    recs.to_csv(rec_file, index=False)
    duplicates.to_csv(dup_file, index=False)
    meta_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"recommendations": len(recs), "duplicates": len(duplicates),
            "recommendations_file": str(rec_file), "duplicates_file": str(dup_file),
            "summary_file": str(meta_file)}
