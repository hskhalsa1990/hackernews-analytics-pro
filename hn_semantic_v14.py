"""v14 semantic discovery: optional sentence embeddings, MMR ranking and topic clusters.

The default auto backend uses SentenceTransformers when installed; otherwise it
uses a transparent TF-IDF fallback and records that choice in exported metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hn_recommend_v13 import _story_text


def _embeddings(stories: pd.DataFrame, backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if backend not in {"auto", "sentence-transformers", "tfidf"}:
        raise ValueError("backend must be auto, sentence-transformers or tfidf")
    texts = _story_text(stories).tolist()
    if not any(str(t).strip() for t in texts):
        raise ValueError("No usable story titles or text were found.")
    fallback_reason = None
    if backend != "tfidf":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            matrix = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=float)
            return matrix, "sentence-transformers", model_name, None
        except Exception as exc:
            if backend == "sentence-transformers":
                if isinstance(exc, ImportError):
                    raise ImportError("Install optional semantic dependencies: pip install -r requirements-ai.txt") from exc
                raise RuntimeError(f"Could not load embedding model {model_name}: {exc}") from exc
            fallback_reason = f"{type(exc).__name__}: {exc}"
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1,
                                 max_features=12000, sublinear_tf=True, strip_accents="unicode")
    matrix = vectorizer.fit_transform(texts)
    return matrix, "tfidf-fallback", "", fallback_reason


def _mmr(query_sim: np.ndarray, pairwise: np.ndarray, candidates: list[int], top_k: int,
         relevance_weight: float = 0.78) -> list[int]:
    """Maximal marginal relevance balances query relevance and result diversity."""
    chosen: list[int] = []
    remaining = candidates.copy()
    while remaining and len(chosen) < top_k:
        def score(idx: int) -> float:
            redundancy = max((float(pairwise[idx, j]) for j in chosen), default=0.0)
            return relevance_weight * float(query_sim[idx]) - (1 - relevance_weight) * redundancy
        best = max(remaining, key=score)
        chosen.append(best)
        remaining.remove(best)
    return chosen


def semantic_recommendations(stories: pd.DataFrame, seed_index: int = 0, top_k: int = 10,
                             min_similarity: float = 0.05, diversity: float = 0.22,
                             backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if stories is None or len(stories) < 2:
        raise ValueError("At least two stories are required.")
    if not 0 <= seed_index < len(stories):
        raise ValueError(f"seed_index must be between 0 and {len(stories)-1}.")
    if top_k < 1 or not 0 <= min_similarity <= 1 or not 0 <= diversity <= 1:
        raise ValueError("top_k must be positive; similarity and diversity must be between 0 and 1.")
    matrix, actual_backend, used_model, fallback_reason = _embeddings(stories, backend, model_name)
    pairwise = cosine_similarity(matrix)
    query_sim = pairwise[seed_index].copy()
    query_sim[seed_index] = -1
    candidates = [i for i in range(len(stories)) if i != seed_index and query_sim[i] >= min_similarity]
    chosen = _mmr(query_sim, pairwise, candidates, top_k, relevance_weight=1.0-diversity)
    cols = [c for c in ("id", "title", "url", "by", "score", "descendants", "time") if c in stories.columns]
    result = stories.iloc[chosen][cols].copy().reset_index(drop=True)
    result.insert(0, "source_row", chosen)
    result.insert(1, "semantic_similarity", [float(query_sim[i]) for i in chosen])
    result.insert(2, "mmr_rank", np.arange(1, len(chosen)+1))
    result.insert(3, "backend", actual_backend)
    return result, {"backend": actual_backend, "model": used_model, "stories": len(stories),
                    "seed_index": int(seed_index), "diversity": float(diversity), "fallback_reason": fallback_reason}


def discover_topics(stories: pd.DataFrame, n_clusters: int = 5, backend: str = "auto",
                    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if stories is None or len(stories) < 2:
        raise ValueError("At least two stories are required for topic discovery.")
    if not 2 <= n_clusters <= min(30, len(stories)):
        raise ValueError(f"n_clusters must be between 2 and {min(30, len(stories))}.")
    matrix, actual_backend, used_model, fallback_reason = _embeddings(stories, backend, model_name)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(matrix)
    texts = _story_text(stories).tolist()
    rows = []
    for cluster in range(n_clusters):
        indices = np.flatnonzero(labels == cluster)
        if not len(indices):
            continue
        # Human-readable representative titles; not asserted to be canonical topic names.
        centroid = np.asarray(matrix[indices].mean(axis=0)).ravel()
        sims = cosine_similarity(matrix[indices], centroid.reshape(1, -1)).ravel()
        representative = int(indices[int(np.argmax(sims))])
        rows.append({"cluster": int(cluster), "story_count": int(len(indices)),
                     "representative_row": representative,
                     "representative_title": str(stories.iloc[representative].get("title", "")),
                     "sample_titles": " | ".join(str(stories.iloc[i].get("title", "")) for i in indices[:5])})
    assignments = stories[[c for c in ("id", "title", "by", "score") if c in stories.columns]].copy()
    assignments["topic_cluster"] = labels
    return pd.DataFrame(rows).sort_values("story_count", ascending=False).reset_index(drop=True), assignments, {
        "backend": actual_backend, "model": used_model, "n_clusters": n_clusters,
        "limitations": "Clusters are unsupervised groups; representative titles are descriptive labels, not validated semantic taxonomies.",
        "fallback_reason": fallback_reason}


def export_v14_semantic(stories: pd.DataFrame, output_dir: str | Path, seed_index: int = 0,
                        top_k: int = 10, min_similarity: float = 0.05, diversity: float = 0.22,
                        n_clusters: int = 5, backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    recs, rec_meta = semantic_recommendations(stories, seed_index, top_k, min_similarity, diversity, backend, model_name)
    topics, assignments, topic_meta = discover_topics(stories, n_clusters, backend, model_name)
    files = {"recommendations": out / "v14_semantic_recommendations.csv",
             "topics": out / "v14_topic_clusters.csv", "assignments": out / "v14_topic_assignments.csv",
             "summary": out / "v14_semantic_summary.json"}
    recs.to_csv(files["recommendations"], index=False); topics.to_csv(files["topics"], index=False)
    assignments.to_csv(files["assignments"], index=False)
    summary = {"version": "14.0.0", "recommendation": rec_meta, "topics": topic_meta,
               "recommendation_count": len(recs), "topic_count": len(topics)}
    files["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"recommendations": len(recs), "topics": len(topics), **{f"{k}_file": str(v) for k,v in files.items()}, "metadata": summary}
