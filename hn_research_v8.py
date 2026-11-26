"""v8 explainable story research: permutation importance, sentiment and topic graph."""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import re
import pandas as pd

from hn_nlp import sentiment_analysis
from hn_predictive import train_virality_model, _features

_STOP = set("a an and are as at be by for from has have how i in is it its of on or our that the their this to was we what when where who why will with you your hn http https com www".split())


def build_topic_network(df: pd.DataFrame, min_cooccurrence: int = 2, max_terms: int = 40) -> dict:
    """Create a transparent title-token co-occurrence graph (not semantic embeddings)."""
    if min_cooccurrence < 1 or max_terms < 2:
        raise ValueError("min_cooccurrence must be >=1 and max_terms >=2")
    if "title" not in df:
        raise ValueError("Dataset must include title")
    docs = []
    frequencies = Counter()
    for title in df.title.fillna("").astype(str):
        tokens = list(dict.fromkeys(t for t in re.findall(r"[a-z][a-z0-9_-]{2,}", title.lower()) if t not in _STOP))
        docs.append(tokens)
        frequencies.update(tokens)
    selected = {w for w, _ in frequencies.most_common(max_terms)}
    edges = Counter()
    for tokens in docs:
        terms = sorted(set(tokens) & selected)
        edges.update(combinations(terms, 2))
    nodes = [{"id": w, "frequency": int(frequencies[w])} for w in sorted(selected, key=lambda x: (-frequencies[x], x))]
    links = [{"source": a, "target": b, "weight": int(n)} for (a, b), n in sorted(edges.items()) if n >= min_cooccurrence]
    return {"method": "title-token co-occurrence", "documents": len(docs), "nodes": nodes, "edges": links,
            "note": "Edges indicate terms appearing in the same title; they do not imply causation or semantic similarity."}


def explain_classifier(df: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """Estimate raw-input feature importance with holdout permutation importance."""
    if repeats < 1:
        raise ValueError("repeats must be >=1")
    model, metrics, _ = train_virality_model(df)
    data = df.copy()
    data["score"] = pd.to_numeric(data.score, errors="coerce")
    data = data.dropna(subset=["score"])
    x = _features(data)
    if "created_at" in data:
        order = pd.to_datetime(data.created_at, utc=True, errors="coerce").sort_values(kind="stable").index
    elif "time" in data:
        order = pd.to_numeric(data.time, errors="coerce").sort_values(kind="stable").index
    else:
        order = data.index
    x = x.loc[order]
    scores = data.loc[order, "score"].clip(lower=0)
    split = metrics["train_rows"]
    y_test = (scores.iloc[split:] >= metrics["high_score_threshold"]).astype(int)
    if y_test.nunique() < 2:
        raise ValueError("Holdout must contain both outcome classes for permutation importance")
    try:
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise ImportError("Install scikit-learn to calculate explainability metrics") from exc
    result = permutation_importance(model, x.iloc[split:], y_test, scoring="roc_auc", n_repeats=repeats, random_state=42, n_jobs=-1)
    return pd.DataFrame({"feature": x.columns, "importance_mean": result.importances_mean,
                         "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def export_v8_research(df: pd.DataFrame, output_dir="reports", min_cooccurrence: int = 2, repeats: int = 5) -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sentiment = sentiment_analysis(df)
    sentiment_path = out / "v8_title_sentiment.csv"
    sentiment.to_csv(sentiment_path, index=False)
    distribution = sentiment.sentiment_label.value_counts().to_dict()
    graph = build_topic_network(df, min_cooccurrence=min_cooccurrence)
    graph_path = out / "v8_topic_network.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    importance = explain_classifier(df, repeats=repeats)
    importance_path = out / "v8_permutation_importance.csv"
    importance.to_csv(importance_path, index=False)
    summary = {"version": "8.0.0", "stories": int(len(df)), "sentiment_distribution": {str(k): int(v) for k, v in distribution.items()},
               "topic_nodes": len(graph["nodes"]), "topic_edges": len(graph["edges"]),
               "importance_top_feature": str(importance.iloc[0].feature) if not importance.empty else None,
               "artifacts": {"sentiment": str(sentiment_path), "topic_network": str(graph_path), "permutation_importance": str(importance_path)}}
    summary_path = out / "v8_research_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
