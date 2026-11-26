"""v15 author-topic graph intelligence.

Builds a bipartite author/topic network from observed stories and topic labels.
Edges represent observed author participation in a topic, not collaboration,
influence, endorsement, or causal relationships.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_author_topic_graph(stories: pd.DataFrame, topic_assignments: pd.DataFrame | None = None,
                             n_clusters: int = 5, backend: str = "tfidf"):
    """Return author metrics, topic metrics, graph JSON and metadata.

    If assignments are omitted, v14 topic discovery supplies exploratory cluster
    labels. Authors with missing names are excluded from author-network metrics.
    """
    if stories is None or stories.empty:
        raise ValueError("At least one story is required.")
    if "by" not in stories.columns and "author" not in stories.columns:
        raise ValueError("Stories must contain a 'by' or 'author' column.")

    if topic_assignments is None:
        from hn_semantic_v14 import discover_topics
        _, topic_assignments, topic_meta = discover_topics(stories, n_clusters=n_clusters, backend=backend)
    else:
        topic_meta = {"backend": "provided-assignments", "model": None}

    if "topic_cluster" not in topic_assignments.columns:
        raise ValueError("topic_assignments must contain 'topic_cluster'.")
    if len(topic_assignments) != len(stories):
        raise ValueError("topic_assignments must have exactly one row per story in the same order.")

    data = stories.reset_index(drop=True).copy()
    data["topic_cluster"] = topic_assignments["topic_cluster"].to_numpy()
    author_col = "by" if "by" in data.columns else "author"
    data["author_name"] = data[author_col].fillna("").astype(str).str.strip()
    data["score_num"] = pd.to_numeric(data.get("score", pd.Series(np.nan, index=data.index)), errors="coerce")
    data = data[(data.author_name != "") & data.topic_cluster.notna()].copy()

    if data.empty:
        raise ValueError("No rows have both an author and a topic assignment.")

    edges_frame = (data.groupby(["author_name", "topic_cluster"], as_index=False)
                   .agg(story_count=("author_name", "size"), mean_score=("score_num", "mean")))
    author_rows = []
    for author, group in edges_frame.groupby("author_name"):
        counts = group.story_count.to_numpy(dtype=float)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0
        max_entropy = float(np.log(len(probabilities))) if len(probabilities) > 1 else 0.0
        author_stories = data[data.author_name == author]
        author_rows.append({
            "author": author,
            "story_count": int(author_stories.shape[0]),
            "topic_count": int(group.topic_cluster.nunique()),
            "weighted_degree": int(counts.sum()),
            "topic_entropy": entropy,
            "normalized_topic_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
            "mean_score": float(author_stories.score_num.mean()) if author_stories.score_num.notna().any() else np.nan,
            "median_score": float(author_stories.score_num.median()) if author_stories.score_num.notna().any() else np.nan,
        })
    authors = pd.DataFrame(author_rows).sort_values(["topic_count", "story_count", "author"], ascending=[False, False, True]).reset_index(drop=True)

    topics = (data.groupby("topic_cluster", as_index=False)
              .agg(story_count=("topic_cluster", "size"), author_count=("author_name", "nunique"),
                   mean_score=("score_num", "mean"))
              .sort_values("story_count", ascending=False).reset_index(drop=True))

    author_nodes = [{"id": f"author:{r.author}", "label": r.author, "kind": "author",
                     "story_count": int(r.story_count), "topic_count": int(r.topic_count),
                     "mean_score": None if pd.isna(r.mean_score) else float(r.mean_score)}
                    for r in authors.itertuples(index=False)]
    topic_nodes = [{"id": f"topic:{int(r.topic_cluster)}", "label": f"Topic {int(r.topic_cluster)}",
                    "kind": "topic", "story_count": int(r.story_count), "author_count": int(r.author_count)}
                   for r in topics.itertuples(index=False)]
    links = [{"source": f"author:{r.author_name}", "target": f"topic:{int(r.topic_cluster)}",
              "weight": int(r.story_count), "mean_score": None if pd.isna(r.mean_score) else float(r.mean_score)}
             for r in edges_frame.itertuples(index=False)]

    # Project the bipartite graph into author pairs. Weighted Jaccard is based on
    # per-topic story counts and is descriptive overlap, not social interaction.
    author_topic = {}
    for r in edges_frame.itertuples(index=False):
        author_topic.setdefault(r.author_name, {})[int(r.topic_cluster)] = int(r.story_count)
    names = sorted(author_topic)
    overlap_rows = []
    for i, left in enumerate(names):
        left_map = author_topic[left]
        for right in names[i + 1:]:
            right_map = author_topic[right]
            shared = set(left_map) & set(right_map)
            if not shared:
                continue
            all_topics = set(left_map) | set(right_map)
            numerator = sum(min(left_map.get(t, 0), right_map.get(t, 0)) for t in all_topics)
            denominator = sum(max(left_map.get(t, 0), right_map.get(t, 0)) for t in all_topics)
            similarity = numerator / denominator if denominator else 0.0
            overlap_rows.append({"author_a": left, "author_b": right,
                                 "shared_topic_count": len(shared), "weighted_jaccard": similarity})
    overlaps = pd.DataFrame(overlap_rows, columns=["author_a", "author_b", "shared_topic_count", "weighted_jaccard"])
    if not overlaps.empty:
        overlaps = overlaps.sort_values(["weighted_jaccard", "shared_topic_count"], ascending=False).reset_index(drop=True)

    graph = {"directed": False, "multigraph": False, "nodes": author_nodes + topic_nodes,
             "links": links,
             "projection_edges": overlaps.to_dict(orient="records"),
             "semantics": "Author-topic participation and shared topic distribution; not collaboration or influence."}
    metadata = {"stories_input": int(len(stories)), "stories_with_author_and_topic": int(len(data)),
                "author_nodes": int(len(author_nodes)), "topic_nodes": int(len(topic_nodes)),
                "author_topic_edges": int(len(links)), "author_overlap_edges": int(len(overlaps)),
                "topic_backend": topic_meta.get("backend"), "limitations": graph["semantics"]}
    return authors, topics, edges_frame, overlaps, graph, metadata


def export_v15_graph(stories: pd.DataFrame, output_dir: str | Path, n_clusters: int = 5,
                     backend: str = "tfidf"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    authors, topics, edges, overlaps, graph, metadata = build_author_topic_graph(
        stories, n_clusters=n_clusters, backend=backend)
    files = {
        "authors": out / "v15_author_graph_metrics.csv",
        "topics": out / "v15_topic_graph_metrics.csv",
        "edges": out / "v15_author_topic_edges.csv",
        "overlaps": out / "v15_author_topic_overlap.csv",
        "graph": out / "v15_author_topic_graph.json",
        "summary": out / "v15_graph_summary.json",
    }
    authors.to_csv(files["authors"], index=False)
    topics.to_csv(files["topics"], index=False)
    edges.to_csv(files["edges"], index=False)
    overlaps.to_csv(files["overlaps"], index=False)
    files["graph"].write_text(json.dumps(graph, indent=2), encoding="utf-8")
    files["summary"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"authors": len(authors), "topics": len(topics), "edges": len(edges),
            "overlaps": len(overlaps), **{f"{k}_file": str(v) for k, v in files.items()},
            "metadata": metadata}
