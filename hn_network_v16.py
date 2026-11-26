"""v16 network structure analytics: weighted PageRank and overlap communities.

Scores describe structural prominence in the observed author-topic graph; they do
not measure real-world influence, collaboration, or causal impact.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hn_graph_v15 import build_author_topic_graph


def _weighted_pagerank(nodes, links, damping=0.85, max_iter=200, tolerance=1e-12):
    ids = [n['id'] for n in nodes]
    if not ids:
        return {}
    index = {node: i for i, node in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    for edge in links:
        a, b = index[edge['source']], index[edge['target']]
        w = max(float(edge.get('weight', 1)), 0.0)
        matrix[a, b] += w
        matrix[b, a] += w
    totals = matrix.sum(axis=1)
    transition = np.zeros_like(matrix)
    for i, total in enumerate(totals):
        if total > 0:
            transition[i] = matrix[i] / total
        else:
            transition[i] = 1.0 / len(ids)
    rank = np.full(len(ids), 1.0 / len(ids))
    for _ in range(max_iter):
        updated = (1 - damping) / len(ids) + damping * transition.T.dot(rank)
        if np.abs(updated - rank).sum() < tolerance:
            rank = updated
            break
        rank = updated
    return {node: float(rank[i]) for i, node in enumerate(ids)}


def _components(authors, overlaps, threshold):
    names = list(authors)
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    if not overlaps.empty:
        for r in overlaps.itertuples(index=False):
            if float(r.weighted_jaccard) >= threshold:
                union(r.author_a, r.author_b)
    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    return {name: i + 1 for i, group in enumerate(ordered) for name in group}, ordered


def build_v16_network(stories, n_clusters=5, backend='tfidf', similarity_threshold=0.15,
                      damping=0.85, max_iter=200, tolerance=1e-12):
    if not 0 <= similarity_threshold <= 1:
        raise ValueError('similarity_threshold must be between 0 and 1.')
    if not 0 < damping < 1:
        raise ValueError('damping must be between 0 and 1.')
    authors, topics, edges, overlaps, graph, meta = build_author_topic_graph(
        stories, n_clusters=n_clusters, backend=backend)
    ranks = _weighted_pagerank(graph['nodes'], graph['links'], damping, max_iter, tolerance)
    author_metrics = authors.copy()
    author_metrics['pagerank'] = author_metrics['author'].map(lambda x: ranks.get(f'author:{x}', 0.0))
    topic_metrics = topics.copy()
    topic_metrics['pagerank'] = topic_metrics['topic_cluster'].map(lambda x: ranks.get(f'topic:{int(x)}', 0.0))
    community_ids, groups = _components(author_metrics.author.tolist(), overlaps, similarity_threshold)
    author_metrics['overlap_community'] = author_metrics.author.map(community_ids)
    community_rows = []
    for cid, members in enumerate(groups, 1):
        community_rows.append({'community_id': cid, 'author_count': len(members),
                               'authors': ', '.join(members),
                               'total_stories': int(author_metrics[author_metrics.author.isin(members)].story_count.sum())})
    communities = pd.DataFrame(community_rows, columns=['community_id','author_count','authors','total_stories'])
    author_metrics = author_metrics.sort_values('pagerank', ascending=False).reset_index(drop=True)
    topic_metrics = topic_metrics.sort_values('pagerank', ascending=False).reset_index(drop=True)
    graph['nodes'] = [{**n, 'pagerank': ranks.get(n['id'], 0.0),
                       **({'overlap_community': community_ids.get(n['label'])} if n['kind']=='author' else {})}
                      for n in graph['nodes']]
    graph['network_analysis'] = {'method':'weighted undirected PageRank', 'damping':damping,
        'similarity_threshold':similarity_threshold, 'community_method':'connected components of author weighted-Jaccard overlap graph',
        'interpretation':'Descriptive graph structure only; not influence, collaboration, or causality.'}
    meta.update({'pagerank_sum': float(sum(ranks.values())), 'community_count':len(groups),
                 'similarity_threshold':similarity_threshold, 'damping':damping})
    return author_metrics, topic_metrics, communities, graph, meta


def export_v16_network(stories, output_dir, n_clusters=5, backend='tfidf', similarity_threshold=0.15, damping=0.85):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    authors, topics, communities, graph, meta = build_v16_network(stories,n_clusters,backend,similarity_threshold,damping)
    files={'authors':out/'v16_author_pagerank.csv','topics':out/'v16_topic_pagerank.csv',
           'communities':out/'v16_overlap_communities.csv','graph':out/'v16_network_graph.json',
           'summary':out/'v16_network_summary.json'}
    authors.to_csv(files['authors'],index=False); topics.to_csv(files['topics'],index=False)
    communities.to_csv(files['communities'],index=False)
    files['graph'].write_text(json.dumps(graph,indent=2),encoding='utf-8')
    files['summary'].write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return {'authors':len(authors),'topics':len(topics),'communities':len(communities),
            **{f'{k}_file':str(v) for k,v in files.items()},'metadata':meta}
