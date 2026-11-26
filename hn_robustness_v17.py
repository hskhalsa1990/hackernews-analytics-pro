"""v17 network robustness analytics: bootstrap rank stability, damping
sensitivity and bridge (betweenness) scores.

All results describe how stable structural rankings are in the observed
author-topic graph. They do not measure influence, collaboration or causality.
"""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from hn_graph_v15 import build_author_topic_graph
from hn_network_v16 import _weighted_pagerank


def _betweenness(nodes, links):
    """Unweighted Brandes betweenness on the undirected graph, normalized to [0, 1]."""
    ids = [n['id'] for n in nodes]
    adj = {i: set() for i in ids}
    for e in links:
        if e['source'] != e['target']:
            adj[e['source']].add(e['target']); adj[e['target']].add(e['source'])
    score = dict.fromkeys(ids, 0.0)
    for s in ids:
        stack, preds = [], {i: [] for i in ids}
        sigma, dist = dict.fromkeys(ids, 0.0), dict.fromkeys(ids, -1)
        sigma[s], dist[s] = 1.0, 0
        queue = deque([s])
        while queue:
            v = queue.popleft(); stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1; queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; preds[w].append(v)
        delta = dict.fromkeys(ids, 0.0)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                score[w] += delta[w]
    n = len(ids)
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 0.0  # undirected: each pair counted twice
    return {k: v * scale for k, v in score.items()}


def _spearman(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 2 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return float('nan')
    return float(joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()))


def build_v17_robustness(stories, n_clusters=5, backend='tfidf', repeats=200, top_k=5,
                         damping=0.85, damping_grid=(0.5, 0.7, 0.85, 0.95), seed=42):
    if repeats < 1:
        raise ValueError('repeats must be at least 1.')
    if top_k < 1:
        raise ValueError('top_k must be at least 1.')
    if not 0 < damping < 1 or any(not 0 < d < 1 for d in damping_grid):
        raise ValueError('damping values must be between 0 and 1.')
    from hn_semantic_v14 import discover_topics
    _, assignments, topic_meta = discover_topics(stories, n_clusters=n_clusters, backend=backend)
    base_authors, _, _, _, graph, meta = build_author_topic_graph(stories, assignments, n_clusters, backend)
    base_ranks = _weighted_pagerank(graph['nodes'], graph['links'], damping)
    base = pd.Series({n['label']: base_ranks[n['id']] for n in graph['nodes'] if n['kind'] == 'author'})
    k = min(top_k, len(base))
    base_top = set(base.nlargest(k).index)

    # Bootstrap stories with fixed topic labels, isolating sampling noise in the graph.
    rng = np.random.default_rng(seed)
    data = stories.reset_index(drop=True)
    boot, rhos, overlaps = [], [], []
    for _ in range(repeats):
        idx = rng.integers(0, len(data), len(data))
        try:
            _, _, _, _, g, _ = build_author_topic_graph(data.iloc[idx].reset_index(drop=True),
                                                     assignments.iloc[idx].reset_index(drop=True),
                                                     n_clusters, backend)
        except ValueError:
            continue
        r = _weighted_pagerank(g['nodes'], g['links'], damping)
        s = pd.Series({n['label']: r[n['id']] for n in g['nodes'] if n['kind'] == 'author'})
        boot.append(s.reindex(base.index))
        rhos.append(_spearman(base, s))
        overlaps.append(len(base_top & set(s.nlargest(k).index)) / k)
    frame = pd.DataFrame(boot)
    done = len(frame)
    result = base_authors[['author', 'story_count']].copy()
    result['pagerank'] = result.author.map(base)
    result['bootstrap_mean'] = result.author.map(frame.mean()) if done else np.nan
    result['ci_low'] = result.author.map(frame.quantile(0.025)) if done else np.nan
    result['ci_high'] = result.author.map(frame.quantile(0.975)) if done else np.nan
    result['presence_rate'] = result.author.map(frame.notna().mean()) if done else np.nan
    result['top_k_frequency'] = result.author.map(
        frame.apply(lambda row: row >= row.nlargest(k).min(), axis=1).mean()) if done else np.nan
    bt = _betweenness(graph['nodes'], graph['links'])
    result['betweenness'] = result.author.map({n['label']: bt[n['id']] for n in graph['nodes'] if n['kind'] == 'author'})
    result = result.sort_values('pagerank', ascending=False).reset_index(drop=True)

    sens = []
    for d in damping_grid:
        rd = _weighted_pagerank(graph['nodes'], graph['links'], d)
        sd = pd.Series({n['label']: rd[n['id']] for n in graph['nodes'] if n['kind'] == 'author'})
        sens.append({'damping': d, 'spearman_vs_base': _spearman(base, sd),
                     'top_k_overlap': len(base_top & set(sd.nlargest(k).index)) / k})
    sensitivity = pd.DataFrame(sens, columns=['damping', 'spearman_vs_base', 'top_k_overlap'])

    topic_bt = pd.DataFrame([{'topic_cluster': int(n['id'].split(':')[1]), 'betweenness': bt[n['id']]}
                             for n in graph['nodes'] if n['kind'] == 'topic']
                            ).sort_values('betweenness', ascending=False).reset_index(drop=True)
    summary = {'repeats_requested': repeats, 'repeats_completed': done, 'top_k': k, 'seed': seed,
               'damping': damping, 'mean_spearman': float(np.nanmean(rhos)) if rhos else None,
               'mean_top_k_overlap': float(np.mean(overlaps)) if overlaps else None,
               'topic_backend': topic_meta.get('backend'),
               'method': 'story bootstrap with fixed topic labels; weighted PageRank; Brandes betweenness',
               'interpretation': 'Stability of structural rankings only; not influence, collaboration, or causality.'}
    return result, sensitivity, topic_bt, summary


def export_v17_robustness(stories, output_dir, n_clusters=5, backend='tfidf', repeats=200,
                          top_k=5, damping=0.85, seed=42):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    authors, sens, topics, summary = build_v17_robustness(stories, n_clusters, backend, repeats, top_k, damping, seed=seed)
    files = {'authors': out / 'v17_author_stability.csv', 'sensitivity': out / 'v17_damping_sensitivity.csv',
             'topics': out / 'v17_topic_bridges.csv', 'summary': out / 'v17_robustness_summary.json'}
    authors.to_csv(files['authors'], index=False); sens.to_csv(files['sensitivity'], index=False)
    topics.to_csv(files['topics'], index=False)
    files['summary'].write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {'authors': len(authors), **{f'{k}_file': str(v) for k, v in files.items()}, 'metadata': summary}
