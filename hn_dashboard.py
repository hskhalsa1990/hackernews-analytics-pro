"""Optional interactive Streamlit dashboard. Run: streamlit run hn_dashboard.py"""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hn_analytics import (load_stories, clean_stories, build_kpis, analyze_stories,
                          engagement_analysis, trend_analysis, sentiment_analysis)
from hn_dashboard_v18 import (search_stories, digest_preview, summary_tables, compare_halves)

st.set_page_config(page_title="HackerNews Analytics Pro", page_icon="📰", layout="wide")
st.title("📰 HackerNews Analytics Pro")
st.caption("Exploratory analytics for Hacker News datasets — demo data is synthetic.")
uploaded = st.sidebar.file_uploader("Upload CSV, JSON or JSONL", type=["csv", "json", "jsonl", "ndjson"])
if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    raw = pd.read_csv(uploaded) if suffix == ".csv" else pd.read_json(uploaded, lines=suffix in [".jsonl", ".ndjson"])
else:
    candidate = ROOT / "HNStories.jsonl"
    if not candidate.exists():
        candidate = ROOT / "demo_stories.jsonl"
    raw = load_stories(candidate)
threshold = st.sidebar.slider("High-performing threshold (points)", 50, 1000, 200, 25)
df = clean_stories(raw)
kpis = build_kpis(df, threshold)
cols = st.columns(4)
for col, label, key in zip(cols, ["Stories", "Mean points", "Median points", "High-performing %"],
                           ["total_stories", "mean_score", "median_score", "high_performing_rate_pct"]):
    col.metric(label, kpis[key] if kpis[key] is not None else "N/A")
tabs = st.tabs(["Overview", "Domains & authors", "Trends", "NLP sentiment", "Data quality", "Live intelligence", "Predictive research", "Explainable AI v8", "Temporal intelligence v9", "Model benchmark v10", "Walk-forward v11", "Cohorts & drift v12", "Author-topic graph v15", "Network structure v16-v17", "Search & compare v18"])
with tabs[0]:
    a = analyze_stories(df)
    st.subheader("Daily story volume")
    st.line_chart(a["daily"].set_index("created_at")["story_count"])
    st.subheader("Average points by UTC hour")
    st.bar_chart(a["hourly"].set_index("hour_utc")["mean_score"])
with tabs[1]:
    e = engagement_analysis(df)
    left, right = st.columns(2)
    left.subheader("Top domains")
    left.dataframe(a["domains"].head(20), use_container_width=True)
    right.subheader("Top authors")
    right.dataframe(e["authors"].head(20), use_container_width=True)
with tabs[2]:
    t = trend_analysis(df)
    st.line_chart(t.set_index("created_at")[["stories", "stories_rolling_mean"]])
    st.dataframe(t[t["volume_anomaly"]], use_container_width=True)
with tabs[3]:
    nlp = sentiment_analysis(df)
    st.subheader("Title sentiment (lexicon baseline)")
    st.bar_chart(nlp["sentiment_label"].value_counts())
    st.dataframe(nlp[["title", "sentiment_positive", "sentiment_negative", "sentiment_label"]].head(100), use_container_width=True)
with tabs[4]:
    from hn_cleaning import quality_report
    st.dataframe(quality_report(df), use_container_width=True)
st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "hn_cleaned.csv", "text/csv")

with tabs[5]:
    st.subheader("Recorded story momentum")
    db_path = ROOT / "hn_monitor.sqlite"
    if db_path.exists():
        from hn_intelligence import story_momentum
        lookback = st.slider("Lookback window (hours)", 1, 168, 24)
        momentum = story_momentum(db_path, lookback)
        if momentum.empty:
            st.info("No repeated observations in this window yet. Run `hn-analytics monitor` to collect snapshots.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Observed stories", len(momentum))
            m2.metric("Rising stories", int((momentum.momentum_status == "Rising").sum()))
            m3.metric("Cooling stories", int((momentum.momentum_status == "Cooling").sum()))
            st.bar_chart(momentum.set_index("title")["score_per_hour"].head(15))
            st.dataframe(momentum, use_container_width=True)
            st.download_button("Download momentum CSV", momentum.to_csv(index=False).encode(), "story_momentum.csv", "text/csv")
    else:
        st.info("No monitoring database found. Start collection with `hn-analytics monitor`.")


with tabs[6]:
    st.subheader("v7 predictive research")
    st.caption("A retrospective high-score classification baseline and engagement outlier scan. It does not promise future story performance.")
    if st.button("Run predictive analysis", type="primary"):
        from hn_predictive import train_virality_model, detect_engagement_anomalies
        try:
            _, metrics, predictions = train_virality_model(df)
            anomalies = detect_engagement_anomalies(df)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
            m2.metric("Precision", f"{metrics['precision']:.3f}")
            m3.metric("Recall", f"{metrics['recall']:.3f}")
            m4.metric("Flagged outliers", int(anomalies.is_anomaly.sum()))
            st.write(f"Training-derived high-score threshold: **{metrics['high_score_threshold']:.1f} points**")
            st.json(metrics)
            st.subheader("Engagement anomalies")
            st.dataframe(anomalies[anomalies.is_anomaly], use_container_width=True)
            st.download_button("Download anomaly report", anomalies.to_csv(index=False).encode(), "engagement_anomalies.csv", "text/csv")
        except (ValueError, ImportError) as exc:
            st.warning(str(exc))


with tabs[7]:
    st.subheader("v8 explainable intelligence")
    st.caption("Permutation importance describes model reliance on the chronological holdout; title co-occurrence edges are lexical, not semantic.")
    if st.button("Run v8 explainability and topic analysis"):
        from hn_research_v8 import build_topic_network, explain_classifier
        try:
            importance = explain_classifier(df, repeats=3)
            graph = build_topic_network(df, min_cooccurrence=2)
            left, right = st.columns(2)
            left.metric("Topic graph nodes", len(graph["nodes"]))
            right.metric("Topic graph edges", len(graph["edges"]))
            st.subheader("Holdout permutation importance")
            st.bar_chart(importance.set_index("feature")["importance_mean"])
            st.dataframe(importance, use_container_width=True)
            st.download_button("Download topic graph JSON", __import__("json").dumps(graph, indent=2), "v8_topic_network.json", "application/json")
            st.download_button("Download feature importance CSV", importance.to_csv(index=False), "v8_permutation_importance.csv", "text/csv")
        except (ValueError, ImportError) as exc:
            st.warning(str(exc))


with tabs[8]:
    st.subheader("v9 temporal intelligence")
    st.caption("Topic evolution counts title tokens in UTC time buckets. Snapshot trend projections are simple descriptive linear baselines, not ML forecasts.")
    from hn_temporal_v9 import build_topic_evolution, forecast_snapshot_engagement, backtest_snapshot_trends
    frequency = st.selectbox("Topic time bucket", ["D", "W", "M"], index=1)
    try:
        evolution = build_topic_evolution(df, freq=frequency)
        st.metric("Periods", len(evolution["periods"]))
        evolution_frame = pd.DataFrame(evolution["periods"])
        if not evolution_frame.empty and evolution["terms"]:
            st.line_chart(evolution_frame.set_index("period")[evolution["terms"][:6]])
            st.dataframe(evolution_frame, use_container_width=True)
        st.download_button("Download topic evolution JSON", __import__("json").dumps(evolution, indent=2), "v9_topic_evolution.json", "application/json")
    except ValueError as exc:
        st.warning(str(exc))
    db_path = ROOT / "hn_monitor.sqlite"
    if db_path.exists():
        horizon = st.slider("Projection horizon (hours)", 1, 48, 6)
        if st.button("Calculate snapshot trend projections"):
            try:
                forecasts = forecast_snapshot_engagement(db_path, horizon_hours=horizon)
                st.dataframe(forecasts, use_container_width=True)
                st.download_button("Download projections CSV", forecasts.to_csv(index=False), "v9_engagement_forecasts.csv", "text/csv")
                st.json(backtest_snapshot_trends(db_path))
            except ValueError as exc:
                st.info(str(exc))
    else:
        st.info("Snapshot forecasting requires a database created by `hn-analytics monitor`.")


with tabs[9]:
    st.subheader("v10 chronological model benchmark")
    st.caption("Compares three scikit-learn classifiers on identical chronological holdout rows. Retrospective results are not guaranteed future performance.")
    if st.button("Run v10 model comparison", type="primary"):
        from hn_benchmark_v10 import benchmark_virality_models
        try:
            leaderboard, predictions, metadata = benchmark_virality_models(df)
            st.metric("Top model by holdout ROC-AUC", metadata["best_model"])
            st.dataframe(leaderboard, use_container_width=True)
            st.bar_chart(leaderboard.set_index("model")[["roc_auc", "average_precision", "f1"]])
            st.download_button("Download model leaderboard", leaderboard.to_csv(index=False), "v10_model_leaderboard.csv", "text/csv")
            st.download_button("Download holdout predictions", predictions.to_csv(index=False), "v10_model_predictions.csv", "text/csv")
            st.json(metadata)
        except (ValueError, ImportError) as exc:
            st.warning(str(exc))


with tabs[10]:
    st.subheader("v11 expanding-window validation and calibration")
    st.caption("Repeated chronological holdouts reduce dependence on one split. Fold thresholds are learned from training history only; calibration scores diagnose probability quality and are not post-hoc calibrated.")
    folds_count = st.slider("Walk-forward folds", 2, 5, 3, key="v11_folds")
    test_fraction = st.slider("Test fraction per fold", 0.05, 0.25, 0.15, 0.05, key="v11_test_fraction")
    if st.button("Run v11 walk-forward benchmark", type="primary"):
        from hn_walkforward_v11 import walk_forward_benchmark
        try:
            summary, fold_metrics, predictions, metadata = walk_forward_benchmark(
                df, n_splits=folds_count, test_fraction=test_fraction)
            st.metric("Best mean ROC-AUC", summary.iloc[0].model)
            st.dataframe(summary, use_container_width=True)
            st.subheader("Fold-by-fold performance")
            st.bar_chart(fold_metrics.pivot(index="fold", columns="model", values="roc_auc"))
            st.subheader("Probability calibration diagnostics")
            st.bar_chart(summary.set_index("model")[["brier_score_mean", "log_loss_mean", "expected_calibration_error_mean"]])
            st.download_button("Download walk-forward summary", summary.to_csv(index=False), "v11_walkforward_summary.csv", "text/csv")
            st.download_button("Download fold metrics", fold_metrics.to_csv(index=False), "v11_walkforward_folds.csv", "text/csv")
            st.download_button("Download out-of-fold predictions", predictions.to_csv(index=False), "v11_walkforward_predictions.csv", "text/csv")
            st.json(metadata)
        except (ValueError, ImportError) as exc:
            st.warning(str(exc))


with tabs[11]:
    st.subheader("v12 author/domain cohorts and chronological drift")
    st.caption("Cohort summaries are descriptive, not causal. PSI is a screening statistic and does not prove model degradation.")
    from hn_profile_v12 import cohort_profiles, chronological_drift
    min_stories = st.slider("Minimum stories per cohort", 1, 20, 3, key="v12_min_stories")
    reference_fraction = st.slider("Earlier reference share", 0.2, 0.8, 0.5, 0.05, key="v12_reference_fraction")
    try:
        authors, domains = cohort_profiles(df, min_stories=min_stories)
        drift, drift_meta = chronological_drift(df, reference_fraction=reference_fraction)
        left, right = st.columns(2)
        left.subheader("Author cohorts")
        left.dataframe(authors, use_container_width=True)
        right.subheader("Domain cohorts")
        right.dataframe(domains, use_container_width=True)
        st.subheader("Chronological feature drift")
        st.dataframe(drift, use_container_width=True)
        st.json(drift_meta)
        st.download_button("Download author cohorts", authors.to_csv(index=False), "v12_author_cohorts.csv", "text/csv")
        st.download_button("Download domain cohorts", domains.to_csv(index=False), "v12_domain_cohorts.csv", "text/csv")
        st.download_button("Download drift report", drift.to_csv(index=False), "v12_drift_report.csv", "text/csv")
    except ValueError as exc:
        st.warning(str(exc))


with st.expander("v13 Story recommendations & near-duplicate discovery", expanded=False):
    st.caption("Content-based recommendations use TF-IDF lexical similarity. Near-duplicate matches are review candidates, not confirmed duplicates.")
    from hn_recommend_v13 import recommend_stories, find_near_duplicates
    if len(df) > 1:
        seed_index = st.number_input("Seed story row", min_value=0, max_value=len(df)-1, value=0, step=1, key="v13_seed")
        top_k = st.slider("Recommendation count", 1, 30, 10, key="v13_top_k")
        min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.05, 0.01, key="v13_min_sim")
        if st.button("Generate v13 recommendations", type="primary"):
            try:
                recs = recommend_stories(df, int(seed_index), top_k, min_sim)
                st.write("**Selected story:**", df.iloc[int(seed_index)].get("title", ""))
                st.dataframe(recs, use_container_width=True)
                st.download_button("Download recommendations", recs.to_csv(index=False), "v13_recommendations.csv", "text/csv")
                dup_threshold = st.slider("Near-duplicate threshold", 0.3, 1.0, 0.75, 0.05, key="v13_dup_threshold")
                duplicates = find_near_duplicates(df, dup_threshold)
                st.subheader("Near-duplicate candidates")
                st.dataframe(duplicates, use_container_width=True)
                st.download_button("Download near-duplicate candidates", duplicates.to_csv(index=False), "v13_near_duplicates.csv", "text/csv")
            except ValueError as exc:
                st.warning(str(exc))
    else:
        st.info("Upload at least two stories to generate recommendations.")


with st.expander("v14 Semantic discovery & topic clustering", expanded=False):
    st.caption("Uses Sentence-Transformers embeddings when the optional package and model are available; otherwise auto mode reports and uses the TF-IDF fallback. Clusters are exploratory, not ground-truth topics.")
    from hn_semantic_v14 import semantic_recommendations, discover_topics
    if len(df) >= 2:
        v14_backend = st.selectbox("Embedding backend", ["auto", "sentence-transformers", "tfidf"], key="v14_backend")
        v14_seed = st.number_input("Semantic seed row", min_value=0, max_value=len(df)-1, value=0, key="v14_seed")
        v14_k = st.slider("Semantic recommendation count", 1, min(30, len(df)-1), min(10, len(df)-1), key="v14_k")
        v14_clusters = st.slider("Topic clusters", 2, min(15, len(df)), min(5, len(df)), key="v14_clusters")
        if st.button("Run v14 semantic discovery", type="primary"):
            try:
                recs, rec_meta = semantic_recommendations(df, int(v14_seed), v14_k, backend=v14_backend)
                topics, assignments, topic_meta = discover_topics(df, v14_clusters, backend=v14_backend)
                st.write("**Selected story:**", df.iloc[int(v14_seed)].get("title", ""))
                st.metric("Embedding backend actually used", rec_meta["backend"])
                st.dataframe(recs, use_container_width=True)
                st.download_button("Download semantic recommendations", recs.to_csv(index=False), "v14_semantic_recommendations.csv", "text/csv")
                st.subheader("Exploratory topic clusters")
                st.dataframe(topics, use_container_width=True)
                st.dataframe(assignments, use_container_width=True)
                st.download_button("Download topic assignments", assignments.to_csv(index=False), "v14_topic_assignments.csv", "text/csv")
            except (ValueError, ImportError, RuntimeError) as exc:
                st.warning(str(exc))
    else:
        st.info("Upload at least two stories for semantic discovery.")


with tabs[12]:
    st.subheader("v15 author-topic graph intelligence")
    st.caption("Edges represent observed author participation in exploratory topic clusters. Shared-topic overlap is not evidence of collaboration, influence or causal relationships.")
    from hn_graph_v15 import build_author_topic_graph
    if len(df) >= 3 and ("by" in df.columns or "author" in df.columns):
        max_clusters = min(15, len(df))
        graph_clusters = st.slider("Exploratory topic clusters", 2, max_clusters, min(5, max_clusters), key="v15_clusters")
        graph_backend = st.selectbox("Topic representation", ["tfidf", "auto", "sentence-transformers"], key="v15_backend")
        if st.button("Build author-topic graph", type="primary", key="v15_graph_run"):
            try:
                authors, topics, edges, overlaps, graph, metadata = build_author_topic_graph(
                    df, n_clusters=graph_clusters, backend=graph_backend)
                g1, g2, g3 = st.columns(3)
                g1.metric("Authors", len(authors))
                g2.metric("Topics", len(topics))
                g3.metric("Author-topic links", len(edges))
                left, right = st.columns(2)
                with left:
                    st.subheader("Author graph metrics")
                    st.dataframe(authors, use_container_width=True)
                with right:
                    st.subheader("Topic graph metrics")
                    st.dataframe(topics, use_container_width=True)
                st.subheader("3D author profile")
                if not authors.empty:
                    import plotly.express as px
                    fig = px.scatter_3d(authors, x="topic_count", y="story_count", z="normalized_topic_entropy",
                                        size="story_count", color="mean_score", hover_name="author",
                                        labels={"topic_count": "Distinct topics", "story_count": "Stories",
                                                "normalized_topic_entropy": "Topic entropy", "mean_score": "Mean score"},
                                        title="Author topic breadth, activity and topic distribution")
                    st.plotly_chart(fig, use_container_width=True)
                st.subheader("Author pairs with shared topic distributions")
                st.dataframe(overlaps.head(100), use_container_width=True)
                st.download_button("Download graph JSON", __import__("json").dumps(graph, indent=2),
                                    "v15_author_topic_graph.json", "application/json")
                st.download_button("Download author metrics", authors.to_csv(index=False),
                                    "v15_author_graph_metrics.csv", "text/csv")
                st.download_button("Download author-topic links", edges.to_csv(index=False),
                                    "v15_author_topic_edges.csv", "text/csv")
                st.json(metadata)
            except (ValueError, ImportError, RuntimeError) as exc:
                st.warning(str(exc))
    else:
        st.info("Upload at least three stories with author names to build the author-topic graph.")


with tabs[13]:
    st.subheader("v16-v17 network structure and robustness")
    st.caption("PageRank, overlap communities and bootstrap stability describe structure in the observed author-topic graph only. They are not influence, collaboration or causal measures.")
    if len(df) >= 3 and ("by" in df.columns or "author" in df.columns):
        max_clusters = min(15, len(df))
        n_clusters = st.slider("Exploratory topic clusters", 2, max_clusters, min(5, max_clusters), key="v16_clusters")
        sim_threshold = st.slider("Overlap similarity threshold", 0.0, 1.0, 0.15, 0.05, key="v16_threshold")
        damping = st.slider("PageRank damping", 0.05, 0.95, 0.85, 0.05, key="v16_damping")
        repeats = st.slider("Bootstrap repeats (v17)", 10, 500, 100, 10, key="v17_repeats")
        top_k = st.slider("Top-k for stability (v17)", 1, 15, 5, key="v17_top_k")
        if st.button("Run network analysis", type="primary", key="v16_run"):
            try:
                from hn_network_v16 import build_v16_network
                from hn_robustness_v17 import build_v17_robustness
                authors, topics, communities, graph, meta = build_v16_network(
                    df, n_clusters, "tfidf", sim_threshold, damping)
                stability, sensitivity, bridges, summary = build_v17_robustness(
                    df, n_clusters, "tfidf", repeats, top_k, damping)
                m1, m2, m3 = st.columns(3)
                m1.metric("Overlap communities", meta["community_count"])
                m2.metric("Mean rank correlation", "N/A" if summary["mean_spearman"] is None else f"{summary['mean_spearman']:.2f}")
                m3.metric(f"Mean top-{summary['top_k']} overlap", "N/A" if summary["mean_top_k_overlap"] is None else f"{summary['mean_top_k_overlap']:.2f}")
                st.subheader("Author PageRank with bootstrap stability")
                st.dataframe(stability, use_container_width=True)
                left, right = st.columns(2)
                left.subheader("Damping sensitivity")
                left.dataframe(sensitivity, use_container_width=True)
                right.subheader("Topic bridge scores")
                right.dataframe(bridges, use_container_width=True)
                st.subheader("Overlap communities")
                st.dataframe(communities, use_container_width=True)
                st.download_button("Download author stability", stability.to_csv(index=False), "v17_author_stability.csv", "text/csv")
                st.download_button("Download network graph JSON", __import__("json").dumps(graph, indent=2), "v16_network_graph.json", "application/json")
                st.json(summary)
            except (ValueError, ImportError, RuntimeError) as exc:
                st.warning(str(exc))
    else:
        st.info("Upload at least three stories with author names to run network analysis.")

with tabs[14]:
    st.subheader("Search stories")
    f1, f2, f3, f4 = st.columns(4)
    keyword = f1.text_input("Keyword in title", "", key="v18_keyword")
    min_points = f2.number_input("Minimum points", min_value=0.0, value=0.0, step=10.0, key="v18_min")
    sort_by = f3.selectbox("Sort by", ["score", "comments", "time"], key="v18_sort")
    window = f4.number_input("Last N days (0 = all)", min_value=0.0, value=0.0, step=1.0, key="v18_days")
    hits = search_stories(df, keyword=keyword, min_score=min_points or None, sort=sort_by,
                          days=window or None, limit=50)
    st.caption(f"{len(hits)} matching stories (showing up to 50)")
    st.dataframe(hits, use_container_width=True)
    st.download_button("Download matches CSV", hits.to_csv(index=False).encode(), "v18_matches.csv", "text/csv")

    st.subheader("Digest preview")
    st.markdown(digest_preview(df, keyword=keyword, days=window or None, top_n=10))

    st.subheader("Summary")
    tables = summary_tables(df, top_n=10)
    st.json(tables["summary"])
    left, right = st.columns(2)
    left.dataframe(tables["domains"], use_container_width=True)
    right.dataframe(tables["authors"], use_container_width=True)

    st.subheader("Earlier half vs later half of this dataset")
    try:
        cmp_result = compare_halves(df)
        m1, m2 = st.columns(2)
        m1.metric("Median points change", "N/A" if cmp_result["median_score"]["change"] is None
                  else f"{cmp_result['median_score']['change']:+.1f}")
        m2.metric("Median comments change", "N/A" if cmp_result["median_comments"]["change"] is None
                  else f"{cmp_result['median_comments']['change']:+.1f}")
        st.dataframe(pd.DataFrame(cmp_result["domain_shifts"]), use_container_width=True)
        st.caption("Descriptive only: no significance test and no adjustment for story age or exposure.")
    except ValueError as exc:
        st.info(str(exc))
