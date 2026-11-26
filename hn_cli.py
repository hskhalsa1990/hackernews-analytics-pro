"""Command-line interface for collection, monitoring and momentum insights."""
import argparse
import time
from pathlib import Path
from hn_analytics import __version__
from hn_live import HackerNewsClient, save_snapshot, growth_alerts
from hn_intelligence import story_momentum, export_momentum
from hn_data_loader import load_stories
from hn_predictive import export_predictive_research
from hn_research_v8 import export_v8_research
from hn_temporal_v9 import export_v9_temporal
from hn_benchmark_v10 import export_v10_benchmark
from hn_walkforward_v11 import export_v11_walkforward
from hn_profile_v12 import export_v12_profile
from hn_recommend_v13 import export_v13_recommendations
from hn_semantic_v14 import export_v14_semantic
from hn_graph_v15 import export_v15_graph
from hn_network_v16 import export_v16_network
from hn_robustness_v17 import export_v17_robustness
from hn_digest import export_digest
from hn_query import run_query, watch_history
from hn_doctor import run_checks
from hn_server import make_server
from hn_stats import stats_for_file, format_text
from hn_compare import compare_files, format_comparison

FEEDS = ["top", "new", "best", "ask", "show", "job"]

def main(argv=None):
    parser = argparse.ArgumentParser(prog="hn-analytics", description="Hacker News analytics utilities")
    parser.add_argument("--version", action="version", version=f"hn-analytics {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch a bounded live feed to JSONL")
    fetch.add_argument("--feed", choices=FEEDS, default="top"); fetch.add_argument("--limit", type=int, default=100)
    fetch.add_argument("--output", default="live_stories.jsonl")
    monitor = sub.add_parser("monitor", help="Poll a feed, store ranked snapshots and print growth alerts")
    monitor.add_argument("--feed", choices=FEEDS, default="top"); monitor.add_argument("--limit", type=int, default=100)
    monitor.add_argument("--db", default="hn_monitor.sqlite"); monitor.add_argument("--polls", type=int, default=2)
    monitor.add_argument("--interval", type=float, default=300); monitor.add_argument("--min-score-gain", type=int, default=20)
    monitor.add_argument("--min-comment-gain", type=int, default=10); monitor.add_argument("--lookback-hours", type=float, default=24)
    insights = sub.add_parser("insights", help="Export story momentum and rank-change analysis from SQLite")
    insights.add_argument("--db", default="hn_monitor.sqlite"); insights.add_argument("--lookback-hours", type=float, default=24)
    insights.add_argument("--output-dir", default="reports")
    research = sub.add_parser("research", help="Run v7 virality classification and engagement anomaly analysis")
    research.add_argument("--input", default="demo_stories.jsonl")
    research.add_argument("--output-dir", default="reports")
    intelligence = sub.add_parser("intelligence", help="Run v8 explainability, title sentiment and topic-network research")
    intelligence.add_argument("--input", default="demo_stories.jsonl")
    intelligence.add_argument("--output-dir", default="reports")
    intelligence.add_argument("--min-cooccurrence", type=int, default=2)
    intelligence.add_argument("--permutation-repeats", type=int, default=5)
    benchmark = sub.add_parser("benchmark", help="Compare v10 classifiers on a chronological holdout")
    benchmark.add_argument("--input", default="demo_stories.jsonl")
    benchmark.add_argument("--output-dir", default="reports")
    benchmark.add_argument("--test-fraction", type=float, default=0.2)
    walkforward = sub.add_parser("walkforward", help="Run v11 expanding-window validation and calibration diagnostics")
    walkforward.add_argument("--input", default="demo_stories.jsonl")
    walkforward.add_argument("--output-dir", default="reports")
    walkforward.add_argument("--folds", type=int, default=3)
    walkforward.add_argument("--test-fraction", type=float, default=0.15)
    walkforward.add_argument("--bootstrap-repeats", type=int, default=500)
    profile = sub.add_parser("profile", help="Run v12 author/domain cohort profiling and chronological drift screening")
    profile.add_argument("--input", default="demo_stories.jsonl")
    profile.add_argument("--output-dir", default="reports")
    profile.add_argument("--min-stories", type=int, default=3)
    profile.add_argument("--reference-fraction", type=float, default=0.5)
    profile.add_argument("--bins", type=int, default=10)
    digest = sub.add_parser("digest", help="Write a Markdown briefing of top and fastest-rising stories")
    digest.add_argument("--input", default="demo_stories.jsonl")
    digest.add_argument("--db", default="hn_monitor.sqlite")
    digest.add_argument("--output-dir", default="reports")
    digest.add_argument("--top-n", type=int, default=10)
    digest.add_argument("--lookback-hours", type=float, default=24)
    digest.add_argument("--format", choices=["markdown", "html"], default="markdown")
    digest.add_argument("--keyword", action="append", default=[], help="Repeatable; limit digest to matching titles")
    digest.add_argument("--days", type=float, help="Limit stories to the last N days ending at the newest story")
    query = sub.add_parser("query", help="Filter a dataset by keyword, domain, author, score and date")
    query.add_argument("--input", default="demo_stories.jsonl")
    query.add_argument("--keyword", action="append", default=[], help="Repeatable; all must appear in title")
    query.add_argument("--domain"); query.add_argument("--author")
    query.add_argument("--min-score", type=float); query.add_argument("--since", help="UTC date, YYYY-MM-DD")
    query.add_argument("--limit", type=int, default=20); query.add_argument("--output", help="Optional CSV path")
    query.add_argument("--sort", choices=["score", "comments", "time"], default="score")
    query.add_argument("--exclude-keyword", action="append", default=[], help="Repeatable; drop titles containing this term")
    query.add_argument("--count", action="store_true", help="Print only the number of matching stories")
    query.add_argument("--days", type=float, help="Keep stories from the last N days, ending at the newest story")
    query.add_argument("--format", choices=["table", "json", "csv"], default="table")
    watch = sub.add_parser("watch", help="List stored stories matching keywords in the snapshot history")
    watch.add_argument("--db", default="hn_monitor.sqlite"); watch.add_argument("--keyword", action="append", required=True)
    watch.add_argument("--lookback-hours", type=float, default=24)
    watch.add_argument("--format", choices=["table", "json"], default="table")
    watch.add_argument("--output", help="Optional CSV path")
    doctor = sub.add_parser("doctor", help="Check Python, packages, sample data and database schema")
    doctor.add_argument("--db", default="hn_monitor.sqlite")
    doctor.add_argument("--json", action="store_true", help="Print checks as JSON")
    serve = sub.add_parser("serve", help="Run a local read-only web page and JSON API on 127.0.0.1")
    serve.add_argument("--host", default="127.0.0.1"); serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--input", default="demo_stories.jsonl")
    serve.add_argument("--db", default="hn_monitor.sqlite")
    stats = sub.add_parser("stats", help="Summarise a dataset: counts, quantiles, top domains and authors")
    stats.add_argument("--input", default="demo_stories.jsonl")
    stats.add_argument("--top-n", type=int, default=5)
    stats.add_argument("--format", choices=["text", "json"], default="text")
    compare = sub.add_parser("compare", help="Compare two story datasets: medians and domain share shifts")
    compare.add_argument("--baseline", required=True); compare.add_argument("--candidate", required=True)
    compare.add_argument("--top-n", type=int, default=5)
    compare.add_argument("--format", choices=["text", "json"], default="text")
    recommend = sub.add_parser("recommend", help="Run v13 content-based story recommendations and near-duplicate discovery")
    recommend.add_argument("--input", default="demo_stories.jsonl")
    recommend.add_argument("--output-dir", default="reports")
    recommend.add_argument("--seed-index", type=int, default=0)
    recommend.add_argument("--top-k", type=int, default=10)
    recommend.add_argument("--min-similarity", type=float, default=0.05)
    recommend.add_argument("--duplicate-threshold", type=float, default=0.75)
    semantic = sub.add_parser("semantic", help="Run v14 semantic recommendations and unsupervised topic discovery")
    semantic.add_argument("--input", default="demo_stories.jsonl")
    semantic.add_argument("--output-dir", default="reports")
    semantic.add_argument("--seed-index", type=int, default=0)
    semantic.add_argument("--top-k", type=int, default=10)
    semantic.add_argument("--min-similarity", type=float, default=0.05)
    semantic.add_argument("--diversity", type=float, default=0.22)
    semantic.add_argument("--clusters", type=int, default=5)
    semantic.add_argument("--backend", choices=["auto", "sentence-transformers", "tfidf"], default="auto")
    semantic.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    graph = sub.add_parser("graph", help="Run v15 author-topic graph intelligence and overlap analysis")
    graph.add_argument("--input", default="demo_stories.jsonl")
    graph.add_argument("--output-dir", default="reports")
    graph.add_argument("--clusters", type=int, default=5)
    graph.add_argument("--backend", choices=["auto", "sentence-transformers", "tfidf"], default="tfidf")
    network = sub.add_parser("network", help="Run v16 weighted PageRank and author overlap communities")
    network.add_argument("--input", default="demo_stories.jsonl")
    network.add_argument("--output-dir", default="reports")
    network.add_argument("--clusters", type=int, default=5)
    network.add_argument("--backend", choices=["auto", "sentence-transformers", "tfidf"], default="tfidf")
    network.add_argument("--similarity-threshold", type=float, default=0.15)
    network.add_argument("--damping", type=float, default=0.85)
    robust = sub.add_parser("robustness", help="Run v17 bootstrap rank stability, damping sensitivity and bridge scores")
    robust.add_argument("--input", default="demo_stories.jsonl")
    robust.add_argument("--output-dir", default="reports")
    robust.add_argument("--clusters", type=int, default=5)
    robust.add_argument("--backend", choices=["auto", "sentence-transformers", "tfidf"], default="tfidf")
    robust.add_argument("--repeats", type=int, default=200)
    robust.add_argument("--top-k", type=int, default=5)
    robust.add_argument("--damping", type=float, default=0.85)
    robust.add_argument("--seed", type=int, default=42)
    temporal = sub.add_parser("temporal", help="Run v9 topic evolution and optional snapshot trend forecasting")
    temporal.add_argument("--input", default="demo_stories.jsonl")
    temporal.add_argument("--db", default=None, help="Optional SQLite database created by monitor")
    temporal.add_argument("--output-dir", default="reports")
    temporal.add_argument("--frequency", default="W", help="Pandas time frequency, e.g. D, W or M")
    temporal.add_argument("--horizon-hours", type=float, default=6.0)
    args = parser.parse_args(argv)
    if args.command == "robustness":
        raw = load_stories(args.input)
        result = export_v17_robustness(raw, args.output_dir, args.clusters, args.backend,
                                       args.repeats, args.top_k, args.damping, args.seed)
        meta = result["metadata"]
        print("v17 network robustness analysis complete:")
        print(f"Authors: {result['authors']}; bootstrap repeats: {meta['repeats_completed']}/{meta['repeats_requested']}")
        print(f"Mean Spearman vs base: {meta['mean_spearman']}; mean top-{meta['top_k']} overlap: {meta['mean_top_k_overlap']}")
        print(f"Author stability: {result['authors_file']}")
        print(f"Summary: {result['summary_file']}")
        return 0
    if args.command == "network":
        raw = load_stories(args.input)
        result = export_v16_network(raw, args.output_dir, args.clusters, args.backend,
                                    args.similarity_threshold, args.damping)
        print("v16 network structure analysis complete:")
        print(f"Authors: {result['authors']}; topics: {result['topics']}; overlap communities: {result['communities']}")
        print(f"Author ranking: {result['authors_file']}")
        print(f"Graph: {result['graph_file']}")
        print(f"Summary: {result['summary_file']}")
        return 0
    if args.command == "graph":
        raw = load_stories(args.input)
        result = export_v15_graph(raw, args.output_dir, args.clusters, args.backend)
        print("v15 author-topic graph intelligence complete:")
        print(f"Authors: {result['authors']}; topics: {result['topics']}; author-topic edges: {result['edges']}")
        print(f"Graph: {result['graph_file']}")
        print(f"Summary: {result['summary_file']}")
        return 0
    if args.command == "semantic":
        raw = load_stories(args.input)
        result = export_v14_semantic(raw, args.output_dir, args.seed_index, args.top_k,
                                     args.min_similarity, args.diversity, args.clusters,
                                     args.backend, args.model)
        print("v14 semantic intelligence complete:")
        print(f"Embedding backend: {result['metadata']['recommendation']['backend']}")
        print(f"Recommendations: {result['recommendations']}; topic clusters: {result['topics']}")
        print(f"Summary: {result['summary_file']}")
        return 0
    if args.command == "recommend":
        raw = load_stories(args.input)
        result = export_v13_recommendations(raw, args.output_dir, args.seed_index,
                                            args.top_k, args.min_similarity,
                                            args.duplicate_threshold)
        print("v13 story recommendation research complete:")
        print(f"Recommendations: {result['recommendations']}; near-duplicate pairs: {result['duplicates']}")
        print(f"Recommendations: {result['recommendations_file']}")
        print(f"Near duplicates: {result['duplicates_file']}")
        print(f"Summary: {result['summary_file']}")
        return 0
    if args.command == "profile":
        raw = load_stories(args.input)
        result = export_v12_profile(raw, args.output_dir, args.min_stories, args.reference_fraction, args.bins)
        print("v12 cohort and drift profiling complete:")
        print(f"Author cohorts: {result['authors']}; domain cohorts: {result['domains']}")
        print(f"Drift report: {result['drift_file']}")
        print(f"Author report: {result['authors_file']}")
        print(f"Domain report: {result['domains_file']}")
        return 0
    if args.command == "walkforward":
        raw = load_stories(args.input)
        result = export_v11_walkforward(raw, args.output_dir, n_splits=args.folds,
                                        test_fraction=args.test_fraction,
                                        bootstrap_repeats=args.bootstrap_repeats)
        print("v11 walk-forward validation complete:")
        print(f"Models: {result['model_count']}; folds: {result['folds']}; OOF predictions: {result['oof_predictions']}")
        print(f"Summary: {result['summary_file']}")
        print(f"Fold metrics: {result['folds_file']}")
        print(f"Predictions: {result['predictions_file']}")
        return 0
    if args.command == "benchmark":
        raw = load_stories(args.input)
        result = export_v10_benchmark(raw, args.output_dir, args.test_fraction)
        print("v10 model benchmark complete:")
        print(f"Models: {result['models']}; best by holdout ROC-AUC: {result['best_model']}")
        print(f"Leaderboard: {result['leaderboard_file']}")
        print(f"Predictions: {result['predictions_file']}")
        return 0
    if args.command == "temporal":
        raw = load_stories(args.input)
        result = export_v9_temporal(raw, args.output_dir, args.db, args.frequency, args.horizon_hours)
        print("v9 temporal intelligence complete:")
        print(f"Topic periods: {result['topic_periods']}; terms: {result['topic_terms']}")
        if "forecast_rows" in result:
            print(f"Story forecasts: {result['forecast_rows']}")
        print(f"Summary: {Path(args.output_dir) / 'v9_temporal_summary.json'}")
        return 0
    if args.command == "intelligence":
        raw = load_stories(args.input)
        result = export_v8_research(raw, args.output_dir, args.min_cooccurrence, args.permutation_repeats)
        print("v8 intelligence research complete:")
        print(f"Stories: {result['stories']}; topic graph: {result['topic_nodes']} nodes / {result['topic_edges']} edges")
        print(f"Top permutation-importance feature: {result['importance_top_feature']}")
        print(f"Summary: {Path(args.output_dir) / 'v8_research_summary.json'}")
        return 0
    if args.command == "research":
        raw = load_stories(args.input)
        result = export_predictive_research(raw, args.output_dir)
        print("Predictive research complete:")
        print(f"Metrics: {result['metrics_file']}")
        print(f"Predictions: {result['predictions_file']} ({result['predictions']} holdout rows)")
        print(f"Anomalies: {result['anomalies_file']} ({result['anomalies']} flagged)")
        print(f"Holdout ROC-AUC: {result['metrics']['roc_auc']:.3f}")
        return 0
    if args.command == "insights":
        result = export_momentum(args.db, args.output_dir, args.lookback_hours)
        print(f"Analysed {result['rows']} stories; CSV: {result['csv']}; JSON: {result['json']}")
        return 0
    if args.command == "compare":
        if args.top_n < 1:
            parser.error("--top-n must be >=1")
        for path in (args.baseline, args.candidate):
            if not Path(path).exists():
                parser.error(f"dataset not found: {path}")
        try:
            result = compare_files(args.baseline, args.candidate, args.top_n)
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            import json
            print(json.dumps(result, indent=2))
        else:
            print(format_comparison(result))
        return 0
    if args.command == "stats":
        if args.top_n < 1:
            parser.error("--top-n must be >=1")
        if not Path(args.input).exists():
            parser.error(f"dataset not found: {args.input}")
        summary = stats_for_file(args.input, args.top_n)
        if args.format == "json":
            import json
            print(json.dumps(summary, indent=2))
        else:
            print(format_text(summary))
        return 0
    if args.command == "serve":
        try:
            server = make_server(args.host, args.port, args.input, args.db)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Serving on http://{args.host}:{server.server_address[1]}/ (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    if args.command == "doctor":
        results, failed = run_checks(args.db)
        if args.json:
            import json
            print(json.dumps({"ok": not failed, "checks": [
                {"name": n, "status": s, "detail": d} for n, s, d in results]}, indent=2))
            return 1 if failed else 0
        for name, status, detail in results:
            print(f"[{status.upper():4}] {name}: {detail}")
        print("Environment has blocking problems." if failed else "No blocking problems found.")
        return 1 if failed else 0
    if args.command == "query":
        if args.limit < 1:
            parser.error("--limit must be >=1")
        if args.days is not None and args.days <= 0:
            parser.error("--days must be >0")
        # --count must not be capped by --limit, so counting uses an effectively unbounded limit.
        result = run_query(args.input, keywords=args.keyword, domain=args.domain, author=args.author,
                           min_score=args.min_score, since=args.since,
                           limit=10**9 if args.count else args.limit, sort=args.sort,
                           exclude=args.exclude_keyword, days=args.days)
        if args.count:
            print(len(result))
            return 0
        if args.format == "csv" and not args.output:
            print(result.to_csv(index=False), end="")
            return 0
        if args.format == "json" and not args.output:
            import json
            records = json.loads(result.to_json(orient="records", date_format="iso"))
            print(json.dumps(records, indent=2))
            return 0
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"Wrote {len(result)} rows to {args.output}")
        else:
            print(result.to_string(index=False) if not result.empty else "No matching stories.")
        return 0
    if args.command == "watch":
        if args.lookback_hours <= 0:
            parser.error("--lookback-hours must be >0")
        result = watch_history(args.db, args.keyword, args.lookback_hours)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"Wrote {len(result)} rows to {args.output}")
        elif args.format == "json":
            import json
            print(json.dumps(json.loads(result.to_json(orient="records", date_format="iso")), indent=2))
        else:
            print(result.to_string(index=False) if not result.empty else "No stored stories match these keywords.")
        return 0
    if args.command == "digest":
        if args.top_n < 1 or args.lookback_hours <= 0:
            parser.error("--top-n must be >=1 and --lookback-hours >0")
        if args.days is not None and args.days <= 0:
            parser.error("--days must be >0")
        result = export_digest(args.input, args.db, args.output_dir, args.top_n, args.lookback_hours, args.format,
                                keywords=args.keyword, days=args.days)
        print(f"Digest: {result['path']} ({result['stories']} stories, {result['momentum_rows']} momentum rows)")
        return 0
    client = HackerNewsClient()
    if args.command == "fetch":
        raw = client.fetch_feed(args.feed, args.limit)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        raw.to_json(args.output, orient="records", lines=True)
        print(f"Saved {len(raw)} stories to {args.output}")
        if client.failures:
            print(f"Skipped {len(client.failures)} items after retries (see client.failures)")
        return 0
    if args.polls < 1 or args.interval < 0 or args.lookback_hours <= 0:
        parser.error("--polls must be >=1, --interval >=0 and --lookback-hours >0")
    for i in range(args.polls):
        raw = client.fetch_feed(args.feed, args.limit)
        count = save_snapshot(args.db, raw)
        print(f"Poll {i+1}/{args.polls}: stored {count} observations"
              + (f"; skipped {len(client.failures)} failed items" if client.failures else ""))
        if i + 1 < args.polls: time.sleep(args.interval)
    alerts = growth_alerts(args.db, min_score_gain=args.min_score_gain,
                           min_comment_gain=args.min_comment_gain, lookback_hours=args.lookback_hours)
    print("Growth alerts:")
    print(alerts.to_string(index=False) if not alerts.empty else "No threshold crossings in stored window.")
    momentum = story_momentum(args.db, args.lookback_hours)
    if not momentum.empty:
        print("\nTop story momentum:")
        print(momentum.head(10).to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
