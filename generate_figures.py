"""Regenerate all README figures from demo_stories.jsonl.

Usage:  python generate_figures.py [input.jsonl]
Figures are written next to this script as fig_*.png. The bundled demo data is synthetic.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from hn_data_loader import load_stories
from hn_cleaning import clean_stories
from hn_benchmark_v10 import benchmark_virality_models
from hn_network_v16 import build_v16_network

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "demo_stories.jsonl"
NOTE = "synthetic demo data" if SRC.name == "demo_stories.jsonl" else SRC.name
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2563eb", "#f97316", "#16a34a", "#7c3aed", "#64748b"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})
df = clean_stories(load_stories(SRC))


def save(fig, name):
    fig.savefig(HERE / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1. Architecture diagram ------------------------------------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=color, ec="white", lw=1.5, alpha=0.95))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.4))
    ax.text(6, 5.95, "HackerNews Analytics Pro - architecture", ha="center", fontsize=14, fontweight="bold")
    box(0.2, 4.2, 2.2, 1.0, "Sources\nJSONL / JSON / CSV\nlive HN API", GREY)
    box(3.0, 4.2, 2.2, 1.0, "hn_data_loader\nhn_live (SQLite\nsnapshots)", BLUE)
    box(5.8, 4.2, 2.2, 1.0, "hn_cleaning\nquality report", BLUE)
    box(8.6, 4.2, 3.1, 1.0, "hn_analytics (API)\nhn_analytics_core\nhn_advanced", BLUE)
    arrow(2.4, 4.7, 3.0, 4.7); arrow(5.2, 4.7, 5.8, 4.7); arrow(8.0, 4.7, 8.6, 4.7)
    cols = [(0.2, "NLP & topics", "hn_nlp\nhn_semantic_v14", PURPLE),
            (2.55, "Prediction", "hn_predictive\nhn_benchmark_v10\nhn_walkforward_v11", ORANGE),
            (4.9, "Temporal &\ncohorts", "hn_temporal_v9\nhn_profile_v12", GREEN),
            (7.25, "Graph & network", "hn_graph_v15\nhn_network_v16\nhn_robustness_v17", PURPLE),
            (9.6, "Recommend &\nresearch", "hn_recommend_v13\nhn_research_v8", ORANGE)]
    for x, t, s, c in cols:
        box(x, 2.15, 2.15, 1.35, t + "\n" + s, c)
        arrow(10.1, 4.2, x + 1.07, 3.5)
    box(0.2, 0.2, 3.4, 1.0, "Interfaces: hn_cli (hn-analytics)\nhn_server  hn_dashboard (Streamlit)", GREY)
    box(4.0, 0.2, 3.4, 1.0, "Outputs: hn_reporting\nhn_digest  hn_visualization  reports/", GREEN)
    box(7.8, 0.2, 3.9, 1.0, "Checks: hn_doctor  hn_stats  hn_compare\nhn_query  161 unit tests", BLUE)
    for x in (1.9, 5.7, 9.75): arrow(x, 2.15, x, 1.2)
    save(fig, "fig_01_architecture.png")


# 2. 3D scatter --------------------------------------------------------------------
def scatter3d():
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(df["hour_utc"], df["descendants"], df["score"], c=df["score"], cmap="viridis",
                   s=df["title_word_count"] * 9, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Comments"); ax.set_zlabel("Points")
    ax.set_title(f"3D engagement map: hour x comments x points\n(bubble size = title words, {NOTE})")
    ax.view_init(elev=22, azim=-58); fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1, label="Points")
    save(fig, "fig_02_3d_engagement_scatter.png")


# 3. 3D bars weekday x time-of-day --------------------------------------------------
def bars3d():
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d = df.assign(bucket=pd.cut(df["hour_utc"], [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]))
    pv = d.pivot_table(index="weekday_utc", columns="bucket", values="score", aggfunc="mean", observed=False)
    pv = pv.reindex([w for w in order if w in pv.index]).fillna(0)
    fig = plt.figure(figsize=(9.5, 7)); ax = fig.add_subplot(111, projection="3d")
    xs, ys = np.meshgrid(np.arange(pv.shape[0]), np.arange(pv.shape[1]), indexing="ij")
    vals = pv.values.ravel(); colors = plt.cm.plasma((vals - vals.min()) / (np.ptp(vals) + 1e-9))
    ax.bar3d(xs.ravel(), ys.ravel(), np.zeros_like(vals), 0.6, 0.6, vals, color=colors, shade=True)
    ax.set_xticks(np.arange(pv.shape[0]) + 0.3); ax.set_xticklabels([w[:3] for w in pv.index])
    ax.set_yticks(np.arange(pv.shape[1]) + 0.3); ax.set_yticklabels(pv.columns.astype(str))
    ax.set_xlabel("Weekday"); ax.set_ylabel("UTC hours"); ax.set_zlabel("Mean points")
    ax.set_title(f"3D bars: mean points by weekday and time of day ({NOTE})"); ax.view_init(elev=26, azim=-52)
    save(fig, "fig_03_3d_weekday_hour_bars.png")


# 4. Distributions -----------------------------------------------------------------
def distributions():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    thr = df["score"].quantile(0.75)
    axs[0].hist(df["score"], bins=18, color=BLUE, edgecolor="white"); axs[0].axvline(thr, color=ORANGE, ls="--", label=f"75th pct = {thr:.0f}")
    axs[0].set(title="Points distribution", xlabel="Points", ylabel="Stories"); axs[0].legend()
    axs[1].hist(df["descendants"], bins=18, color=GREEN, edgecolor="white"); axs[1].set(title="Comments distribution", xlabel="Comments")
    axs[2].boxplot([df.loc[df.weekday_utc == w, "score"] for w in df.weekday_utc.unique()], tick_labels=[w[:3] for w in df.weekday_utc.unique()])
    axs[2].set(title="Points by weekday", ylabel="Points")
    fig.suptitle(f"Distributions ({NOTE})", fontweight="bold", y=1.03)
    save(fig, "fig_04_distributions.png")


# 5. Top domains / authors ---------------------------------------------------------
def tops():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    g = df.groupby("domain").agg(n=("id", "size"), pts=("score", "mean")).sort_values("n").tail(8)
    axs[0].barh(g.index, g["n"], color=BLUE); axs[0].set(title="Top domains by story count", xlabel="Stories")
    for i, (n, p) in enumerate(zip(g["n"], g["pts"])): axs[0].text(n + 0.1, i, f"avg {p:.0f} pts", va="center", fontsize=8)
    a = df.groupby("by").agg(n=("id", "size"), pts=("score", "sum")).sort_values("pts").tail(8)
    axs[1].barh(a.index, a["pts"], color=ORANGE); axs[1].set(title="Top authors by total points", xlabel="Total points")
    save(fig, "fig_05_top_domains_authors.png")


# 6. Daily trend -------------------------------------------------------------------
def trend():
    d = df.groupby("date").agg(stories=("id", "size"), points=("score", "sum"), comments=("descendants", "sum"))
    d.index = pd.to_datetime(d.index)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(d.index, d["stories"], color=BLUE, alpha=0.35, label="Stories / day"); ax.set_ylabel("Stories")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(d.index, d["points"].rolling(3, min_periods=1).mean(), color=ORANGE, lw=2.2, marker="o", label="Points (3-day avg)")
    ax2.plot(d.index, d["comments"].rolling(3, min_periods=1).mean(), color=GREEN, lw=2.2, marker="s", label="Comments (3-day avg)")
    ax2.set_ylabel("Points / comments"); fig.autofmt_xdate()
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=8)
    ax.set_title(f"Daily activity trend ({NOTE})"); save(fig, "fig_06_daily_trend.png")


# 7. Correlation heatmap -----------------------------------------------------------
def heatmap():
    cols = ["score", "descendants", "title_word_count", "title_char_count", "hour_utc"]
    c = df[cols].corr(); fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right"); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)): ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, shrink=0.8); ax.set_title(f"Correlation heatmap ({NOTE})")
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "fig_07_correlation_heatmap.png")


# 8. Model comparison ---------------------------------------------------------------
def models():
    res = benchmark_virality_models(df)[0].set_index("model")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics)); w = 0.25; fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (m, c) in enumerate(zip(res.index, [BLUE, ORANGE, GREEN])):
        ax.bar(x + (i - 1) * w, res.loc[m, metrics], w, label=m, color=c)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title(f"Virality model comparison - same chronological holdout ({NOTE}, {int(res.test_rows.iloc[0])} test rows)"); ax.legend()
    save(fig, "fig_08_model_comparison.png")
    return res


# 9. Network ------------------------------------------------------------------------
def network():
    authors, topics, comm, graph, summ = build_v16_network(df)
    a = authors.sort_values("pagerank").tail(10)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    axs[0].barh(a["author"], a["pagerank"], color=PURPLE); axs[0].set(title="Author PageRank (top 10)", xlabel="PageRank")
    t = topics.sort_values("topic_cluster")
    axs[1].bar(t["topic_cluster"].astype(str), t["story_count"], color=GREEN, label="Stories")
    ax2 = axs[1].twinx(); ax2.spines["right"].set_visible(True); ax2.plot(t["topic_cluster"].astype(str), t["mean_score"], color=ORANGE, marker="o", lw=2, label="Mean points")
    axs[1].set(title="Topic clusters: size vs mean points", xlabel="Topic cluster", ylabel="Stories"); ax2.set_ylabel("Mean points")
    fig.suptitle(f"Network structure ({NOTE})", fontweight="bold", y=1.03); save(fig, "fig_09_network_pagerank_topics.png")


# 10. Author-topic graph picture ----------------------------------------------------
def graph_picture():
    authors, topics, comm, graph, summ = build_v16_network(df)
    top = authors.sort_values("pagerank").tail(12)
    tc = sorted(topics["topic_cluster"].tolist())
    pos = {}
    for i, au in enumerate(top["author"]): pos[("a", au)] = (0, i / (len(top) - 1))
    for i, t in enumerate(tc): pos[("t", t)] = (1, i / max(len(tc) - 1, 1))
    from hn_graph_v15 import build_author_topic_graph
    edges = build_author_topic_graph(df)[2]
    ac, tcn, wc = "author_name", "topic_cluster", "story_count"
    fig, ax = plt.subplots(figsize=(8, 7)); ax.axis("off")
    for _, r in edges.iterrows():
        k1, k2 = ("a", r[ac]), ("t", r[tcn])
        if k1 in pos and k2 in pos:
            ax.plot([pos[k1][0], pos[k2][0]], [pos[k1][1], pos[k2][1]], color=GREY, alpha=0.5, lw=0.6 + float(r[wc]) * 0.9)
    for (kind, name), (x, y) in pos.items():
        ax.scatter(x, y, s=420 if kind == "t" else 260, color=PURPLE if kind == "t" else BLUE, zorder=3, edgecolor="white")
        ax.text(x + (0.06 if kind == "t" else -0.04), y, (f"topic {name}" if kind == "t" else str(name)), ha="left" if kind == "t" else "right", va="center", fontsize=8)
    ax.set_xlim(-0.5, 1.4); ax.set_title(f"Author - topic graph (top 12 authors by PageRank, {NOTE})\nline thickness = stories by that author in the topic")
    save(fig, "fig_10_author_topic_graph.png")


if __name__ == "__main__":
    architecture(); scatter3d(); bars3d(); distributions(); tops(); trend(); heatmap()
    res = models(); network(); graph_picture()
    print(res.round(3).to_markdown() if hasattr(res, "to_markdown") else res.round(3))
