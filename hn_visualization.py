import matplotlib.pyplot as plt
import seaborn as sns

def plot_overview(df, analyses):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if not analyses["daily"].empty:
        axes[0, 0].plot(analyses["daily"]["created_at"], analyses["daily"]["story_count"])
    axes[0, 0].set_title("Story volume over time")
    if df["score"].notna().any():
        upper = df["score"].quantile(.99)
        sns.histplot(df["score"].dropna().clip(upper=upper), bins=35, ax=axes[0, 1])
    axes[0, 1].set_title("Score distribution (clipped at 99th percentile)")
    top = analyses["domains"].head(10).sort_values("story_count")
    if not top.empty:
        axes[1, 0].barh(top["domain"], top["story_count"])
    axes[1, 0].set_title("Top domains by story count")
    if not analyses["hourly"].empty:
        axes[1, 1].bar(analyses["hourly"]["hour_utc"], analyses["hourly"]["mean_score"])
    axes[1, 1].set_title("Average score by UTC hour")
    axes[1, 1].set_xlabel("Hour (UTC)")
    fig.tight_layout()
    return fig


def plot_3d_engagement(df):
    """3D score/comments/title-length view; returns a Matplotlib figure."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    data = df[["score", "descendants", "title_word_count"]].dropna()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    if not data.empty:
        ax.scatter(data["score"], data["descendants"], data["title_word_count"],
                   c=data["score"], cmap="viridis", alpha=.7)
    ax.set_xlabel("Points")
    ax.set_ylabel("Comments")
    ax.set_zlabel("Title words")
    ax.set_title("3D story engagement landscape")
    fig.tight_layout()
    return fig
