"""v18.2 digest: a Markdown briefing of top stories and fastest-rising stories.

``build_digest`` is a pure function over DataFrames so it can be tested without
files. ``export_digest`` loads the dataset and/or SQLite history and writes
``digest.md``. Rankings are descriptive; see the limitations printed in the report.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hn_cleaning import clean_stories, text_or_empty
from hn_data_loader import load_stories
from hn_intelligence import story_momentum

LIMITATIONS = (
    "Rankings are descriptive snapshots of the observed data. Score and comment "
    "counts are confounded by story age and exposure, and rising status describes "
    "change between recorded observations, not a forecast or a measure of quality."
)


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _link_title(title: str, url: str) -> str:
    title = text_or_empty(title) or "(untitled)"
    url = text_or_empty(url)
    return f"[{title}]({url})" if url.startswith(("http://", "https://")) else title


def build_digest(stories: pd.DataFrame | None = None, momentum: pd.DataFrame | None = None,
                 top_n: int = 10, generated_at: datetime | None = None) -> str:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# Hacker News digest", "", f"_Generated {stamp}_", ""]

    lines += ["## Top stories by score", ""]
    if stories is None or stories.empty:
        lines += ["No story data available.", ""]
    else:
        top = stories.assign(score=pd.to_numeric(stories["score"], errors="coerce"))
        top = top.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n)
        if top.empty:
            lines += ["No scored stories available.", ""]
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            domain = text_or_empty(row.get("domain")) or "self/unknown"
            author = text_or_empty(row.get("by")) or "unknown"
            lines.append(f"{rank}. {_link_title(row.get('title'), row.get('url'))} — "
                         f"{_fmt_int(row.get('score'))} points, by {author} ({domain})")
        lines.append("")

    lines += ["## Most discussed stories", ""]
    if stories is None or stories.empty or "descendants" not in stories:
        lines += ["No comment data available.", ""]
    else:
        busy = stories.assign(descendants=pd.to_numeric(stories["descendants"], errors="coerce"))
        busy = busy.dropna(subset=["descendants"]).sort_values("descendants", ascending=False).head(top_n)
        busy = busy[busy["descendants"] > 0]
        if busy.empty:
            lines += ["No stories have comments.", ""]
        for rank, (_, row) in enumerate(busy.iterrows(), start=1):
            lines.append(f"{rank}. {_link_title(row.get('title'), row.get('url'))} — "
                         f"{_fmt_int(row.get('descendants'))} comments")
        lines.append("")

    lines += ["## Fastest rising stories", ""]
    if momentum is None or momentum.empty:
        lines += ["No snapshot history available for momentum analysis.", ""]
    else:
        rising = momentum[momentum["momentum_status"] == "Rising"]
        rising = rising.dropna(subset=["score_per_hour"]).sort_values("score_per_hour", ascending=False).head(top_n)
        if rising.empty:
            lines += ["No stories are rising in the observed window.", ""]
        for rank, (_, row) in enumerate(rising.iterrows(), start=1):
            lines.append(f"{rank}. {_link_title(row.get('title'), row.get('url'))} — "
                         f"+{_fmt_int(row.get('score_gain'))} points "
                         f"({float(row.get('score_per_hour')):.1f}/hour over "
                         f"{float(row.get('elapsed_hours')):.1f} h)")
        lines.append("")

    lines += ["---", "", f"_{LIMITATIONS}_", ""]
    return "\n".join(lines)


_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]*)\)")


def markdown_to_html(markdown: str) -> str:
    """Convert the digest's Markdown subset to a standalone HTML page.

    Every text fragment is HTML-escaped before tags are added, so story titles
    and authors cannot inject markup. Supported: '# ' and '## ' headings, '---'
    rules, '_italic_' lines, numbered list items, and [text](http(s) url) links.
    """
    def inline(text: str) -> str:
        out, pos = [], 0
        for m in _LINK.finditer(text):
            out.append(html.escape(text[pos:m.start()]))
            out.append(f'<a href="{html.escape(m.group(2), quote=True)}" rel="noopener noreferrer">'
                       f'{html.escape(m.group(1))}</a>')
            pos = m.end()
        out.append(html.escape(text[pos:]))
        return "".join(out)

    body, in_list = [], False
    def close_list():
        nonlocal in_list
        if in_list:
            body.append("</ol>"); in_list = False
    for line in markdown.splitlines():
        if re.match(r"^\d+\. ", line):
            if not in_list:
                body.append("<ol>"); in_list = True
            body.append(f"<li>{inline(re.sub(r'^\d+\. ', '', line, count=1))}</li>")
            continue
        close_list()
        if not line.strip():
            continue
        if line.startswith("## "):
            body.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.strip() == "---":
            body.append("<hr>")
        elif line.startswith("_") and line.endswith("_") and len(line) > 2:
            body.append(f"<p class=\"meta\">{inline(line[1:-1])}</p>")
        else:
            body.append(f"<p>{inline(line)}</p>")
    close_list()
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Hacker News digest</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}"
            "h1{border-bottom:2px solid #ff6600}.meta{color:#666;font-size:.9rem}ol li{margin:.4rem 0}"
            "@media (prefers-color-scheme:dark){body{background:#111;color:#ddd}a{color:#ffb070}}"
            "</style></head><body>\n" + "\n".join(body) + "\n</body></html>\n")


def _keyword_filter(df: pd.DataFrame, column: str, terms) -> pd.DataFrame:
    """Keep rows whose column contains every term (case-insensitive, literal)."""
    mask = pd.Series(True, index=df.index)
    for term in terms:
        mask &= df[column].map(text_or_empty).str.contains(re.escape(term), case=False, regex=True)
    return df[mask]


def export_digest(input_path=None, db_path=None, output_dir="reports", top_n: int = 10,
                  lookback_hours: float = 24, fmt: str = "markdown", keywords=(), days=None) -> dict:
    if fmt not in {"markdown", "html"}:
        raise ValueError("fmt must be 'markdown' or 'html'")
    """Load available sources, write digest.md, and return summary metadata."""
    stories = None
    if input_path and Path(input_path).exists():
        stories = clean_stories(load_stories(input_path))
    momentum = None
    if db_path and Path(db_path).exists():
        momentum = story_momentum(db_path, lookback_hours)
    if days is not None and stories is not None:
        from hn_query import window_last_days
        stories = window_last_days(stories, days)
    terms = [k.strip() for k in keywords if k.strip()]
    if terms:
        if stories is not None:
            stories = _keyword_filter(stories, "title", terms)
        if momentum is not None:
            momentum = _keyword_filter(momentum, "title", terms)
    markdown = build_digest(stories, momentum, top_n=top_n)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / ("digest.html" if fmt == "html" else "digest.md")
    path.write_text(markdown_to_html(markdown) if fmt == "html" else markdown, encoding="utf-8")
    return {"path": str(path), "stories": 0 if stories is None else len(stories),
            "momentum_rows": 0 if momentum is None else len(momentum)}
