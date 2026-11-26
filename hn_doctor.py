"""v18.4 doctor: check the local environment and data files before running workflows.

Each check returns (name, status, detail) where status is 'ok', 'warn' or 'fail'.
Only missing required packages or an unsupported Python version are failures.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

REQUIRED = ["pandas", "numpy", "sklearn", "requests"]
OPTIONAL = {
    "matplotlib": "viz", "seaborn": "viz", "plotly": "viz/dashboard",
    "streamlit": "dashboard", "networkx": "ai", "sentence_transformers": "ai (semantic backend)",
    "pyspark": "spark",
}


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_python(version=None):
    v = version or sys.version_info
    ok = tuple(v[:2]) >= (3, 10)
    return ("python", "ok" if ok else "fail", f"{v[0]}.{v[1]}.{v[2]} (need >= 3.10)")


def check_packages():
    results = []
    for mod in REQUIRED:
        results.append((f"required:{mod}", "ok" if _has(mod) else "fail",
                        "installed" if _has(mod) else "missing; run pip install -e ."))
    for mod, extra in OPTIONAL.items():
        results.append((f"optional:{mod}", "ok" if _has(mod) else "warn",
                        "installed" if _has(mod) else f"not installed (needed for {extra})"))
    return results


def check_database(db_path):
    path = Path(db_path)
    if not path.exists():
        return ("database", "warn", f"{path} not found (created by `monitor`)")
    try:
        with sqlite3.connect(path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(story_snapshots)")}
    except sqlite3.DatabaseError as exc:
        return ("database", "fail", f"unreadable: {exc}")
    if not cols:
        return ("database", "warn", "no story_snapshots table yet")
    if "rank" not in cols:
        return ("database", "warn", "older schema without a rank column; will migrate on next write")
    return ("database", "ok", "story_snapshots schema current")


def check_sample(path="demo_stories.jsonl"):
    p = Path(path)
    return ("sample_data", "ok" if p.exists() else "warn",
            f"{p} present" if p.exists() else f"{p} not found; run from the project root")


def run_checks(db_path="hn_monitor.sqlite", sample_path="demo_stories.jsonl"):
    results = [check_python(), *check_packages(), check_database(db_path), check_sample(sample_path)]
    failed = any(status == "fail" for _, status, _ in results)
    return results, failed
