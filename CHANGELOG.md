# Changelog

Newest first. The README carries the same notes with usage examples.

## 18.19.0
- Dashboard tab "Search & compare v18" (search, digest preview, summary, earlier-vs-later comparison).
- Logic in `dashboard_v18.py`, unit-tested without Streamlit.

## 18.18.0
- `digest --days N` window (shared with `query --days`).

## 18.17.0
- `query --days N` relative window anchored to the newest story.

## 18.16.0
- `hn-analytics compare` for two story datasets (medians and domain share shifts).

## 18.15.0
- `hn-analytics stats` dataset summary (text or JSON).

## 18.14.0
- `digest --keyword` topic filter.
- `query --format csv` to stdout.

## 18.13.0
- `query --exclude-keyword` (repeatable) and `query --count`.

## 18.12.0
- `watch --format json` and `watch --output` CSV.
- `doctor --json`.

## 18.11.0
- `query --sort` (score, comments, time) and `--format json`.
- Digest "Most discussed stories" section.

## 18.10.0
- Installable wheel built and verified in a clean virtual environment.

## 18.9.0
- Holdout permutation test for classifier ROC-AUC (p-value and null 95th percentile).

## 18.8.0
- Compressed inputs (`.csv.gz`, `.json.gz`, `.jsonl.gz`, `.ndjson.gz`).
- Keyword and minimum-score filter on the `serve` home page, escaped output.

## 18.7.0
- Fix: `/api/query` limit capped at 500; non-finite and non-numeric `min_score` rejected.

## 18.6.0
- `hn-analytics serve`: loopback-only digest page and JSON API (standard library).

## 18.5.0
- `digest --format html` with escaped, dark-mode-aware output.

## 18.4.0
- `hn-analytics doctor` environment and data checks; CI runs CLI smoke tests.

## 18.3.0
- `hn-analytics query` (keyword, domain, author, score, date filters) and `watch`.

## 18.2.0
- `hn-analytics digest` Markdown briefing.

## 18.1.0
- Live client: retries with backoff, per-item failure isolation, bounded concurrency; `--version`.

## 18.0.0
- Fix: id-less rows no longer collapse during cleaning.
- Fix: missing values no longer stored as the string `"nan"`.
