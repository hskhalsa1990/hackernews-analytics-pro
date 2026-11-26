from pathlib import Path
import pandas as pd

def load_stories(path):
    """Load CSV, JSON, JSON Lines or NDJSON; a trailing .gz is decompressed automatically."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    name = path.name.lower()
    base = name[:-3] if name.endswith(".gz") else name
    suffix = Path(base).suffix
    if suffix == ".csv":
        return pd.read_csv(path, compression="infer")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, compression="infer")
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True, compression="infer")
        except ValueError:
            return pd.read_json(path, compression="infer")
    raise ValueError(f"Unsupported file format: {path.suffix}")

def fetch_recent_stories(query="", hits_per_page=100, pages=1):
    """Fetch a bounded set of recent stories from the public Algolia HN Search API."""
    import requests
    if not 1 <= hits_per_page <= 1000 or not 1 <= pages <= 10:
        raise ValueError("Use hits_per_page 1..1000 and pages 1..10.")
    rows = []
    for page in range(pages):
        params = {"tags": "story", "hitsPerPage": hits_per_page, "page": page}
        if query:
            params["query"] = query
        response = requests.get("https://hn.algolia.com/api/v1/search", params=params, timeout=30)
        response.raise_for_status()
        rows.extend(response.json().get("hits", []))
    return pd.DataFrame(rows)
