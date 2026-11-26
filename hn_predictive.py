"""Leakage-aware story outcome classification and engagement anomaly detection.

The classifier estimates whether a story will land in the high-score group for
this dataset. It is a retrospective baseline, not a guarantee of future reach.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["title", "domain", "type", "author", "hour_utc", "weekday_utc",
                   "title_word_count", "title_char_count"]


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "created_at" in data:
        dt = pd.to_datetime(data["created_at"], utc=True, errors="coerce")
    elif "time" in data:
        dt = pd.to_datetime(pd.to_numeric(data["time"], errors="coerce"), unit="s", utc=True, errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    title = data.get("title", pd.Series("", index=data.index)).fillna("").astype(str)
    url = data.get("url", pd.Series("", index=data.index)).fillna("").astype(str)
    from urllib.parse import urlparse
    domain = data.get("domain", url.map(lambda value: urlparse(value).netloc.lower().removeprefix("www.")))
    result = pd.DataFrame(index=data.index)
    result["title"] = title
    result["domain"] = domain.fillna("unknown").astype(str).replace("", "unknown")
    result["type"] = data.get("type", pd.Series("unknown", index=data.index)).fillna("unknown").astype(str)
    result["author"] = data.get("by", data.get("author", pd.Series("unknown", index=data.index))).fillna("unknown").astype(str)
    result["hour_utc"] = dt.dt.hour.fillna(-1).astype(int)
    result["weekday_utc"] = dt.dt.dayofweek.fillna(-1).astype(int)
    result["title_word_count"] = title.str.findall(r"\b[\w'-]+\b").str.len().astype(float)
    result["title_char_count"] = title.str.len().astype(float)
    return result[FEATURE_COLUMNS]


def _permutation_auc(y_true, probabilities, permutations: int, seed: int):
    """Holdout label-permutation test for ROC-AUC.

    Shuffles the holdout labels and recomputes AUC to build a null distribution.
    p = (1 + count(null >= observed)) / (1 + permutations). This checks the fitted
    model against chance on this holdout only; it does not correct for the
    model selection or tuning that preceded it.
    """
    from sklearn.metrics import roc_auc_score
    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    observed = float(roc_auc_score(y_true, probabilities))
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(y_true), probabilities) for _ in range(permutations)])
    p_value = float((1 + np.sum(null >= observed)) / (1 + permutations))
    return observed, p_value, float(np.percentile(null, 95))


def train_virality_model(df: pd.DataFrame, min_rows: int = 40, high_score_quantile: float = 0.75,
                         permutations: int = 200, seed: int = 42):
    """Train a chronological RandomForest classifier without score leakage.

    The high-score threshold is learned from training rows only, then applied
    unchanged to the holdout. Scores/comments are targets, never features.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                     precision_score, recall_score, roc_auc_score)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc
    data = df.copy()
    if "score" not in data:
        raise ValueError("Dataset must include a score column")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["score"])
    if len(data) < min_rows:
        raise ValueError(f"Need at least {min_rows} scored stories; found {len(data)}")
    if not 0.5 < high_score_quantile < 1:
        raise ValueError("high_score_quantile must be between 0.5 and 1")
    x = _features(data)
    if "created_at" in data:
        order = pd.to_datetime(data.created_at, utc=True, errors="coerce").sort_values(kind="stable").index
    elif "time" in data:
        order = pd.to_numeric(data.time, errors="coerce").sort_values(kind="stable").index
    else:
        order = data.index
    x = x.loc[order]
    scores = data.loc[order, "score"].clip(lower=0)
    split = min(max(int(len(x) * 0.8), 1), len(x) - 1)
    threshold = float(scores.iloc[:split].quantile(high_score_quantile))
    y = (scores >= threshold).astype(int)
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Chronological split does not contain both outcome classes; use a larger or more varied dataset")
    categorical = ["domain", "type", "author", "weekday_utc"]
    numeric = ["hour_utc", "title_word_count", "title_char_count"]
    prep = ColumnTransformer([
        ("title_tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2), stop_words="english"), "title"),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
    model = Pipeline([("features", prep), ("classifier", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))])
    model.fit(x.iloc[:split], y_train)
    probabilities = model.predict_proba(x.iloc[split:])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    auc_observed, auc_p, auc_null_95 = _permutation_auc(y_test.to_numpy(), probabilities, permutations, seed)
    metrics = {
        "train_rows": int(split), "test_rows": int(len(y_test)),
        "high_score_threshold": threshold, "threshold_quantile": high_score_quantile,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "roc_auc_permutation_p_value": auc_p,
        "roc_auc_null_95th_percentile": auc_null_95,
        "permutations": int(permutations),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "split": "chronological 80/20", "target": f"score >= training {high_score_quantile:.0%} quantile",
        "predictors": FEATURE_COLUMNS,
    }
    predictions = pd.DataFrame({
        "actual_high_score": y_test.to_numpy(), "predicted_high_score": predicted,
        "high_score_probability": probabilities,
        "actual_score": scores.iloc[split:].to_numpy(),
    }, index=x.index[split:]).rename_axis("source_index").reset_index()
    return model, metrics, predictions


def detect_engagement_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Flag unusual score/comment/title-length combinations with IsolationForest."""
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    data = df.copy().reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=["title", "score", "descendants", "anomaly_score", "is_anomaly"])
    title = data.get("title", pd.Series("", index=data.index)).fillna("").astype(str)
    matrix = pd.DataFrame({
        "score": pd.to_numeric(data.get("score", 0), errors="coerce"),
        "comments": pd.to_numeric(data.get("descendants", data.get("comments", 0)), errors="coerce"),
        "title_words": title.str.findall(r"\b[\w'-]+\b").str.len(),
        "title_chars": title.str.len(),
    })
    if len(data) < 5:
        result = data.copy(); result["anomaly_score"] = np.nan; result["is_anomaly"] = False
        return result
    detector = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              IsolationForest(n_estimators=250, contamination=contamination, random_state=42))
    detector.fit(matrix)
    estimator = detector[-1]
    transformed = detector[:-1].transform(matrix)
    result = data.copy()
    result["anomaly_score"] = -estimator.score_samples(transformed)
    result["is_anomaly"] = estimator.predict(transformed) == -1
    return result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def export_predictive_research(df: pd.DataFrame, output_dir="reports") -> dict:
    """Run the v7 classifier and anomaly detector, writing reproducible reports."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    _, metrics, predictions = train_virality_model(df)
    anomalies = detect_engagement_anomalies(df)
    metrics_path = out / "virality_model_metrics.json"
    predictions_path = out / "virality_predictions.csv"
    anomalies_path = out / "engagement_anomalies.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    anomalies.to_csv(anomalies_path, index=False)
    return {"metrics": metrics, "predictions": len(predictions), "anomalies": int(anomalies.is_anomaly.sum()),
            "metrics_file": str(metrics_path), "predictions_file": str(predictions_path),
            "anomalies_file": str(anomalies_path)}
