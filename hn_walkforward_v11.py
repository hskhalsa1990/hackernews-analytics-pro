"""v11: expanding-window evaluation and probability calibration diagnostics.

This module evaluates the existing v10 classifiers across multiple chronological
holdout windows. Each fold learns its own outcome threshold from its training
history only. It also reports probability calibration and bootstrap AUC intervals.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from hn_predictive import _features, FEATURE_COLUMNS


def _calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Weighted expected calibration error using equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if mask.any():
            error += float(mask.sum() / total) * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return float(error)


def _auc_bootstrap_interval(y_true: np.ndarray, probability: np.ndarray,
                            repeats: int = 500, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap interval; resamples rows and skips single-class draws."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(repeats):
        sample = rng.integers(0, n, n)
        if np.unique(y_true[sample]).size == 2:
            values.append(float(roc_auc_score(y_true[sample], probability[sample])))
    if len(values) < max(30, repeats // 10):
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def walk_forward_benchmark(
    df: pd.DataFrame,
    min_rows: int = 60,
    high_score_quantile: float = 0.75,
    n_splits: int = 3,
    test_fraction: float = 0.15,
    bootstrap_repeats: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run expanding-window validation for LR, Random Forest and Extra Trees.

    All folds preserve temporal order. Each fold's target threshold is learned
    from that fold's training scores, while score/comments remain excluded from
    the feature matrix. AUC is null for a test window containing one class.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, average_precision_score,
            brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

    if "score" not in df.columns:
        raise ValueError("Dataset must include a score column")
    if not 0.5 < high_score_quantile < 1:
        raise ValueError("high_score_quantile must be between 0.5 and 1")
    if not 2 <= n_splits <= 10:
        raise ValueError("n_splits must be between 2 and 10")
    if not 0.05 <= test_fraction <= 0.3:
        raise ValueError("test_fraction must be between 0.05 and 0.30")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats cannot be negative")

    data = df.copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["score"])
    if len(data) < min_rows:
        raise ValueError(f"Need at least {min_rows} scored stories; found {len(data)}")

    if "created_at" in data:
        order = pd.to_datetime(data.created_at, utc=True, errors="coerce").sort_values(kind="stable").index
    elif "time" in data:
        order = pd.to_numeric(data.time, errors="coerce").sort_values(kind="stable").index
    else:
        order = data.index
    data = data.loc[order]
    features = _features(data)
    scores = data["score"].clip(lower=0)

    test_size = max(2, int(np.floor(len(data) * test_fraction)))
    initial_train = len(data) - n_splits * test_size
    if initial_train < max(25, int(len(data) * 0.35)):
        raise ValueError("Not enough rows for requested walk-forward folds; reduce n_splits or test_fraction")

    categorical = ["domain", "type", "author", "weekday_utc"]
    numeric = ["hour_utc", "title_word_count", "title_char_count"]
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    fold_rows, prediction_rows = [], []

    for fold in range(n_splits):
        train_end = initial_train + fold * test_size
        test_end = min(train_end + test_size, len(data))
        if test_end <= train_end:
            continue
        threshold = float(scores.iloc[:train_end].quantile(high_score_quantile))
        y_train = (scores.iloc[:train_end] >= threshold).astype(int)
        y_test = (scores.iloc[train_end:test_end] >= threshold).astype(int)
        if y_train.nunique() < 2:
            raise ValueError(f"Fold {fold + 1} training history contains only one class; collect more varied data")

        for name, estimator in estimators.items():
            preprocessing = ColumnTransformer([
                ("title_tfidf", TfidfVectorizer(max_features=1800, ngram_range=(1, 2), stop_words="english"), "title"),
                ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ])
            model = Pipeline([("features", preprocessing), ("classifier", estimator)])
            model.fit(features.iloc[:train_end], y_train)
            probability = model.predict_proba(features.iloc[train_end:test_end])[:, 1]
            predicted = (probability >= 0.5).astype(int)
            actual = y_test.to_numpy()
            has_two_classes = np.unique(actual).size == 2
            auc = float(roc_auc_score(actual, probability)) if has_two_classes else float("nan")
            ap = float(average_precision_score(actual, probability)) if has_two_classes else float("nan")
            fold_rows.append({
                "fold": fold + 1, "model": name, "train_rows": train_end,
                "test_rows": len(actual), "threshold": threshold,
                "positive_rate": float(actual.mean()),
                "accuracy": float(accuracy_score(actual, predicted)),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "roc_auc": auc, "average_precision": ap,
                "brier_score": float(brier_score_loss(actual, probability)),
                "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
                "expected_calibration_error": _calibration_error(actual, probability),
                "test_has_both_classes": bool(has_two_classes),
            })
            prediction_rows.extend({
                "fold": fold + 1, "model": name, "source_index": features.index[train_end + offset],
                "actual_high_score": int(actual[offset]), "predicted_high_score": int(predicted[offset]),
                "high_score_probability": float(probability[offset]),
                "actual_score": float(scores.iloc[train_end + offset]), "training_threshold": threshold,
            } for offset in range(len(actual)))

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision",
                      "brier_score", "log_loss", "expected_calibration_error"]
    summary_rows = []
    for name, group in folds.groupby("model"):
        row = {"model": name, "folds": int(len(group)), "total_test_rows": int(group.test_rows.sum())}
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean()) if group[metric].notna().any() else float("nan")
            row[f"{metric}_std"] = float(group[metric].std(ddof=0)) if group[metric].notna().any() else float("nan")
        combined = predictions[predictions.model == name]
        y = combined.actual_high_score.to_numpy(dtype=int)
        p = combined.high_score_probability.to_numpy(dtype=float)
        if np.unique(y).size == 2 and bootstrap_repeats:
            low, high = _auc_bootstrap_interval(y, p, bootstrap_repeats, seed=42)
        else:
            low, high = float("nan"), float("nan")
        row["pooled_roc_auc_ci95_low"] = low
        row["pooled_roc_auc_ci95_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["roc_auc_mean", "average_precision_mean", "brier_score_mean"],
        ascending=[False, False, True], na_position="last").reset_index(drop=True)
    metadata = {
        "version": "11.0.0", "method": "expanding-window walk-forward validation",
        "folds": int(folds.fold.nunique()), "initial_train_rows": int(initial_train),
        "test_rows_per_fold": int(test_size), "high_score_quantile": high_score_quantile,
        "threshold_policy": "recomputed from each fold's training scores only",
        "predictors": FEATURE_COLUMNS, "excluded_predictors": ["score", "descendants", "comments"],
        "models": list(estimators), "best_model_by_mean_roc_auc": str(summary.iloc[0].model),
        "calibration": "Brier score, log loss and equal-width expected calibration error; no post-hoc calibration fitted",
        "confidence_interval": f"pooled out-of-fold percentile bootstrap, {bootstrap_repeats} resamples",
        "limitations": "Observational Hacker News data; fold thresholds define a relative score class, not an absolute future outcome. Single-class test folds have undefined ROC-AUC and average precision.",
    }
    return summary, folds, predictions, metadata


def export_v11_walkforward(df: pd.DataFrame, output_dir="reports", **kwargs) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, folds, predictions, metadata = walk_forward_benchmark(df, **kwargs)
    paths = {
        "summary": out / "v11_walkforward_model_summary.csv",
        "folds": out / "v11_walkforward_fold_metrics.csv",
        "predictions": out / "v11_walkforward_predictions.csv",
        "metadata": out / "v11_walkforward_metadata.json",
    }
    summary.to_csv(paths["summary"], index=False)
    folds.to_csv(paths["folds"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metadata["artifacts"] = {key: str(path) for key, path in paths.items()}
    paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False, default=lambda value: None), encoding="utf-8")
    return {**metadata, "summary_file": str(paths["summary"]), "folds_file": str(paths["folds"]),
            "predictions_file": str(paths["predictions"]), "metadata_file": str(paths["metadata"]),
            "model_count": len(summary), "fold_metric_rows": len(folds), "oof_predictions": len(predictions)}
