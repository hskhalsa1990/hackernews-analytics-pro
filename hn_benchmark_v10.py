"""v10: leakage-aware chronological model benchmarking.

Compares complementary scikit-learn classifiers on the same time-ordered holdout.
This is retrospective research, not a guarantee of future story performance.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from hn_predictive import _features, FEATURE_COLUMNS


def benchmark_virality_models(
    df: pd.DataFrame,
    min_rows: int = 60,
    high_score_quantile: float = 0.75,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare Logistic Regression, Random Forest and Extra Trees.

    Preprocessing is fitted on training rows only. The outcome threshold is
    calculated from training scores only. All estimators receive identical
    chronological train/test rows and predictor columns.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, average_precision_score,
            brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

    if "score" not in df.columns:
        raise ValueError("Dataset must include a score column")
    if not 0.5 < high_score_quantile < 1:
        raise ValueError("high_score_quantile must be between 0.5 and 1")
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4")

    data = df.copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["score"])
    if len(data) < min_rows:
        raise ValueError(f"Need at least {min_rows} scored stories; found {len(data)}")

    features = _features(data)
    if "created_at" in data:
        order = pd.to_datetime(data.created_at, utc=True, errors="coerce").sort_values(kind="stable").index
    elif "time" in data:
        order = pd.to_numeric(data.time, errors="coerce").sort_values(kind="stable").index
    else:
        order = data.index
    features = features.loc[order]
    scores = data.loc[order, "score"].clip(lower=0)
    split = min(max(int(len(features) * (1 - test_fraction)), 1), len(features) - 1)
    threshold = float(scores.iloc[:split].quantile(high_score_quantile))
    target = (scores >= threshold).astype(int)
    y_train, y_test = target.iloc[:split], target.iloc[split:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Chronological split must contain both classes; use a larger or more varied dataset")

    categorical = ["domain", "type", "author", "weekday_utc"]
    numeric = ["hour_utc", "title_word_count", "title_char_count"]
    preprocessing = ColumnTransformer([
        ("title_tfidf", TfidfVectorizer(max_features=1800, ngram_range=(1, 2), stop_words="english"), "title"),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    rows, prediction_frames = [], []
    for name, estimator in estimators.items():
        model = Pipeline([("features", preprocessing), ("classifier", estimator)])
        model.fit(features.iloc[:split], y_train)
        probability = model.predict_proba(features.iloc[split:])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "average_precision": float(average_precision_score(y_test, probability)),
            "brier_score": float(brier_score_loss(y_test, probability)),
            "train_rows": int(split), "test_rows": int(len(y_test)),
        }
        rows.append(row)
        prediction_frames.append(pd.DataFrame({
            "model": name, "source_index": features.index[split:],
            "actual_high_score": y_test.to_numpy(), "predicted_high_score": predicted,
            "high_score_probability": probability,
            "actual_score": scores.iloc[split:].to_numpy(),
        }))

    leaderboard = pd.DataFrame(rows).sort_values(
        ["roc_auc", "average_precision", "brier_score"], ascending=[False, False, True]
    ).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metadata = {
        "version": "10.0.0", "method": "same chronological holdout for all models",
        "split": f"chronological {1-test_fraction:.0%}/{test_fraction:.0%}",
        "high_score_threshold": threshold, "threshold_quantile": high_score_quantile,
        "target": "score >= training-only score quantile",
        "predictors": FEATURE_COLUMNS, "excluded_predictors": ["score", "descendants", "comments"],
        "selection_rule": "highest ROC-AUC; tie-break by average precision, then lower Brier score",
        "best_model": str(leaderboard.iloc[0]["model"]),
        "caution": "Retrospective single chronological holdout; benchmark results are sample- and period-dependent, not causal or guaranteed forecasts.",
    }
    return leaderboard, predictions, metadata


def export_v10_benchmark(df: pd.DataFrame, output_dir="reports", test_fraction: float = 0.2) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaderboard, predictions, metadata = benchmark_virality_models(df, test_fraction=test_fraction)
    leaderboard_path = out / "v10_model_leaderboard.csv"
    predictions_path = out / "v10_model_predictions.csv"
    summary_path = out / "v10_benchmark_summary.json"
    leaderboard.to_csv(leaderboard_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metadata["artifacts"] = {
        "leaderboard": str(leaderboard_path),
        "predictions": str(predictions_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "leaderboard_file": str(leaderboard_path),
            "predictions_file": str(predictions_path), "summary_file": str(summary_path),
            "models": len(leaderboard), "holdout_predictions": len(predictions)}
