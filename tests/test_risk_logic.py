"""Tests for shared risk and dashboard helper logic."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard.helpers import (
    build_active_xgb_metrics,
    confusion_matrix_values,
    decision_from_probability,
    merchant_risk_summary,
    normalize_artifact_path,
    prepare_alert_dataframe,
    summarize_alerts,
)
from src.utils.demo_selection import select_demo_indices
from src.utils.risk import (
    classify_probability,
    clip_probability,
    clip_risk_score,
    normalize_amount,
    risk_score_from_probability,
)


def test_probability_and_risk_clipping():
    assert clip_probability(-3) == 0.0
    assert clip_probability(2.5) == 1.0
    assert risk_score_from_probability(1.8) == 100.0
    assert clip_risk_score(-1) == 0.0
    assert clip_risk_score(145) == 100.0
    assert normalize_amount(-15) == 0.0


def test_decision_bands():
    assert classify_probability(0.05, model_threshold=0.9, save_threshold=0.1).decision == "APPROVED"
    assert classify_probability(0.25, model_threshold=0.9, save_threshold=0.1).decision == "REVIEW"
    assert classify_probability(0.65, model_threshold=0.9, save_threshold=0.1).decision == "FLAGGED"
    assert classify_probability(0.95, model_threshold=0.9, save_threshold=0.1).decision == "BLOCKED"
    assert decision_from_probability(0.95, 0.9, 0.1) == "BLOCKED"


def test_prepare_alert_dataframe_empty_collection_is_safe():
    df = prepare_alert_dataframe([])
    summary = summarize_alerts(df)
    assert df.empty
    assert summary["alert_count"] == 0
    assert summary["average_risk"] == 0.0


def test_merchant_summary_uses_average_and_max_probability_not_sum():
    df = prepare_alert_dataframe(
        [
            {"transaction_id": "a", "merchant_name": "Shop A", "fraud_probability": 0.2, "Amount": 10},
            {"transaction_id": "b", "merchant_name": "Shop A", "fraud_probability": 0.8, "Amount": 20},
        ],
        model_threshold=0.9,
        save_threshold=0.1,
    )
    merchants = merchant_risk_summary(df)
    row = merchants.iloc[0]
    assert row["avg_probability"] == 0.5
    assert row["max_probability"] == 0.8
    assert row["avg_risk_score"] == 50.0
    assert row["max_risk_score"] == 80.0


def test_demo_selection_produces_varied_probability_bands():
    probabilities = np.array([0.03, 0.12, 0.22, 0.45, 0.55, 0.72, 0.91, 0.97, 0.99])
    selected = select_demo_indices(
        probabilities,
        max_alerts=6,
        save_threshold=0.1,
        model_threshold=0.9,
        random_state=7,
    )
    decisions = {
        classify_probability(probabilities[idx], model_threshold=0.9, save_threshold=0.1).decision
        for idx in selected
    }
    assert {"REVIEW", "FLAGGED", "BLOCKED"}.issubset(decisions)


def test_artifact_path_normalization_accepts_windows_and_linux_paths():
    windows_path = r"models\artifacts\xgboost_roc_curve.png"
    normalized = normalize_artifact_path(windows_path)
    assert isinstance(normalized, Path)
    assert normalized.as_posix() == "models/artifacts/xgboost_roc_curve.png"


def test_active_model_metrics_override_old_training_threshold_and_match_confusion():
    old_training_metrics = {
        "precision": 0.21,
        "recall": 0.89,
        "f1": 0.34,
        "threshold": 0.90,
        "auc": 0.98,
        "auprc": 0.82,
    }
    active_counts = {
        "precision": 0.7410714285714286,
        "recall": 0.8469387755102041,
        "f1": 0.7904761904761906,
        "true_negatives": 56835,
        "false_positives": 29,
        "false_negatives": 15,
        "true_positives": 83,
        "predicted_fraud_count": 112,
    }
    row = build_active_xgb_metrics(old_training_metrics, active_counts, active_threshold=0.999)

    assert row["threshold"] == 0.999
    assert row["precision"] == active_counts["precision"]
    assert row["recall"] == active_counts["recall"]
    assert row["f1"] == active_counts["f1"]
    assert row["metric_source"] == "active_threshold_recomputed"
    assert confusion_matrix_values(row) == [[56835, 29], [15, 83]]
