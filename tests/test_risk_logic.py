"""Tests for shared risk and dashboard helper logic."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard.helpers import (
    ANALYTICS_SCOPE_CAPTION,
    build_active_xgb_metrics,
    confusion_matrix_values,
    cumulative_alerted_amount_by_event_time,
    decision_from_probability,
    format_probability_display,
    merchant_risk_summary,
    has_confusion_count_fields,
    night_time_mask,
    normalize_artifact_path,
    prepare_alert_dataframe,
    summarize_alerts,
)
from src.streaming.local_demo_alerts import coerce_csv_row as coerce_local_demo_csv_row
from src.utils.demo_selection import select_demo_indices
from src.utils.risk import (
    classify_probability,
    clip_probability,
    clip_risk_score,
    extract_transaction_amount,
    normalize_amount,
    risk_score_from_probability,
)


def _raw_creditcard_csv_row(amount: str | None = "123.45") -> dict[str, str]:
    row = {"Time": "10.0", "Class": "0"}
    if amount is not None:
        row["Amount"] = amount
    for index in range(1, 29):
        row[f"V{index}"] = "0.0"
    return row


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


def test_prepare_alert_dataframe_clips_probability_and_risk_ranges():
    df = prepare_alert_dataframe(
        [
            {"transaction_id": "low", "fraud_probability": -1, "risk_score": -20, "Amount": 3},
            {"transaction_id": "high", "fraud_probability": 2, "risk_score": 150, "Amount": 4},
        ],
        model_threshold=0.9,
        save_threshold=0.1,
    )
    assert df["fraud_probability"].between(0, 1).all()
    assert df["risk_score"].between(0, 100).all()


def test_probability_display_formats_high_values_without_changing_numeric_value():
    stored_probability = 0.9997

    assert format_probability_display(stored_probability) == "≥99.95%"
    assert stored_probability == 0.9997
    assert format_probability_display(0.12345) == "12.35%"
    assert format_probability_display(None) == "N/A"


def test_demo_alert_generation_preserves_original_amount_when_amount_exists():
    raw_row = _raw_creditcard_csv_row("123.45")
    event = coerce_local_demo_csv_row(raw_row)

    assert extract_transaction_amount(raw_row) == 123.45
    assert event["Amount"] == 123.45
    assert event["amount_source"] == "raw_creditcard_csv"


def test_no_silent_9999_fallback_for_real_csv_rows():
    raw_row = _raw_creditcard_csv_row("1.23")
    event = coerce_local_demo_csv_row(raw_row)

    assert event["Amount"] == 1.23
    assert event["Amount"] != 99.99
    assert event["amount_source"] == "raw_creditcard_csv"


def test_missing_amount_fallback_is_explicitly_labeled():
    raw_row = _raw_creditcard_csv_row(None)
    event = coerce_local_demo_csv_row(raw_row)

    assert event["Amount"] == 0.0
    assert event["amount_source"] == "fallback_missing_amount"


def test_analytics_caption_scope_is_clear():
    assert "all stored alerts" in ANALYTICS_SCOPE_CAPTION
    assert "recent-alert limit" in ANALYTICS_SCOPE_CAPTION
    assert "live feed table" in ANALYTICS_SCOPE_CAPTION


def test_cumulative_timeline_prefers_event_time_before_stored_time():
    df = prepare_alert_dataframe(
        [
            {
                "transaction_id": "stored-first-event-later",
                "fraud_probability": 0.4,
                "Amount": 10,
                "event_time_utc": "2026-01-02T09:00:00Z",
                "stored_at_utc": "2026-06-05T00:00:01Z",
            },
            {
                "transaction_id": "stored-later-event-first",
                "fraud_probability": 0.8,
                "Amount": 5,
                "event_time_utc": "2026-01-01T09:00:00Z",
                "stored_at_utc": "2026-06-05T00:00:02Z",
            },
        ],
        model_threshold=0.9,
        save_threshold=0.1,
    )
    timeline = cumulative_alerted_amount_by_event_time(df)

    assert timeline["transaction_id"].tolist() == [
        "stored-later-event-first",
        "stored-first-event-later",
    ]
    assert timeline["cum_amount"].tolist() == [5.0, 15.0]


def test_night_time_mask_uses_transaction_event_time_not_stored_time():
    df = prepare_alert_dataframe(
        [
            {
                "transaction_id": "night-event",
                "fraud_probability": 0.4,
                "Amount": 10,
                "event_time_utc": "2026-01-02T23:30:00Z",
                "stored_at_utc": "2026-06-05T12:00:00Z",
            },
            {
                "transaction_id": "day-event",
                "fraud_probability": 0.4,
                "Amount": 10,
                "transaction_time_utc": "2026-01-02T14:30:00Z",
                "stored_at_utc": "2026-06-05T23:00:00Z",
            },
        ],
        model_threshold=0.9,
        save_threshold=0.1,
    )
    assert night_time_mask(df).tolist() == [True, False]


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


def test_merchant_summary_calculates_block_rate_and_review_count():
    df = prepare_alert_dataframe(
        [
            {"transaction_id": "a", "merchant_name": "Shop A", "fraud_probability": 0.2, "Amount": 10},
            {"transaction_id": "b", "merchant_name": "Shop A", "fraud_probability": 0.7, "Amount": 20},
            {"transaction_id": "c", "merchant_name": "Shop A", "fraud_probability": 0.95, "Amount": 30},
            {"transaction_id": "d", "merchant_name": "Shop A", "fraud_probability": 0.98, "Amount": 40},
        ],
        model_threshold=0.9,
        save_threshold=0.1,
    )
    row = merchant_risk_summary(df).iloc[0]
    assert row["alert_count"] == 4
    assert row["review_count"] == 1
    assert row["flagged_count"] == 1
    assert row["blocked_count"] == 2
    assert row["block_rate"] == 0.5
    assert row["median_risk_score"] == 82.5


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


def test_demo_selection_includes_varied_amounts_when_available():
    probabilities = np.array([0.12, 0.18, 0.32, 0.55, 0.62, 0.78, 0.91, 0.95, 0.99, 0.93, 0.44, 0.73])
    amounts = np.array([0.0, 1.0, 12.0, 0.0, 48.0, 125.0, 1.0, 300.0, 0.0, 80.0, 25.0, 500.0])
    selected = select_demo_indices(
        probabilities,
        max_alerts=9,
        save_threshold=0.1,
        model_threshold=0.9,
        amounts=amounts,
        random_state=3,
    )
    selected_amounts = amounts[selected]
    buckets = {
        "tiny" if amount <= 1 else "medium" if amount < 100 else "high"
        for amount in selected_amounts
    }
    assert len(buckets) >= 2
    assert (selected_amounts > 1).any()


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


def test_partial_threshold_summary_metrics_do_not_claim_confusion_counts():
    summary_only_metrics = {
        "precision": 0.74,
        "recall": 0.84,
        "f1": 0.79,
        "threshold": 0.999,
        "metric_source": "best_f1_threshold_summary",
    }

    assert has_confusion_count_fields(summary_only_metrics) is False

    full_count_metrics = {
        **summary_only_metrics,
        "true_negatives": 56835,
        "false_positives": 29,
        "false_negatives": 15,
        "true_positives": 83,
    }
    assert has_confusion_count_fields(full_count_metrics) is True
