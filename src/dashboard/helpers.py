"""Pure helpers used by the Streamlit fraud dashboard and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.utils.risk import (
    DEFAULT_MODEL_THRESHOLD,
    DEFAULT_SAVE_THRESHOLD,
    classify_probability,
    normalize_alert_document,
)


def normalize_artifact_path(value: Any, fallback: Path | str | None = None) -> Path:
    """Normalize Windows/Linux artifact paths for display."""
    if value:
        return Path(str(value).replace("\\", "/"))
    if fallback is None:
        return Path()
    return Path(str(fallback).replace("\\", "/"))


def short_transaction_id(value: Any, width: int = 10) -> str:
    """Return a compact transaction id while preserving the full id elsewhere."""
    text = str(value or "")
    if len(text) <= width + 3:
        return text
    return f"{text[:width]}..."


def format_currency(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def format_risk(value: Any) -> str:
    try:
        return f"{float(value):.0f} / 100"
    except (TypeError, ValueError):
        return "0 / 100"


def prepare_alert_dataframe(
    records: Iterable[Dict[str, Any]],
    *,
    model_threshold: float = DEFAULT_MODEL_THRESHOLD,
    save_threshold: float = DEFAULT_SAVE_THRESHOLD,
) -> pd.DataFrame:
    """Return a normalized dashboard DataFrame from raw Mongo records."""
    normalized = [
        normalize_alert_document(
            record,
            default_threshold=model_threshold,
            save_threshold=save_threshold,
            recompute_decision=True,
        )
        for record in records
    ]
    if not normalized:
        return pd.DataFrame()

    df = pd.DataFrame(normalized)
    for column in ["fraud_probability", "risk_score", "Amount", "hour_of_day"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["fraud_probability"] = df["fraud_probability"].clip(0.0, 1.0)
    df["risk_score"] = df["risk_score"].clip(0.0, 100.0)
    df["Amount"] = df["Amount"].clip(lower=0.0)
    df["hour_of_day"] = df["hour_of_day"].astype(int).clip(0, 23)

    if "stored_at_utc" in df.columns:
        df["stored_at_utc"] = pd.to_datetime(df["stored_at_utc"], errors="coerce")
    if "processing_timestamp" in df.columns:
        df["processing_timestamp"] = pd.to_datetime(df["processing_timestamp"], errors="coerce")
    if "event_time_utc" in df.columns:
        df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], errors="coerce")
    if "transaction_time_utc" in df.columns:
        df["transaction_time_utc"] = pd.to_datetime(df["transaction_time_utc"], errors="coerce")

    df["short_transaction_id"] = df["transaction_id"].map(short_transaction_id)
    df["amount_display"] = df["Amount"].map(format_currency)
    df["probability_display"] = df["fraud_probability"].map(format_percent)
    df["risk_display"] = df["risk_score"].map(format_risk)
    if "recommended_action" not in df.columns:
        df["recommended_action"] = ""
    df["action_short"] = df["recommended_action"].astype(str).str.slice(0, 92)
    return df


def summarize_alerts(df: pd.DataFrame) -> Dict[str, float | int]:
    """Build KPI values from normalized alert rows."""
    if df.empty:
        return {
            "alert_count": 0,
            "review_count": 0,
            "flagged_count": 0,
            "blocked_count": 0,
            "alerted_amount": 0.0,
            "blocked_amount": 0.0,
            "average_risk": 0.0,
        }

    decisions = df["decision"].astype(str).str.upper()
    return {
        "alert_count": int(len(df)),
        "review_count": int((decisions == "REVIEW").sum()),
        "flagged_count": int((decisions == "FLAGGED").sum()),
        "blocked_count": int((decisions == "BLOCKED").sum()),
        "alerted_amount": float(df["Amount"].sum()),
        "blocked_amount": float(df.loc[decisions == "BLOCKED", "Amount"].sum()),
        "average_risk": float(df["risk_score"].mean()),
    }


def merchant_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate merchant risk without summing probabilities."""
    if df.empty or "merchant_name" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "event_time_utc" in df.columns:
        df["alert_event_time"] = pd.to_datetime(df["event_time_utc"], errors="coerce")
    elif "transaction_time_utc" in df.columns:
        df["alert_event_time"] = pd.to_datetime(df["transaction_time_utc"], errors="coerce")
    elif "stored_at_utc" in df.columns:
        df["alert_event_time"] = pd.to_datetime(df["stored_at_utc"], errors="coerce")
    else:
        df["alert_event_time"] = pd.NaT
    grouped = (
        df.assign(
            is_flagged=df["decision"].isin(["REVIEW", "FLAGGED"]).astype(int),
            is_blocked=(df["decision"] == "BLOCKED").astype(int),
        )
        .groupby("merchant_name", dropna=False)
        .agg(
            alert_count=("transaction_id", "count"),
            flagged_count=("is_flagged", "sum"),
            blocked_count=("is_blocked", "sum"),
            avg_probability=("fraud_probability", "mean"),
            max_probability=("fraud_probability", "max"),
            total_amount=("Amount", "sum"),
            last_alert_time=("alert_event_time", "max"),
        )
        .reset_index()
    )
    grouped["avg_risk_score"] = (grouped["avg_probability"] * 100).clip(0, 100)
    grouped["max_risk_score"] = (grouped["max_probability"] * 100).clip(0, 100)
    return grouped.sort_values(
        ["blocked_count", "avg_risk_score", "total_amount"],
        ascending=[False, False, False],
    )


def decision_from_probability(probability: float, model_threshold: float, save_threshold: float) -> str:
    """Test-friendly wrapper for dashboard/API decision bands."""
    return classify_probability(probability, model_threshold, save_threshold).decision


def metrics_from_confusion_counts(
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
    true_positives: int,
) -> Dict[str, float | int]:
    """Compute precision, recall, and F1 from confusion-matrix counts."""
    tp = int(true_positives)
    fp = int(false_positives)
    fn = int(false_negatives)
    tn = int(true_negatives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "predicted_fraud_count": tp + fp,
    }


def build_active_xgb_metrics(
    stored_metrics: Dict[str, Any],
    active_metrics: Dict[str, Any],
    active_threshold: float,
) -> Dict[str, Any]:
    """Return XGBoost metrics with active-threshold fields overriding stored training metrics."""
    merged = dict(stored_metrics)
    if active_metrics:
        for key in [
            "precision",
            "recall",
            "f1",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "predicted_fraud_count",
            "fraud_cases",
            "validation_rows",
        ]:
            if key in active_metrics:
                merged[key] = active_metrics[key]
    merged["threshold"] = float(active_threshold)
    merged["metric_source"] = "active_threshold_recomputed" if active_metrics else "stored_training_artifact"
    return merged


def confusion_matrix_values(metrics: Dict[str, Any]) -> list[list[int]]:
    """Return [[TN, FP], [FN, TP]] from a metric row."""
    return [
        [int(metrics.get("true_negatives", 0)), int(metrics.get("false_positives", 0))],
        [int(metrics.get("false_negatives", 0)), int(metrics.get("true_positives", 0))],
    ]
