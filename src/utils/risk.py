"""Shared risk scoring, validation, and decision-band helpers."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

ALLOWED_DECISIONS = {"APPROVED", "REVIEW", "FLAGGED", "BLOCKED"}
DEFAULT_SAVE_THRESHOLD = 0.10
DEFAULT_MODEL_THRESHOLD = 0.50


@dataclass(frozen=True)
class RiskDecision:
    """Decision-band result for one scored transaction."""

    decision: str
    block_threshold: float
    recommended_action: str


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float, falling back when conversion fails."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def clip_probability(value: Any) -> float:
    """Return a valid fraud probability in the closed interval [0, 1]."""
    return min(max(to_float(value, 0.0), 0.0), 1.0)


def clip_risk_score(value: Any) -> float:
    """Return a valid 0-100 risk score."""
    return min(max(to_float(value, 0.0), 0.0), 100.0)


def risk_score_from_probability(probability: Any) -> float:
    """Convert a probability into a clipped 0-100 risk score."""
    return round(clip_probability(probability) * 100.0, 2)


def normalize_amount(value: Any) -> float:
    """Return a numeric non-negative transaction amount."""
    return max(to_float(value, 0.0), 0.0)


def normalize_threshold(value: Any, minimum: float = 0.0) -> float:
    """Return a clipped probability threshold with a lower bound."""
    return max(clip_probability(value), minimum)


def classify_probability(
    probability: Any,
    model_threshold: Any = DEFAULT_MODEL_THRESHOLD,
    save_threshold: Any = DEFAULT_SAVE_THRESHOLD,
) -> RiskDecision:
    """Map probability into APPROVED, REVIEW, FLAGGED, or BLOCKED bands."""
    prob = clip_probability(probability)
    save_at = normalize_threshold(save_threshold, 0.0)
    block_at = max(normalize_threshold(model_threshold, 0.0), 0.50)

    if prob >= block_at:
        return RiskDecision(
            decision="BLOCKED",
            block_threshold=block_at,
            recommended_action=(
                "Block the transaction, notify the customer, and open a high-priority fraud investigation."
            ),
        )
    if prob >= 0.50:
        return RiskDecision(
            decision="FLAGGED",
            block_threshold=block_at,
            recommended_action="Hold for step-up authentication or manual analyst review before approval.",
        )
    if prob >= save_at:
        return RiskDecision(
            decision="REVIEW",
            block_threshold=block_at,
            recommended_action="Route to the review queue and monitor related customer activity.",
        )
    return RiskDecision(
        decision="APPROVED",
        block_threshold=block_at,
        recommended_action="Approve the transaction and continue passive monitoring.",
    )


def parse_timestamp(value: Any) -> datetime | None:
    """Parse common timestamp values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not value:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def hour_of_day_from_document(document: Dict[str, Any]) -> int:
    """Return hour 0-23 from event time, explicit hour, dataset Time, or stored timestamp."""
    for field in ("event_time_utc", "transaction_time_utc"):
        parsed = parse_timestamp(document.get(field))
        if parsed is not None:
            return int(parsed.hour)
    if "hour_of_day" in document:
        hour = int(to_float(document.get("hour_of_day"), -1))
        if 0 <= hour <= 23:
            return hour
    if "Time" in document:
        return int((to_float(document.get("Time"), 0.0) % 86400) // 3600)
    for field in ("processing_timestamp", "scored_at_utc", "stored_at_utc"):
        parsed = parse_timestamp(document.get(field))
        if parsed is not None:
            return int(parsed.hour)
    return 0


def normalize_alert_document(
    alert: Dict[str, Any],
    *,
    default_threshold: float = DEFAULT_MODEL_THRESHOLD,
    save_threshold: float = DEFAULT_SAVE_THRESHOLD,
    recompute_decision: bool = True,
) -> Dict[str, Any]:
    """Return a Mongo/dashboard-safe fraud alert document."""
    document = dict(alert)
    probability = clip_probability(document.get("fraud_probability", 0.0))
    threshold = normalize_threshold(document.get("threshold", default_threshold), 0.50)
    amount = normalize_amount(document.get("Amount", document.get("amount", 0.0)))
    computed = classify_probability(probability, threshold, save_threshold)

    decision = str(document.get("decision", "")).upper()
    if recompute_decision or decision not in ALLOWED_DECISIONS:
        decision = computed.decision

    document["transaction_id"] = str(document.get("transaction_id") or uuid.uuid4())
    document["fraud_probability"] = probability
    document["risk_score"] = risk_score_from_probability(probability)
    document["Amount"] = amount
    document["threshold"] = threshold
    document["block_threshold"] = computed.block_threshold
    document["decision"] = decision
    document["recommended_action"] = document.get("recommended_action") or computed.recommended_action
    document["hour_of_day"] = hour_of_day_from_document(document)

    parsed_timestamp = parse_timestamp(document.get("processing_timestamp"))
    if parsed_timestamp is not None:
        document["processing_timestamp"] = parsed_timestamp
    for timestamp_field in ("event_time_utc", "transaction_time_utc"):
        parsed_event_time = parse_timestamp(document.get(timestamp_field))
        if parsed_event_time is not None:
            document[timestamp_field] = parsed_event_time
    return document
