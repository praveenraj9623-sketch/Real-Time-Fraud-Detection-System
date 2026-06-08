"""Generate demo fraud alerts for the Streamlit dashboard.

This module is intentionally local and beginner-friendly: it reads a sample of
creditcard.csv, scores rows with the trained model, and writes high-risk rows to
MongoDB. It is useful when Kafka/Docker streaming is not running.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.api.main import top_shap_features
from src.processing.feature_engineering import (
    ALL_COLUMNS,
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    engineer_transaction_features,
    load_feature_stats,
)
from src.storage.mongodb_client import MongoDBClient
from src.utils.demo_selection import select_demo_indices
from src.utils.risk import (
    classify_probability,
    clip_probability,
    extract_transaction_amount,
    normalize_alert_document,
    risk_score_from_probability,
)

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw/creditcard.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_classifier.joblib"))
OPTIMAL_THRESHOLD_PATH = Path(os.getenv("OPTIMAL_THRESHOLD_PATH", "models/optimal_threshold.txt"))

MERCHANT_NAMES = [
    "Metro Grocers",
    "Bluebird Electronics",
    "Nova Travel",
    "Urban Fuel",
    "Acme Marketplace",
    "Silverline Pharmacy",
    "Cloud Nine Digital",
    "Harbor Hotel",
    "QuickServe Dining",
    "Northstar Apparel",
    "Crescent Jewelry",
    "Peak Fitness",
]


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the trained fraud model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    bundle = joblib.load(model_path)
    if bundle.get("placeholder") or ("model" not in bundle and "pipeline" not in bundle):
        raise ValueError("The current model file is a placeholder. Run train_batch first.")
    return bundle


def load_optimal_threshold(bundle: dict[str, Any]) -> float:
    """Read the tuned threshold used for FLAGGED/BLOCKED decisions."""
    if OPTIMAL_THRESHOLD_PATH.exists():
        try:
            return float(OPTIMAL_THRESHOLD_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return float(bundle.get("threshold", 0.5))


def build_transaction(row: pd.Series, rng: random.Random | None = None) -> dict[str, Any]:
    """Create a transaction event with merchant and transaction metadata."""
    rng = rng or random.Random()
    event: dict[str, Any] = {}
    amount = extract_transaction_amount(row)
    amount_source = "raw_creditcard_csv" if amount is not None else "fallback_missing_amount"
    for column in ALL_COLUMNS:
        if column == "Amount":
            event[column] = float(amount if amount is not None else 0.0)
            continue
        value = row[column]
        event[column] = int(value) if column == TARGET_COLUMN else float(value)

    missing = [column for column in BASE_FEATURE_COLUMNS if column not in event]
    if missing:
        raise ValueError(f"Missing required transaction fields: {missing}")

    simulated_event_time = (
        datetime.now(timezone.utc) - timedelta(minutes=rng.randint(0, 7 * 24 * 60))
    )
    event["transaction_id"] = str(uuid.uuid4())
    event["event_time_utc"] = simulated_event_time.isoformat()
    event["transaction_time_utc"] = simulated_event_time.isoformat()
    event["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
    event["merchant_name"] = rng.choice(MERCHANT_NAMES)
    event["amount_source"] = amount_source
    return event


def generate_demo_alerts(
    max_rows: int = 20000,
    alert_threshold: float = 0.1,
    max_alerts: int = 50,
    include_shap: bool = True,
    mongo_client: MongoDBClient | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Score CSV rows and insert the highest-risk demo alerts into MongoDB."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    bundle = load_model_bundle()
    predictor = bundle.get("model") or bundle.get("pipeline")
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    feature_stats = bundle.get("feature_stats") or load_feature_stats()
    decision_threshold = load_optimal_threshold(bundle)
    rng = random.Random(random_state)

    df = pd.read_csv(DATA_PATH, nrows=max_rows)
    missing_columns = [column for column in ALL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing columns: {missing_columns}")

    transactions = [build_transaction(row, rng) for _, row in df.iterrows()]
    engineered_rows = [engineer_transaction_features(txn, feature_stats) for txn in transactions]
    X = pd.DataFrame(engineered_rows)[feature_columns].astype(float)
    probabilities = np.clip(predictor.predict_proba(X)[:, 1], 0.0, 1.0)

    alert_threshold = clip_probability(alert_threshold)
    candidate_indices = np.where(probabilities >= alert_threshold)[0]
    if len(candidate_indices) == 0:
        return {
            "scanned": len(df),
            "inserted": 0,
            "alert_threshold": alert_threshold,
            "message": "No rows crossed the selected alert threshold.",
        }

    ranked_indices = select_demo_indices(
        probabilities,
        max_alerts=max_alerts,
        save_threshold=alert_threshold,
        model_threshold=decision_threshold,
        amounts=df["Amount"].astype(float).to_numpy(),
        random_state=random_state,
    )
    alerts = []
    for idx in ranked_indices:
        probability = clip_probability(probabilities[idx])
        risk = classify_probability(probability, decision_threshold, alert_threshold)
        shap_rows = top_shap_features(bundle, X.iloc[[idx]]) if include_shap else []
        transaction = transactions[idx]
        alert = normalize_alert_document(
            {
                **transaction,
                "actual_class": transaction.get(TARGET_COLUMN),
                "fraud_probability": probability,
                "decision": risk.decision,
                "risk_score": risk_score_from_probability(probability),
                "threshold": decision_threshold,
                "block_threshold": risk.block_threshold,
                "top_shap_features": shap_rows,
                "recommended_action": risk.recommended_action,
                "stream_alert_threshold": alert_threshold,
                "alert_reason": "dashboard_demo_generator",
                "scored_at_utc": datetime.now(timezone.utc),
            },
            default_threshold=decision_threshold,
            save_threshold=alert_threshold,
        )
        alerts.append(alert)

    client = mongo_client or MongoDBClient()
    inserted = client.insert_many_fraud_alerts(alerts)
    return {
        "scanned": len(df),
        "candidates": int(len(candidate_indices)),
        "inserted": inserted,
        "alert_threshold": alert_threshold,
        "max_probability": float(probabilities.max()),
        "decision_threshold": decision_threshold,
        "decision_mix": {
            "review": sum(alert["decision"] == "REVIEW" for alert in alerts),
            "flagged": sum(alert["decision"] == "FLAGGED" for alert in alerts),
            "blocked": sum(alert["decision"] == "BLOCKED" for alert in alerts),
        },
    }
