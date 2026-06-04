"""FastAPI service for real-time fraud scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.processing.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_STATS_PATH,
    engineer_transaction_features,
    load_feature_stats,
)
from src.utils.risk import (
    DEFAULT_SAVE_THRESHOLD,
    classify_probability,
    clip_probability,
    normalize_amount,
    risk_score_from_probability,
)

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_classifier.joblib"))
OPTIMAL_THRESHOLD_PATH = Path(os.getenv("OPTIMAL_THRESHOLD_PATH", "models/optimal_threshold.txt"))
TRAINING_STATS_PATH = Path(os.getenv("TRAINING_STATS_PATH", "models/training_stats.json"))
DEFAULT_FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.50"))
ALERT_SAVE_THRESHOLD = float(os.getenv("ALERT_SAVE_THRESHOLD", str(DEFAULT_SAVE_THRESHOLD)))

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Scores credit card transactions and explains fraud risk.",
    version="2.0.0",
)

_model_bundle: Optional[dict[str, Any]] = None
_shap_explainer: Any = None


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed between this transaction and first transaction in dataset")
    Amount: float = Field(..., description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    class Config:
        extra = "allow"


class ShapFeature(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: str
    value: Optional[float] = None


class FraudScoreResponse(BaseModel):
    transaction_id: Optional[str] = None
    fraud_probability: float
    decision: str
    risk_score: float
    threshold: float
    block_threshold: float
    top_shap_features: List[ShapFeature]
    recommended_action: str


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a pydantic model as a dict for v1/v2 compatibility."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_model_bundle() -> dict[str, Any]:
    """Load the trained XGBoost model bundle once per process."""
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run python -m src.modeling.train_batch first.",
            )
        bundle = joblib.load(MODEL_PATH)
        if not isinstance(bundle, dict):
            raise HTTPException(status_code=503, detail=f"Invalid model bundle at {MODEL_PATH}.")
        if bundle.get("placeholder") or ("model" not in bundle and "pipeline" not in bundle):
            raise HTTPException(
                status_code=503,
                detail="Placeholder model found. Run python -m src.modeling.train_batch to create a real model.",
            )
        _model_bundle = bundle
    return _model_bundle


def get_predictor(bundle: dict[str, Any]) -> Any:
    if "model" in bundle:
        return bundle["model"]
    if "pipeline" in bundle:
        return bundle["pipeline"]
    raise HTTPException(status_code=503, detail="Model bundle has no predictor.")


def get_optimal_threshold(bundle: dict[str, Any]) -> float:
    """Resolve threshold from tuned text file, then bundle, then env default."""
    if OPTIMAL_THRESHOLD_PATH.exists():
        try:
            return float(OPTIMAL_THRESHOLD_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return float(bundle.get("threshold", DEFAULT_FRAUD_THRESHOLD))


def get_training_statistics(bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return training statistics for /health."""
    stats = load_json_file(TRAINING_STATS_PATH)
    if stats:
        return stats
    if bundle and bundle.get("feature_stats"):
        return {"dataset": bundle["feature_stats"]}
    feature_stats = load_json_file(FEATURE_STATS_PATH)
    if feature_stats:
        return {"dataset": feature_stats}
    return {}


def get_shap_explainer(bundle: dict[str, Any]) -> Any:
    """Build and cache a SHAP TreeExplainer for the XGBoost model."""
    global _shap_explainer
    if _shap_explainer is None:
        predictor = get_predictor(bundle)
        if hasattr(predictor, "named_steps") and "model" in predictor.named_steps:
            predictor = predictor.named_steps["model"]
        _shap_explainer = shap.TreeExplainer(predictor)
    return _shap_explainer


def top_shap_features(bundle: dict[str, Any], X: pd.DataFrame) -> list[dict[str, Any]]:
    """Return top five SHAP drivers by absolute contribution."""
    try:
        explainer = get_shap_explainer(bundle)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        values = np.asarray(shap_values)
        if values.ndim == 2:
            values = values[0]
        top_indices = np.argsort(np.abs(values))[-5:][::-1]
        rows = []
        for idx in top_indices:
            shap_value = float(values[idx])
            feature = X.columns[idx]
            rows.append(
                {
                    "feature": feature,
                    "feature_value": float(X.iloc[0, idx]),
                    "value": float(X.iloc[0, idx]),
                    "shap_value": shap_value,
                    "direction": "risk_increasing" if shap_value >= 0 else "risk_decreasing",
                }
            )
        return rows
    except Exception:
        return []


def classify_decision(probability: float, threshold: float) -> tuple[str, float, str]:
    """Map probability into APPROVED, REVIEW, FLAGGED, or BLOCKED."""
    result = classify_probability(probability, threshold, ALERT_SAVE_THRESHOLD)
    return result.decision, result.block_threshold, result.recommended_action


def score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Score a raw transaction dictionary and return the full API response."""
    bundle = get_model_bundle()
    feature_stats = bundle.get("feature_stats") or load_feature_stats()
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    payload = dict(payload)
    payload["Amount"] = normalize_amount(payload.get("Amount", 0.0))
    engineered = engineer_transaction_features(payload, feature_stats)
    X = pd.DataFrame([engineered])[feature_columns].astype(float)
    predictor = get_predictor(bundle)
    probability = clip_probability(predictor.predict_proba(X)[:, 1][0])
    threshold = clip_probability(get_optimal_threshold(bundle))
    decision, block_threshold, action = classify_decision(probability, threshold)

    return {
        "transaction_id": payload.get("transaction_id"),
        "fraud_probability": probability,
        "decision": decision,
        "risk_score": risk_score_from_probability(probability),
        "threshold": threshold,
        "block_threshold": block_threshold,
        "top_shap_features": top_shap_features(bundle, X),
        "recommended_action": action,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Return model and training metadata used by monitoring clients."""
    try:
        bundle = get_model_bundle()
        threshold = get_optimal_threshold(bundle)
        return {
            "status": "ok",
            "model_version": bundle.get("model_version") or bundle.get("mlflow_run_id") or "local",
            "model_type": bundle.get("model_type", "xgboost"),
            "optimal_threshold": threshold,
            "alert_save_threshold": ALERT_SAVE_THRESHOLD,
            "training_dataset_statistics": get_training_statistics(bundle),
        }
    except HTTPException as exc:
        return {
            "status": "model_missing",
            "model_version": None,
            "optimal_threshold": DEFAULT_FRAUD_THRESHOLD,
            "alert_save_threshold": ALERT_SAVE_THRESHOLD,
            "training_dataset_statistics": get_training_statistics(None),
            "detail": exc.detail,
        }


@app.post("/score", response_model=FraudScoreResponse)
def score_transaction(transaction: Transaction) -> FraudScoreResponse:
    """Score a single transaction."""
    try:
        response = score_payload(model_to_dict(transaction))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FraudScoreResponse(**response)


@app.post("/batch-score")
def batch_score(transactions: List[Transaction]) -> Dict[str, Any]:
    """Score a list of transactions and return aggregate fraud statistics."""
    if not transactions:
        raise HTTPException(status_code=400, detail="Request body must contain at least one transaction.")

    results = []
    for transaction in transactions:
        try:
            results.append(score_payload(model_to_dict(transaction)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    probabilities = [row["fraud_probability"] for row in results]
    decision_counts = {
        "APPROVED": sum(row["decision"] == "APPROVED" for row in results),
        "REVIEW": sum(row["decision"] == "REVIEW" for row in results),
        "FLAGGED": sum(row["decision"] == "FLAGGED" for row in results),
        "BLOCKED": sum(row["decision"] == "BLOCKED" for row in results),
    }
    flagged_or_blocked = decision_counts["REVIEW"] + decision_counts["FLAGGED"] + decision_counts["BLOCKED"]

    return {
        "transaction_count": len(results),
        "decision_counts": decision_counts,
        "flagged_or_blocked_count": flagged_or_blocked,
        "estimated_fraud_rate": flagged_or_blocked / len(results),
        "average_fraud_probability": float(np.mean(probabilities)),
        "max_fraud_probability": float(np.max(probabilities)),
        "average_risk_score": float(np.mean([row["risk_score"] for row in results])),
        "results": results,
    }
