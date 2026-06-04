"""Evaluate the saved fraud classifier on the validation split."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from src.modeling.threshold_tuner import evaluate_thresholds
from src.processing.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_classifier.joblib"))
VALIDATION_DATA_PATH = Path(os.getenv("VALIDATION_DATA_PATH", "models/validation_data.csv"))


def evaluate() -> None:
    """Print validation metrics for the saved XGBoost classifier."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python -m src.modeling.train_batch first.")
    if not VALIDATION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found at {VALIDATION_DATA_PATH}. Run python -m src.modeling.train_batch first."
        )

    model_bundle = joblib.load(MODEL_PATH)
    predictor = model_bundle.get("model") or model_bundle.get("pipeline")
    if predictor is None:
        raise ValueError("Model bundle must contain 'model' or 'pipeline'.")

    threshold = float(model_bundle.get("threshold", 0.5))
    feature_columns = model_bundle.get("feature_columns", FEATURE_COLUMNS)

    validation_df = pd.read_csv(VALIDATION_DATA_PATH)
    X_valid = validation_df[feature_columns].astype(float)
    y_valid = validation_df[TARGET_COLUMN].astype(int)

    y_scores = predictor.predict_proba(X_valid)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    print(f"Threshold: {threshold:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_valid, y_scores):.4f}")
    print(f"AUPRC: {average_precision_score(y_valid, y_scores):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, y_pred))
    print("\nClassification report:")
    print(classification_report(y_valid, y_pred, zero_division=0))
    print("\nThreshold table sample:")
    print(evaluate_thresholds(y_valid, y_scores).sort_values("f1", ascending=False).head(10))


if __name__ == "__main__":
    evaluate()
