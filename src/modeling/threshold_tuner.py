"""Tune the fraud decision threshold for the trained XGBoost model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score

from src.processing.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_classifier.joblib"))
VALIDATION_DATA_PATH = Path(os.getenv("VALIDATION_DATA_PATH", "models/validation_data.csv"))
OPTIMAL_THRESHOLD_PATH = Path(os.getenv("OPTIMAL_THRESHOLD_PATH", "models/optimal_threshold.txt"))
THRESHOLD_METRICS_PATH = Path(os.getenv("THRESHOLD_METRICS_PATH", "models/threshold_metrics.csv"))
THRESHOLD_PLOT_PATH = Path(os.getenv("THRESHOLD_PLOT_PATH", "models/precision_recall_tradeoff.png"))
THRESHOLD_SUMMARY_PATH = Path(os.getenv("THRESHOLD_SUMMARY_PATH", "models/threshold_tuning_summary.json"))
TRAINING_STATS_PATH = Path(os.getenv("TRAINING_STATS_PATH", "models/training_stats.json"))


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    predicted_fraud_count: int = 0
    business_cost: float = 0.0


def default_threshold_grid() -> np.ndarray:
    """Return a dense threshold grid from 0.01 through 0.999."""
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.90, 90),
                np.linspace(0.91, 0.99, 10),
                np.linspace(0.991, 0.999, 9),
            ]
        )
    )


def evaluate_thresholds(
    y_true,
    y_scores,
    thresholds=None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 20.0,
) -> pd.DataFrame:
    """Evaluate precision, recall, and F1 across candidate thresholds."""
    if thresholds is None:
        thresholds = default_threshold_grid()

    rows = []
    y_true_array = np.asarray(y_true).astype(int)
    for threshold in thresholds:
        y_pred = (np.asarray(y_scores) >= threshold).astype(int)
        tn = int(((y_true_array == 0) & (y_pred == 0)).sum())
        fp = int(((y_true_array == 0) & (y_pred == 1)).sum())
        fn = int(((y_true_array == 1) & (y_pred == 0)).sum())
        tp = int(((y_true_array == 1) & (y_pred == 1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "predicted_fraud_count": int(y_pred.sum()),
                "false_positive_cost": float(false_positive_cost),
                "false_negative_cost": float(false_negative_cost),
                "business_cost": float(fp * false_positive_cost + fn * false_negative_cost),
            }
        )
    return pd.DataFrame(rows)


def find_best_threshold(y_true, y_scores, metric: str = "f1") -> ThresholdResult:
    """Find the threshold that maximizes a selected metric."""
    results = evaluate_thresholds(y_true, y_scores)
    if metric not in results.columns:
        raise ValueError(f"metric must be one of {list(results.columns)}")

    best_row = results.sort_values(metric, ascending=False).iloc[0]
    return ThresholdResult(
        threshold=float(best_row["threshold"]),
        precision=float(best_row["precision"]),
        recall=float(best_row["recall"]),
        f1=float(best_row["f1"]),
        false_positives=int(best_row.get("false_positives", 0)),
        false_negatives=int(best_row.get("false_negatives", 0)),
        true_positives=int(best_row.get("true_positives", 0)),
        true_negatives=int(best_row.get("true_negatives", 0)),
        predicted_fraud_count=int(best_row.get("predicted_fraud_count", 0)),
        business_cost=float(best_row.get("business_cost", 0.0)),
    )


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the saved XGBoost model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run python -m src.modeling.train_batch first.")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected a model bundle dict at {model_path}")
    return bundle


def load_validation_data(path: Path = VALIDATION_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the validation set saved by train_batch.py."""
    if not path.exists():
        raise FileNotFoundError(f"Validation data not found at {path}. Run python -m src.modeling.train_batch first.")
    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Validation data missing columns: {missing}")
    return df[FEATURE_COLUMNS].astype(float), df[TARGET_COLUMN].astype(int)


def predict_probabilities(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Score validation rows with the XGBoost model bundle."""
    if "model" in bundle:
        return bundle["model"].predict_proba(X)[:, 1]
    if "pipeline" in bundle:
        return bundle["pipeline"].predict_proba(X)[:, 1]
    raise ValueError("Model bundle must contain either 'model' or 'pipeline'.")


def save_tradeoff_plot(results: pd.DataFrame, output_path: Path = THRESHOLD_PLOT_PATH) -> None:
    """Plot precision, recall, and F1 against threshold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["threshold"], results["precision"], marker="o", label="Precision")
    ax.plot(results["threshold"], results["recall"], marker="o", label="Recall")
    ax.plot(results["threshold"], results["f1"], marker="o", label="F1")
    best = results.sort_values("f1", ascending=False).iloc[0]
    ax.axvline(best["threshold"], color="black", linestyle="--", linewidth=1, label="Best F1 threshold")
    ax.set_xlabel("Fraud probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Tradeoff by Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def result_to_dict(result: ThresholdResult) -> dict[str, Any]:
    """Serialize a threshold result."""
    return {
        "threshold": result.threshold,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "true_negatives": result.true_negatives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "true_positives": result.true_positives,
        "predicted_fraud_count": result.predicted_fraud_count,
        "business_cost": result.business_cost,
    }


def update_training_stats(best_f1: ThresholdResult, best_cost: ThresholdResult) -> None:
    """Persist tuned threshold metadata for the API and dashboard."""
    if not TRAINING_STATS_PATH.exists():
        return
    payload = json.loads(TRAINING_STATS_PATH.read_text(encoding="utf-8"))
    payload["optimal_threshold"] = best_f1.threshold
    payload["threshold_tuning"] = {
        "best_f1": result_to_dict(best_f1),
        "best_business_cost": result_to_dict(best_cost),
        "source": str(THRESHOLD_METRICS_PATH),
        "summary": str(THRESHOLD_SUMMARY_PATH),
    }
    TRAINING_STATS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tune_threshold(
    model_path: Path = MODEL_PATH,
    validation_path: Path = VALIDATION_DATA_PATH,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 20.0,
) -> ThresholdResult:
    """Tune threshold over a dense grid and save F1 plus business-cost choices."""
    bundle = load_model_bundle(model_path)
    X_valid, y_valid = load_validation_data(validation_path)
    scores = predict_probabilities(bundle, X_valid)

    thresholds = default_threshold_grid()
    results = evaluate_thresholds(
        y_valid,
        scores,
        thresholds=thresholds,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    best_f1_row = results.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    best_cost_row = results.sort_values(["business_cost", "false_negatives", "false_positives"]).iloc[0]
    best = ThresholdResult(
        threshold=float(best_f1_row["threshold"]),
        precision=float(best_f1_row["precision"]),
        recall=float(best_f1_row["recall"]),
        f1=float(best_f1_row["f1"]),
        false_positives=int(best_f1_row["false_positives"]),
        false_negatives=int(best_f1_row["false_negatives"]),
        true_positives=int(best_f1_row["true_positives"]),
        true_negatives=int(best_f1_row["true_negatives"]),
        predicted_fraud_count=int(best_f1_row["predicted_fraud_count"]),
        business_cost=float(best_f1_row["business_cost"]),
    )
    best_cost = ThresholdResult(
        threshold=float(best_cost_row["threshold"]),
        precision=float(best_cost_row["precision"]),
        recall=float(best_cost_row["recall"]),
        f1=float(best_cost_row["f1"]),
        false_positives=int(best_cost_row["false_positives"]),
        false_negatives=int(best_cost_row["false_negatives"]),
        true_positives=int(best_cost_row["true_positives"]),
        true_negatives=int(best_cost_row["true_negatives"]),
        predicted_fraud_count=int(best_cost_row["predicted_fraud_count"]),
        business_cost=float(best_cost_row["business_cost"]),
    )

    THRESHOLD_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(THRESHOLD_METRICS_PATH, index=False)
    save_tradeoff_plot(results, THRESHOLD_PLOT_PATH)
    summary = {
        "best_f1_threshold": result_to_dict(best),
        "best_business_cost_threshold": result_to_dict(best_cost),
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "threshold_grid_size": int(len(thresholds)),
        "threshold_metrics_path": str(THRESHOLD_METRICS_PATH),
        "tradeoff_plot_path": str(THRESHOLD_PLOT_PATH),
    }
    THRESHOLD_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    OPTIMAL_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPTIMAL_THRESHOLD_PATH.write_text(f"{best.threshold:.8f}\n", encoding="utf-8")

    bundle["threshold"] = best.threshold
    bundle["threshold_tuning"] = {
        "best_f1": result_to_dict(best),
        "best_business_cost": result_to_dict(best_cost),
        "threshold_metrics_path": str(THRESHOLD_METRICS_PATH),
        "tradeoff_plot_path": str(THRESHOLD_PLOT_PATH),
        "summary_path": str(THRESHOLD_SUMMARY_PATH),
    }
    joblib.dump(bundle, model_path)
    update_training_stats(best, best_cost)

    print("Threshold tuning complete.")
    print(f"Optimal threshold: {best.threshold:.6f}")
    print(f"Precision: {best.precision:.6f}")
    print(f"Recall: {best.recall:.6f}")
    print(f"F1: {best.f1:.6f}")
    print(f"Saved threshold metrics to: {THRESHOLD_METRICS_PATH}")
    print(f"Saved threshold summary to: {THRESHOLD_SUMMARY_PATH}")
    print(f"Saved tradeoff plot to: {THRESHOLD_PLOT_PATH}")
    print(f"Saved optimal threshold to: {OPTIMAL_THRESHOLD_PATH}")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost fraud threshold.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to saved XGBoost model bundle")
    parser.add_argument("--validation-data", default=str(VALIDATION_DATA_PATH), help="Validation CSV path")
    parser.add_argument("--false-positive-cost", type=float, default=1.0, help="Business cost for a false alarm")
    parser.add_argument("--false-negative-cost", type=float, default=20.0, help="Business cost for a missed fraud")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_threshold(
        Path(args.model),
        Path(args.validation_data),
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
    )
