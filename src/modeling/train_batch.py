"""Batch training for XGBoost and Isolation Forest fraud detectors."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.modeling.threshold_tuner import evaluate_thresholds
from src.processing.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_STATS_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    build_spark_session,
    load_feature_stats,
    run_feature_engineering,
)

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_classifier.joblib"))
XGBOOST_MODEL_PATH = Path(os.getenv("XGBOOST_MODEL_PATH", "models/xgboost_fraud_detector.joblib"))
ANOMALY_MODEL_PATH = Path(os.getenv("ANOMALY_MODEL_PATH", "models/anomaly_detector.joblib"))
ISOLATION_MODEL_PATH = Path(os.getenv("ISOLATION_MODEL_PATH", "models/isolation_forest_detector.joblib"))
VALIDATION_DATA_PATH = Path(os.getenv("VALIDATION_DATA_PATH", "models/validation_data.csv"))
OPTIMAL_THRESHOLD_PATH = Path(os.getenv("OPTIMAL_THRESHOLD_PATH", "models/optimal_threshold.txt"))
TRAINING_STATS_PATH = Path(os.getenv("TRAINING_STATS_PATH", "models/training_stats.json"))
MODEL_METRICS_PATH = Path(os.getenv("MODEL_METRICS_PATH", "models/model_metrics.json"))
ARTIFACTS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", "models/artifacts"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-realtime")
MODEL_REGISTRY_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "FraudDetector")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def json_default(value: Any) -> Any:
    """JSON serializer for numpy and pandas scalar types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def ensure_processed_dataset(processed_path: Path = PROCESSED_DATA_PATH) -> None:
    """Create the processed parquet if it is not already available."""
    if processed_path.exists():
        return
    print(f"Processed dataset not found at {processed_path}. Running feature engineering first...")
    run_feature_engineering(output_path=processed_path)


def load_processed_dataset(processed_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed parquet into pandas.

    Pandas/pyarrow is the preferred path here because it avoids a second Spark
    JVM startup during model training, which is fragile on some Windows setups.
    """
    ensure_processed_dataset(processed_path)
    try:
        df = pd.read_parquet(processed_path)
    except Exception:
        spark = build_spark_session("FraudBatchTraining")
        spark.sparkContext.setLogLevel("WARN")
        df = spark.read.parquet(str(processed_path)).toPandas()
        spark.stop()

    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Processed parquet is missing columns: {missing}")
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/validation split."""
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def build_dataset_stats_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Build fallback stats when feature_stats.json is unavailable."""
    class_counts = df[TARGET_COLUMN].astype(int).value_counts().sort_index().to_dict()
    row_count = len(df)
    fraud_count = int(class_counts.get(1, 0))
    non_fraud_count = int(class_counts.get(0, 0))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "class_distribution": {str(key): int(value) for key, value in class_counts.items()},
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_rate": fraud_count / row_count if row_count else 0.0,
        "amount_mean": float(df["Amount"].mean()),
        "amount_std": float(df["Amount"].std(ddof=0) or 1.0),
        "amount_min": float(df["Amount"].min()),
        "amount_avg": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "amount_max": float(df["Amount"].max()),
        "feature_columns": FEATURE_COLUMNS,
    }


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Oversample the minority fraud class using SMOTE."""
    positive_count = int((y_train == 1).sum())
    if positive_count < 2:
        return X_train, y_train, {
            "enabled": False,
            "reason": "Need at least two fraud rows for SMOTE.",
            "minority_count_before": positive_count,
        }

    k_neighbors = min(5, positive_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=TARGET_COLUMN).astype(int)

    return X_resampled, y_resampled, {
        "enabled": True,
        "sampling_strategy": "auto",
        "k_neighbors": k_neighbors,
        "minority_count_before": positive_count,
        "minority_count_after": int((y_resampled == 1).sum()),
        "majority_count_after": int((y_resampled == 0).sum()),
    }


def best_threshold_for_scores(y_true: pd.Series, scores: np.ndarray) -> Dict[str, float]:
    """Find the best F1 threshold from the shared dense tuning grid."""
    results = evaluate_thresholds(y_true, scores)
    best = results.sort_values("f1", ascending=False).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "true_negatives": int(best.get("true_negatives", 0)),
        "false_positives": int(best.get("false_positives", 0)),
        "false_negatives": int(best.get("false_negatives", 0)),
        "true_positives": int(best.get("true_positives", 0)),
        "predicted_fraud_count": int(best.get("predicted_fraud_count", 0)),
        "business_cost": float(best.get("business_cost", 0.0)),
    }


def safe_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, scores))


def evaluate_model(y_true: pd.Series, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute fraud metrics at a selected threshold."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, scores),
        "auprc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_fraud_count": int(y_pred.sum()),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Legit", "Fraud"])
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve(y_true: pd.Series, scores: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_precision_recall_curve(y_true: pd.Series, scores: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def log_artifacts(confusion_path: Path, roc_path: Path, pr_path: Path) -> None:
    for artifact in [confusion_path, roc_path, pr_path]:
        if artifact.exists():
            mlflow.log_artifact(str(artifact))


def stringify_params(params: Dict[str, Any]) -> Dict[str, str]:
    """Convert model parameters to MLflow-friendly strings."""
    return {str(key): str(value)[:500] for key, value in params.items()}


def train_xgboost(
    X_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
    smote_info: Dict[str, Any],
    dataset_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str]:
    """Train and log the XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="xgboost_fraud_classifier") as run:
        model.fit(X_train_resampled, y_train_resampled)
        scores = model.predict_proba(X_valid)[:, 1]
        threshold_info = best_threshold_for_scores(y_valid, scores)
        threshold = threshold_info["threshold"]
        metrics = evaluate_model(y_valid, scores, threshold)
        y_pred = (scores >= threshold).astype(int)

        confusion_path = ARTIFACTS_DIR / "xgboost_confusion_matrix.png"
        roc_path = ARTIFACTS_DIR / "xgboost_roc_curve.png"
        pr_path = ARTIFACTS_DIR / "xgboost_precision_recall_curve.png"
        save_confusion_matrix(y_valid, y_pred, "XGBoost Confusion Matrix", confusion_path)
        save_roc_curve(y_valid, scores, "XGBoost ROC Curve", roc_path)
        save_precision_recall_curve(y_valid, scores, "XGBoost Precision-Recall Curve", pr_path)

        params = {
            **{f"xgb_{key}": value for key, value in model.get_params().items()},
            "feature_count": len(FEATURE_COLUMNS),
            "features": ",".join(FEATURE_COLUMNS),
            "smote_enabled": smote_info.get("enabled"),
            "smote_k_neighbors": smote_info.get("k_neighbors"),
            "train_rows_after_smote": len(X_train_resampled),
            "validation_rows": len(X_valid),
            "original_fraud_rate": dataset_stats.get("fraud_rate"),
        }
        mlflow.log_params(stringify_params(params))
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")
        log_artifacts(confusion_path, roc_path, pr_path)

        bundle = {
            "model_type": "xgboost",
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": threshold,
            "threshold_tuning": threshold_info,
            "metrics": metrics,
            "feature_stats": dataset_stats,
            "model_version": run.info.run_id,
            "mlflow_run_id": run.info.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "confusion_matrix": str(confusion_path),
                "roc_curve": str(roc_path),
                "precision_recall_curve": str(pr_path),
            },
        }

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        if XGBOOST_MODEL_PATH != MODEL_PATH:
            joblib.dump(bundle, XGBOOST_MODEL_PATH)
        return bundle, run.info.run_id, f"runs:/{run.info.run_id}/model"


def normalize_anomaly_scores(raw_scores: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    """Convert raw anomaly scores to a 0-1 fraud-risk scale."""
    denominator = score_max - score_min
    if denominator <= 1e-12:
        return np.zeros_like(raw_scores, dtype=float)
    return np.clip((raw_scores - score_min) / denominator, 0.0, 1.0)


def train_isolation_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    dataset_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str]:
    """Train and log the Isolation Forest anomaly detector."""
    contamination = min(max(float(dataset_stats.get("fraud_rate", 0.002)), 0.0005), 0.05)
    normal_train = X_train[y_train == 0]
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=350,
                    max_samples="auto",
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name="isolation_forest_anomaly_detector") as run:
        pipeline.fit(normal_train)
        raw_valid_scores = -pipeline.decision_function(X_valid)
        score_min = float(raw_valid_scores.min())
        score_max = float(raw_valid_scores.max())
        scores = normalize_anomaly_scores(raw_valid_scores, score_min, score_max)
        threshold_info = best_threshold_for_scores(y_valid, scores)
        threshold = threshold_info["threshold"]
        metrics = evaluate_model(y_valid, scores, threshold)
        y_pred = (scores >= threshold).astype(int)

        confusion_path = ARTIFACTS_DIR / "isolation_forest_confusion_matrix.png"
        roc_path = ARTIFACTS_DIR / "isolation_forest_roc_curve.png"
        pr_path = ARTIFACTS_DIR / "isolation_forest_precision_recall_curve.png"
        save_confusion_matrix(y_valid, y_pred, "Isolation Forest Confusion Matrix", confusion_path)
        save_roc_curve(y_valid, scores, "Isolation Forest ROC Curve", roc_path)
        save_precision_recall_curve(y_valid, scores, "Isolation Forest Precision-Recall Curve", pr_path)

        params = {
            **{f"isolation_{key}": value for key, value in pipeline.named_steps["model"].get_params().items()},
            "feature_count": len(FEATURE_COLUMNS),
            "features": ",".join(FEATURE_COLUMNS),
            "training_rows_normal_only": len(normal_train),
            "validation_rows": len(X_valid),
            "score_min": score_min,
            "score_max": score_max,
            "contamination": contamination,
        }
        mlflow.log_params(stringify_params(params))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        log_artifacts(confusion_path, roc_path, pr_path)

        bundle = {
            "model_type": "isolation_forest",
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": threshold,
            "score_min": score_min,
            "score_max": score_max,
            "metrics": metrics,
            "feature_stats": dataset_stats,
            "model_version": run.info.run_id,
            "mlflow_run_id": run.info.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "confusion_matrix": str(confusion_path),
                "roc_curve": str(roc_path),
                "precision_recall_curve": str(pr_path),
            },
        }

        ANOMALY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, ANOMALY_MODEL_PATH)
        if ISOLATION_MODEL_PATH != ANOMALY_MODEL_PATH:
            joblib.dump(bundle, ISOLATION_MODEL_PATH)
        return bundle, run.info.run_id, f"runs:/{run.info.run_id}/model"


def save_training_outputs(
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    xgb_bundle: Dict[str, Any],
    isolation_bundle: Dict[str, Any],
    best_model_name: str,
    dataset_stats: Dict[str, Any],
    smote_info: Dict[str, Any],
) -> None:
    """Persist validation data, threshold, and dashboard-friendly metadata."""
    VALIDATION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_df = X_valid.copy()
    validation_df[TARGET_COLUMN] = y_valid.values
    validation_df.to_csv(VALIDATION_DATA_PATH, index=False)

    OPTIMAL_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPTIMAL_THRESHOLD_PATH.write_text(f"{xgb_bundle['threshold']:.8f}\n", encoding="utf-8")

    metrics_payload = {
        "best_model": best_model_name,
        "registered_model_name": MODEL_REGISTRY_NAME,
        "xgboost": xgb_bundle["metrics"],
        "isolation_forest": isolation_bundle["metrics"],
        "artifacts": {
            "xgboost": xgb_bundle["artifacts"],
            "isolation_forest": isolation_bundle["artifacts"],
        },
    }
    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, default=json_default), encoding="utf-8")

    training_stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_stats,
        "smote": smote_info,
        "validation_rows": len(X_valid),
        "model_metrics": metrics_payload,
        "optimal_threshold": xgb_bundle["threshold"],
    }
    TRAINING_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_STATS_PATH.write_text(json.dumps(training_stats, indent=2, default=json_default), encoding="utf-8")


def register_best_model(best_model_uri: str, best_model_name: str) -> str:
    """Register the winning MLflow model as FraudDetector."""
    try:
        registered = mlflow.register_model(best_model_uri, MODEL_REGISTRY_NAME)
        message = (
            f"Registered {best_model_name} as {MODEL_REGISTRY_NAME} "
            f"version {registered.version}."
        )
        print(message)
        return message
    except Exception as exc:
        message = f"Could not register model in MLflow registry: {exc}"
        print(message)
        return message


def train() -> None:
    """Train both requested models, log MLflow runs, and save model artifacts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_processed_dataset()
    dataset_stats = load_feature_stats(FEATURE_STATS_PATH)
    if "row_count" not in dataset_stats:
        dataset_stats = build_dataset_stats_from_dataframe(df)

    X_train, X_valid, y_train, y_valid = split_dataset(df)
    original_negative_count = int((y_train == 0).sum())
    original_positive_count = int((y_train == 1).sum())
    scale_pos_weight = original_negative_count / max(original_positive_count, 1)
    X_train_resampled, y_train_resampled, smote_info = apply_smote(X_train, y_train)

    print("Training XGBoost classifier...")
    xgb_bundle, _xgb_run_id, xgb_model_uri = train_xgboost(
        X_train_resampled,
        y_train_resampled,
        X_valid,
        y_valid,
        scale_pos_weight,
        smote_info,
        dataset_stats,
    )

    print("Training Isolation Forest anomaly detector...")
    isolation_bundle, _iforest_run_id, iforest_model_uri = train_isolation_forest(
        X_train,
        y_train,
        X_valid,
        y_valid,
        dataset_stats,
    )

    if xgb_bundle["metrics"]["auprc"] >= isolation_bundle["metrics"]["auprc"]:
        best_model_name = "xgboost"
        best_model_uri = xgb_model_uri
    else:
        best_model_name = "isolation_forest"
        best_model_uri = iforest_model_uri

    registration_message = register_best_model(best_model_uri, best_model_name)
    save_training_outputs(
        X_valid,
        y_valid,
        xgb_bundle,
        isolation_bundle,
        best_model_name,
        dataset_stats,
        smote_info,
    )

    print("Training completed.")
    print(f"Saved XGBoost model to: {MODEL_PATH}")
    print(f"Saved Isolation Forest model to: {ANOMALY_MODEL_PATH}")
    print(f"Saved validation data to: {VALIDATION_DATA_PATH}")
    print(f"Saved optimal threshold to: {OPTIMAL_THRESHOLD_PATH}")
    print(f"Best model by AUPRC: {best_model_name}")
    print(registration_message)
    print("XGBoost metrics:")
    for key, value in xgb_bundle["metrics"].items():
        print(f"  {key}: {value:.6f}")
    print("Isolation Forest metrics:")
    for key, value in isolation_bundle["metrics"].items():
        print(f"  {key}: {value:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection batch models.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    train()
