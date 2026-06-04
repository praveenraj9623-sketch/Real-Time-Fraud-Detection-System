"""PySpark feature engineering for the credit card fraud dataset.

The source Kaggle dataset contains Time, V1-V28, Amount, and Class. This
module creates the engineered features used consistently by batch training,
threshold tuning, streaming scoring, and the FastAPI service.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from sklearn.model_selection import train_test_split

load_dotenv()

TARGET_COLUMN = "Class"
BASE_FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ENGINEERED_FEATURE_COLUMNS = [
    "amount_log",
    "hour_of_day",
    "is_night_transaction",
    "amount_zscore",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
ALL_COLUMNS = BASE_FEATURE_COLUMNS + [TARGET_COLUMN]

RAW_DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw/creditcard.csv"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed/fraud_features.parquet"))
FEATURE_STATS_PATH = Path(os.getenv("FEATURE_STATS_PATH", "data/processed/feature_stats.json"))


def build_spark_session(app_name: str = "FraudFeatureEngineering") -> SparkSession:
    """Create a local Spark session for feature engineering jobs."""
    return (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "4"))
        .getOrCreate()
    )


def get_spark_schema(include_target: bool = True) -> StructType:
    """Return the schema for raw creditcard.csv records."""
    columns = ALL_COLUMNS if include_target else BASE_FEATURE_COLUMNS
    return StructType([StructField(col, DoubleType(), True) for col in columns])


def get_transaction_event_schema(include_target: bool = True) -> StructType:
    """Return a Kafka transaction event schema with producer metadata."""
    fields = [
        StructField("transaction_id", StringType(), True),
        StructField("processing_timestamp", StringType(), True),
        StructField("merchant_name", StringType(), True),
    ]
    fields.extend(get_spark_schema(include_target=include_target).fields)
    return StructType(fields)


def validate_raw_columns(df: pd.DataFrame) -> None:
    """Validate that the source dataframe has the expected Kaggle columns."""
    missing_columns = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate the Kaggle credit card fraud CSV with pandas."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv and place it there."
        )

    df = pd.read_csv(path)
    validate_raw_columns(df)
    if len(df) < 100:
        raise ValueError(
            "The dataset has too few rows. Replace data/raw/creditcard.csv with the full Kaggle file."
        )
    return df[ALL_COLUMNS].copy()


def add_engineered_features(
    df: DataFrame,
    amount_mean: float | None = None,
    amount_std: float | None = None,
) -> Tuple[DataFrame, Dict[str, float]]:
    """Add fraud-focused amount and time features to a Spark dataframe."""
    if amount_mean is None or amount_std is None:
        stats = df.select(
            F.avg("Amount").alias("amount_mean"),
            F.stddev_pop("Amount").alias("amount_std"),
        ).collect()[0]
        amount_mean = float(stats["amount_mean"] or 0.0)
        amount_std = float(stats["amount_std"] or 0.0)

    safe_std = amount_std if amount_std > 0 else 1.0
    hour_expr = F.floor(F.pmod(F.col("Time"), F.lit(86400.0)) / F.lit(3600.0)).cast("int")

    engineered_df = (
        df.withColumn("amount_log", F.log1p(F.col("Amount")))
        .withColumn("hour_of_day", hour_expr.cast("double"))
        .withColumn(
            "is_night_transaction",
            F.when((hour_expr >= 22) | (hour_expr < 6), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn("amount_zscore", (F.col("Amount") - F.lit(amount_mean)) / F.lit(safe_std))
    )

    return engineered_df, {"amount_mean": amount_mean, "amount_std": amount_std}


def compute_dataset_statistics(df: DataFrame, amount_stats: Dict[str, float]) -> Dict[str, Any]:
    """Compute dataset and class-distribution statistics for reporting."""
    row_count = int(df.count())
    distribution_rows = (
        df.groupBy(TARGET_COLUMN)
        .agg(F.count("*").alias("count"))
        .orderBy(TARGET_COLUMN)
        .collect()
    )
    class_distribution = {
        str(int(row[TARGET_COLUMN])): int(row["count"]) for row in distribution_rows if row[TARGET_COLUMN] is not None
    }
    fraud_count = class_distribution.get("1", 0)
    non_fraud_count = class_distribution.get("0", 0)

    amount_summary = df.select(
        F.min("Amount").alias("amount_min"),
        F.avg("Amount").alias("amount_avg"),
        F.expr("percentile_approx(Amount, 0.5)").alias("amount_median"),
        F.max("Amount").alias("amount_max"),
    ).collect()[0]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "class_distribution": class_distribution,
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_rate": fraud_count / row_count if row_count else 0.0,
        "amount_mean": amount_stats["amount_mean"],
        "amount_std": amount_stats["amount_std"],
        "amount_min": float(amount_summary["amount_min"] or 0.0),
        "amount_avg": float(amount_summary["amount_avg"] or 0.0),
        "amount_median": float(amount_summary["amount_median"] or 0.0),
        "amount_max": float(amount_summary["amount_max"] or 0.0),
        "feature_columns": FEATURE_COLUMNS,
    }


def run_feature_engineering(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
    stats_path: str | Path = FEATURE_STATS_PATH,
    engine: str = "pandas",
) -> Dict[str, Any]:
    """Load creditcard.csv, engineer features, save parquet, and return stats."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    stats_path = Path(stats_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    if engine == "pandas":
        return run_pandas_feature_engineering(input_path, output_path, stats_path)

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    raw_df = (
        spark.read.option("header", True)
        .schema(get_spark_schema(include_target=True))
        .csv(str(input_path))
    )

    engineered_df, amount_stats = add_engineered_features(raw_df)
    engineered_df = engineered_df.select(FEATURE_COLUMNS + [TARGET_COLUMN])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.write.mode("overwrite").parquet(str(output_path))

    stats = compute_dataset_statistics(engineered_df, amount_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Feature engineering complete.")
    print(f"Saved processed parquet to: {output_path}")
    print(f"Saved feature statistics to: {stats_path}")
    print("Class distribution:")
    for label, count in sorted(stats["class_distribution"].items()):
        label_name = "fraud" if label == "1" else "legitimate"
        print(f"  Class {label} ({label_name}): {count:,}")
    print(f"Fraud rate: {stats['fraud_rate']:.6f}")

    spark.stop()
    return stats


def run_pandas_feature_engineering(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
    stats_path: str | Path = FEATURE_STATS_PATH,
) -> Dict[str, Any]:
    """Create the same engineered features with pandas for local Windows runs."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    stats_path = Path(stats_path)
    df = load_creditcard_csv(input_path)

    amount_mean = float(df["Amount"].mean())
    amount_std = float(df["Amount"].std(ddof=0) or 1.0)
    hour_of_day = ((df["Time"] % 86400) // 3600).astype(int)

    df["amount_log"] = df["Amount"].clip(lower=0).apply(math.log1p)
    df["hour_of_day"] = hour_of_day.astype(float)
    df["is_night_transaction"] = ((hour_of_day >= 22) | (hour_of_day < 6)).astype(float)
    df["amount_zscore"] = (df["Amount"] - amount_mean) / amount_std

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    df[FEATURE_COLUMNS + [TARGET_COLUMN]].to_parquet(output_path, index=False)

    class_counts = df[TARGET_COLUMN].astype(int).value_counts().sort_index().to_dict()
    row_count = len(df)
    fraud_count = int(class_counts.get(1, 0))
    non_fraud_count = int(class_counts.get(0, 0))
    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "class_distribution": {str(key): int(value) for key, value in class_counts.items()},
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_rate": fraud_count / row_count if row_count else 0.0,
        "amount_mean": amount_mean,
        "amount_std": amount_std,
        "amount_min": float(df["Amount"].min()),
        "amount_avg": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "amount_max": float(df["Amount"].max()),
        "feature_columns": FEATURE_COLUMNS,
        "engine": "pandas",
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Feature engineering complete.")
    print(f"Saved processed parquet to: {output_path}")
    print(f"Saved feature statistics to: {stats_path}")
    print("Class distribution:")
    for label, count in sorted(stats["class_distribution"].items()):
        label_name = "fraud" if label == "1" else "legitimate"
        print(f"  Class {label} ({label_name}): {count:,}")
    print(f"Fraud rate: {stats['fraud_rate']:.6f}")
    return stats


def load_feature_stats(path: str | Path = FEATURE_STATS_PATH) -> Dict[str, Any]:
    """Load persisted feature statistics for inference-time feature engineering."""
    path = Path(path)
    if not path.exists():
        return {"amount_mean": 0.0, "amount_std": 1.0, "feature_columns": FEATURE_COLUMNS}
    return json.loads(path.read_text(encoding="utf-8"))


def engineer_transaction_features(
    transaction: Dict[str, Any],
    feature_stats: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Apply the same feature engineering to a single transaction JSON payload."""
    feature_stats = feature_stats or load_feature_stats()
    missing = [col for col in BASE_FEATURE_COLUMNS if col not in transaction]
    if missing:
        raise ValueError(f"Transaction missing required raw features: {missing}")

    row: Dict[str, float] = {col: float(transaction[col]) for col in BASE_FEATURE_COLUMNS}
    amount = row["Amount"]
    time_value = row["Time"]
    amount_mean = float(feature_stats.get("amount_mean", 0.0) or 0.0)
    amount_std = float(feature_stats.get("amount_std", 1.0) or 1.0)
    safe_std = amount_std if amount_std > 0 else 1.0

    hour_of_day = int((time_value % 86400) // 3600)
    row["amount_log"] = math.log1p(max(amount, 0.0))
    row["hour_of_day"] = float(hour_of_day)
    row["is_night_transaction"] = 1.0 if hour_of_day >= 22 or hour_of_day < 6 else 0.0
    row["amount_zscore"] = (amount - amount_mean) / safe_std
    return row


def build_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train-test split for fraud classification."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_columns() -> List[str]:
    """Return model feature columns in training/inference order."""
    return FEATURE_COLUMNS.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fraud feature parquet with PySpark.")
    parser.add_argument("--input", default=str(RAW_DATA_PATH), help="Path to data/raw/creditcard.csv")
    parser.add_argument("--output", default=str(PROCESSED_DATA_PATH), help="Processed parquet output path")
    parser.add_argument("--stats-output", default=str(FEATURE_STATS_PATH), help="Feature statistics JSON path")
    parser.add_argument("--engine", choices=["pandas", "spark"], default="pandas", help="Local engine to create features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_engineering(args.input, args.output, args.stats_output, engine=args.engine)
