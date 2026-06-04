"""Spark Structured Streaming consumer that scores Kafka transactions through FastAPI."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json

from src.processing.feature_engineering import BASE_FEATURE_COLUMNS, TARGET_COLUMN, get_transaction_event_schema
from src.storage.mongodb_client import MongoDBClient

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
FASTAPI_SCORE_URL = os.getenv("FASTAPI_SCORE_URL", "http://127.0.0.1:8001/score")
FRAUD_ALERT_THRESHOLD = float(os.getenv("STREAM_FRAUD_ALERT_THRESHOLD", "0.5"))
CHECKPOINT_LOCATION = os.getenv("SPARK_CHECKPOINT_LOCATION", ".spark-checkpoints/fraud-consumer")


def clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON and Mongo-friendly scalars."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def row_to_payload(row: pd.Series) -> Dict[str, Any]:
    """Build the API payload from a Spark/Pandas row."""
    payload: Dict[str, Any] = {}
    for key, value in row.to_dict().items():
        cleaned = clean_value(value)
        if cleaned is not None:
            payload[key] = cleaned
    missing = [column for column in BASE_FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(f"Streaming row missing required scoring fields: {missing}")
    return payload


def call_score_api(session: requests.Session, payload: Dict[str, Any], score_url: str) -> Dict[str, Any]:
    """Call the FastAPI /score endpoint for one transaction."""
    response = session.post(score_url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def process_pandas_batch(
    pandas_df: pd.DataFrame,
    batch_id: int,
    session: requests.Session,
    mongo_client: MongoDBClient,
    score_url: str,
    alert_threshold: float,
) -> None:
    """Score each row in a micro-batch and write fraud alerts to MongoDB."""
    if pandas_df.empty:
        print(f"Batch {batch_id}: rows=0 scored=0 flagged=0")
        return

    scored_count = 0
    flagged_alerts: List[Dict[str, Any]] = []
    probabilities: List[float] = []

    for _, row in pandas_df.iterrows():
        payload = row_to_payload(row)
        try:
            score = call_score_api(session, payload, score_url)
        except Exception as exc:
            print(f"Batch {batch_id}: failed to score transaction {payload.get('transaction_id')}: {exc}")
            continue

        scored_count += 1
        probability = float(score.get("fraud_probability", 0.0))
        probabilities.append(probability)
        if probability > alert_threshold:
            alert = {
                **payload,
                **score,
                "stream_alert_threshold": alert_threshold,
                "alert_reason": f"fraud_probability_above_{alert_threshold}",
                "scored_at_utc": datetime.now(timezone.utc),
            }
            if TARGET_COLUMN in payload:
                alert["actual_class"] = payload[TARGET_COLUMN]
            flagged_alerts.append(alert)

    inserted = mongo_client.insert_many_fraud_alerts(flagged_alerts)
    avg_probability = float(np.mean(probabilities)) if probabilities else 0.0
    max_probability = float(np.max(probabilities)) if probabilities else 0.0
    print(
        f"Batch {batch_id}: rows={len(pandas_df)} scored={scored_count} "
        f"flagged={inserted} avg_probability={avg_probability:.4f} max_probability={max_probability:.4f}"
    )


def build_spark_session() -> SparkSession:
    """Create Spark session with Kafka connector."""
    return (
        SparkSession.builder.appName("RealTimeFraudConsumer")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "2"))
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .getOrCreate()
    )


def start_stream(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic: str = KAFKA_TOPIC,
    score_url: str = FASTAPI_SCORE_URL,
    alert_threshold: float = FRAUD_ALERT_THRESHOLD,
) -> None:
    """Start Spark Structured Streaming from Kafka and write MongoDB fraud alerts."""
    Path(CHECKPOINT_LOCATION).mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    mongo_client = MongoDBClient()

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    schema = get_transaction_event_schema(include_target=True)
    parsed_stream = (
        raw_stream.selectExpr("CAST(value AS STRING) AS json_value")
        .select(from_json(col("json_value"), schema).alias("data"))
        .select("data.*")
    )

    def process_batch(batch_df, batch_id: int) -> None:
        pandas_df = batch_df.toPandas()
        process_pandas_batch(
            pandas_df=pandas_df,
            batch_id=batch_id,
            session=session,
            mongo_client=mongo_client,
            score_url=score_url,
            alert_threshold=alert_threshold,
        )

    query = (
        parsed_stream.writeStream.foreachBatch(process_batch)
        .outputMode("append")
        .trigger(processingTime="10 seconds")
        .option("checkpointLocation", CHECKPOINT_LOCATION)
        .start()
    )

    print("Spark fraud consumer started.")
    print(f"Kafka: {bootstrap_servers} topic={topic}")
    print(f"Scoring endpoint: {score_url}")
    print(f"MongoDB collection: {mongo_client.collection_name}")
    query.awaitTermination()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume Kafka transactions and score them with FastAPI.")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=KAFKA_TOPIC, help="Kafka topic")
    parser.add_argument("--score-url", default=FASTAPI_SCORE_URL, help="FastAPI /score URL")
    parser.add_argument("--alert-threshold", type=float, default=FRAUD_ALERT_THRESHOLD, help="Mongo alert cutoff")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_stream(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        score_url=args.score_url,
        alert_threshold=args.alert_threshold,
    )
