"""Local dashboard demo feeder without Kafka or Docker.

This script reads creditcard.csv, calls the running FastAPI /score endpoint,
and writes high-risk transactions to MongoDB. Use it when MongoDB is installed
locally but Docker/Kafka is not available.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from src.processing.feature_engineering import ALL_COLUMNS, BASE_FEATURE_COLUMNS, TARGET_COLUMN
from src.storage.mongodb_client import MongoDBClient
from src.utils.risk import clip_probability, extract_transaction_amount, normalize_alert_document

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw/creditcard.csv"))
FASTAPI_SCORE_URL = os.getenv("FASTAPI_SCORE_URL", "http://127.0.0.1:8001/score")

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
]


def coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert one CSV row into a transaction dictionary."""
    event: Dict[str, Any] = {}
    amount = extract_transaction_amount(row)
    amount_source = "raw_creditcard_csv" if amount is not None else "fallback_missing_amount"
    for column in ALL_COLUMNS:
        if column == "Amount":
            event[column] = float(amount if amount is not None else 0.0)
            continue
        value = row.get(column)
        if value is None or value == "":
            continue
        event[column] = int(float(value)) if column == TARGET_COLUMN else float(value)
    event["amount_source"] = amount_source

    missing = [column for column in BASE_FEATURE_COLUMNS if column not in event]
    if missing:
        raise ValueError(f"CSV row missing required fields: {missing}")
    return event


def add_demo_metadata(event: Dict[str, Any]) -> Dict[str, Any]:
    """Add fields normally created by the Kafka producer."""
    simulated_event_time = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 7 * 24 * 60))
    event["transaction_id"] = str(uuid.uuid4())
    event["event_time_utc"] = simulated_event_time.isoformat()
    event["transaction_time_utc"] = simulated_event_time.isoformat()
    event["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
    event["merchant_name"] = random.choice(MERCHANT_NAMES)
    return event


def run_demo_feeder(
    data_path: Path,
    score_url: str,
    max_rows: int,
    alert_threshold: float,
    delay: float,
) -> None:
    """Score CSV rows and insert fraud alerts into MongoDB."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    mongo_client = MongoDBClient()
    session = requests.Session()
    scanned = 0
    inserted = 0

    print(f"Reading: {data_path}")
    print(f"Scoring endpoint: {score_url}")
    print(f"MongoDB collection: {mongo_client.collection_name}")
    print(f"Alert threshold: {alert_threshold}")

    with data_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if scanned >= max_rows:
                break

            transaction = add_demo_metadata(coerce_csv_row(row))
            try:
                response = session.post(score_url, json=transaction, timeout=10)
                response.raise_for_status()
                score = response.json()
            except Exception as exc:
                print(f"Failed to score row {scanned}: {exc}")
                scanned += 1
                continue

            probability = clip_probability(score.get("fraud_probability", 0.0))
            if probability >= alert_threshold:
                alert = normalize_alert_document(
                    {
                        **transaction,
                        **score,
                        "actual_class": transaction.get(TARGET_COLUMN),
                        "stream_alert_threshold": alert_threshold,
                        "alert_reason": "local_demo_feeder",
                        "scored_at_utc": datetime.now(timezone.utc),
                    },
                    save_threshold=alert_threshold,
                )
                mongo_client.insert_fraud_alert(alert)
                inserted += 1

            scanned += 1
            if scanned % 100 == 0:
                print(f"Scanned {scanned:,} rows, inserted {inserted:,} alerts")

            if delay > 0:
                time.sleep(delay)

    print(f"Done. Scanned {scanned:,} rows and inserted {inserted:,} alerts.")
    print("Refresh Streamlit at http://localhost:8502")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate the dashboard without Kafka.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to data/raw/creditcard.csv")
    parser.add_argument("--score-url", default=FASTAPI_SCORE_URL, help="FastAPI /score URL")
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum CSV rows to scan")
    parser.add_argument("--alert-threshold", type=float, default=0.5, help="Minimum fraud probability to save")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay between rows")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo_feeder(
        data_path=Path(args.data),
        score_url=args.score_url,
        max_rows=args.max_rows,
        alert_threshold=args.alert_threshold,
        delay=args.delay,
    )
