"""Kafka producer that publishes credit card transactions as JSON events."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from kafka import KafkaProducer

from src.processing.feature_engineering import ALL_COLUMNS, BASE_FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.risk import extract_transaction_amount

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw/creditcard.csv"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
DEFAULT_DELAY_SECONDS = float(os.getenv("STREAM_SLEEP_SECONDS", "0.1"))

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


def json_serializer(value: Dict[str, Any]) -> bytes:
    """Serialize Kafka messages as UTF-8 JSON."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def build_producer(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS) -> KafkaProducer:
    """Create the Kafka producer."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=json_serializer,
        retries=5,
        linger_ms=20,
    )


def coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert CSV strings into numeric transaction values."""
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
        if column == TARGET_COLUMN:
            event[column] = int(float(value))
        else:
            event[column] = float(value)
    event["amount_source"] = amount_source
    missing = [column for column in BASE_FEATURE_COLUMNS if column not in event]
    if missing:
        raise ValueError(f"CSV row missing required transaction fields: {missing}")
    return event


def build_transaction_event(row: Dict[str, str]) -> Dict[str, Any]:
    """Add simulated real-time metadata to a Kaggle transaction row."""
    event = coerce_csv_row(row)
    event["transaction_id"] = str(uuid.uuid4())
    event["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
    event["merchant_name"] = random.choice(MERCHANT_NAMES)
    return event


def stream_transactions(
    data_path: Path = DATA_PATH,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic: str = KAFKA_TOPIC,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    max_transactions: int | None = None,
) -> None:
    """Read creditcard.csv row by row and publish every transaction to Kafka."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Place creditcard.csv there.")

    producer = build_producer(bootstrap_servers)
    print(f"Publishing transactions from {data_path} to Kafka topic '{topic}'...")
    print(f"Delay between messages: {delay_seconds:.3f} seconds")

    published = 0
    with data_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if max_transactions is not None and published >= max_transactions:
                break
            event = build_transaction_event(row)
            producer.send(topic, value=event)
            published += 1

            if published % 100 == 0:
                producer.flush()
                print(f"Published {published:,} transactions")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    producer.flush()
    producer.close()
    print(f"Finished publishing {published:,} transactions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream creditcard.csv transactions into Kafka.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to data/raw/creditcard.csv")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=KAFKA_TOPIC, help="Kafka topic")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between messages in seconds")
    parser.add_argument("--max-transactions", type=int, default=None, help="Optional cap for demos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stream_transactions(
        data_path=Path(args.data),
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        delay_seconds=args.delay,
        max_transactions=args.max_transactions,
    )
