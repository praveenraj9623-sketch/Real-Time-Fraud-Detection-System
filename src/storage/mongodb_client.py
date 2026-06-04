"""MongoDB client for fraud alert storage and dashboard analytics."""

from __future__ import annotations

import math
import os
from datetime import datetime, time, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from src.utils.risk import normalize_alert_document

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "fraud_detection")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "fraud_alerts")


def clean_for_mongo(value: Any) -> Any:
    """Convert numpy/pandas values and NaN into Mongo-safe values."""
    if isinstance(value, dict):
        return {key: clean_for_mongo(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_for_mongo(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    return value


def serialize_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Mongo documents into JSON/Streamlit-friendly dictionaries."""
    serialized: Dict[str, Any] = {}
    for key, value in document.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            serialized[key] = [
                serialize_document(item) if isinstance(item, dict) else item for item in value
            ]
        elif isinstance(value, dict):
            serialized[key] = serialize_document(value)
        else:
            serialized[key] = value
    return serialized


class MongoDBClient:
    """Small MongoDB access layer for fraud alerts."""

    def __init__(
        self,
        uri: str = MONGO_URI,
        database_name: str = MONGO_DATABASE,
        collection_name: str = MONGO_COLLECTION,
    ) -> None:
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = MongoClient(uri)
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.collection.create_index([("transaction_id", ASCENDING)])
        self.collection.create_index([("stored_at_utc", DESCENDING)])
        self.collection.create_index([("merchant_name", ASCENDING)])
        self.collection.create_index([("fraud_probability", DESCENDING)])

    def insert_fraud_alert(self, transaction_data: Dict[str, Any]) -> str:
        """Insert one fraud alert document and return its MongoDB id."""
        document = clean_for_mongo(normalize_alert_document(dict(transaction_data)))
        document.setdefault("stored_at_utc", datetime.now(timezone.utc))
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def insert_many_fraud_alerts(self, transactions: Iterable[Dict[str, Any]]) -> int:
        """Insert multiple fraud alerts and return the inserted count."""
        now = datetime.now(timezone.utc)
        documents = []
        for transaction in transactions:
            document = clean_for_mongo(normalize_alert_document(dict(transaction)))
            document.setdefault("stored_at_utc", now)
            documents.append(document)
        if not documents:
            return 0
        result = self.collection.insert_many(documents)
        return len(result.inserted_ids)

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent fraud alerts."""
        cursor = self.collection.find({}).sort("stored_at_utc", DESCENDING).limit(limit)
        rows = []
        for doc in cursor:
            normalized = normalize_alert_document(serialize_document(doc))
            rows.append(serialize_document(normalized))
        return rows

    def clear_fraud_alerts(self) -> int:
        """Delete all fraud alerts and return the number removed."""
        result = self.collection.delete_many({})
        return int(result.deleted_count)

    def get_alert_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Find one fraud alert by transaction id."""
        document = self.collection.find_one({"transaction_id": transaction_id})
        if not document:
            return None
        return serialize_document(normalize_alert_document(serialize_document(document)))

    def get_fraud_rate_by_hour(self) -> List[Dict[str, Any]]:
        """Return hourly alert rates based on the fraud_alerts collection."""
        documents = [
            normalize_alert_document(serialize_document(doc))
            for doc in self.collection.find({})
        ]
        if not documents:
            return []

        df = pd.DataFrame(documents)
        total_alerts = len(df)
        df["hour_of_day"] = pd.to_numeric(df.get("hour_of_day", 0), errors="coerce").fillna(0).astype(int).clip(0, 23)
        df["fraud_probability"] = pd.to_numeric(df.get("fraud_probability", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        df["risk_score"] = pd.to_numeric(df.get("risk_score", 0.0), errors="coerce").fillna(0.0).clip(0, 100)
        df["Amount"] = pd.to_numeric(df.get("Amount", 0.0), errors="coerce").fillna(0.0).clip(lower=0)
        df["decision"] = df.get("decision", "REVIEW").astype(str).str.upper()
        grouped = (
            df.assign(
                is_blocked=(df["decision"] == "BLOCKED").astype(int),
                is_flagged=df["decision"].isin(["REVIEW", "FLAGGED"]).astype(int),
            )
            .groupby("hour_of_day")
            .agg(
                fraud_alerts=("fraud_probability", "count"),
                avg_probability=("fraud_probability", "mean"),
                max_probability=("fraud_probability", "max"),
                avg_risk_score=("risk_score", "mean"),
                max_risk_score=("risk_score", "max"),
                total_amount=("Amount", "sum"),
                blocked_count=("is_blocked", "sum"),
                flagged_count=("is_flagged", "sum"),
            )
            .reindex(range(24), fill_value=0)
            .reset_index()
        )
        rows = []
        for _, row in grouped.iterrows():
            fraud_alerts = int(row.get("fraud_alerts", 0))
            rows.append(
                {
                    "hour_of_day": int(row["hour_of_day"]),
                    "fraud_alerts": fraud_alerts,
                    "fraud_rate": fraud_alerts / total_alerts if total_alerts else 0.0,
                    "avg_probability": float(row.get("avg_probability") or 0.0),
                    "max_probability": float(row.get("max_probability") or 0.0),
                    "avg_risk_score": float(row.get("avg_risk_score") or 0.0),
                    "max_risk_score": float(row.get("max_risk_score") or 0.0),
                    "total_amount": float(row.get("total_amount") or 0.0),
                    "blocked_count": int(row.get("blocked_count") or 0),
                    "flagged_count": int(row.get("flagged_count") or 0),
                }
            )
        return rows

    def get_total_fraud_amount_today(self) -> float:
        """Return total alerted transaction amount for the current UTC day."""
        now = datetime.now(timezone.utc)
        start_of_day = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        pipeline = [
            {"$match": {"stored_at_utc": {"$gte": start_of_day}}},
            {"$group": {"_id": None, "total_amount": {"$sum": "$Amount"}}},
        ]
        result = list(self.collection.aggregate(pipeline))
        if not result:
            return 0.0
        return float(result[0].get("total_amount") or 0.0)

    def get_high_risk_merchants(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return merchants ranked by alert count and risk probability."""
        documents = [
            normalize_alert_document(serialize_document(doc))
            for doc in self.collection.find({})
        ]
        if not documents:
            return []

        df = pd.DataFrame(documents)
        df["merchant_name"] = df.get("merchant_name", "UNKNOWN").fillna("UNKNOWN")
        df["fraud_probability"] = pd.to_numeric(df.get("fraud_probability", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
        df["risk_score"] = pd.to_numeric(df.get("risk_score", 0.0), errors="coerce").fillna(0.0).clip(0, 100)
        df["Amount"] = pd.to_numeric(df.get("Amount", 0.0), errors="coerce").fillna(0.0).clip(lower=0)
        df["decision"] = df.get("decision", "REVIEW").astype(str).str.upper()
        if "event_time_utc" in df.columns:
            df["alert_event_time"] = pd.to_datetime(df["event_time_utc"], errors="coerce")
        elif "transaction_time_utc" in df.columns:
            df["alert_event_time"] = pd.to_datetime(df["transaction_time_utc"], errors="coerce")
        elif "stored_at_utc" in df.columns:
            df["alert_event_time"] = pd.to_datetime(df["stored_at_utc"], errors="coerce")
        else:
            df["alert_event_time"] = pd.NaT
        if "stored_at_utc" in df.columns:
            df["stored_at_utc"] = pd.to_datetime(df["stored_at_utc"], errors="coerce")

        grouped = (
            df.assign(
                is_flagged=df["decision"].isin(["REVIEW", "FLAGGED"]).astype(int),
                is_blocked=(df["decision"] == "BLOCKED").astype(int),
            )
            .groupby("merchant_name", dropna=False)
            .agg(
                alert_count=("transaction_id", "count"),
                flagged_count=("is_flagged", "sum"),
                blocked_count=("is_blocked", "sum"),
                avg_probability=("fraud_probability", "mean"),
                max_probability=("fraud_probability", "max"),
                avg_risk_score=("risk_score", "mean"),
                max_risk_score=("risk_score", "max"),
                total_amount=("Amount", "sum"),
                last_alert_time=("alert_event_time", "max"),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(
            ["blocked_count", "avg_risk_score", "total_amount"],
            ascending=[False, False, False],
        ).head(limit)
        grouped["last_alert_time"] = grouped["last_alert_time"].astype(str)
        return grouped.to_dict(orient="records")


def get_client() -> MongoDBClient:
    """Factory used by scripts and dashboard code."""
    return MongoDBClient()


def insert_fraud_alert(transaction_data: Dict[str, Any]) -> str:
    return get_client().insert_fraud_alert(transaction_data)


def save_flagged_transaction(transaction: Dict[str, Any]) -> str:
    return insert_fraud_alert(transaction)


def save_flagged_transactions(transactions: Iterable[Dict[str, Any]]) -> int:
    return get_client().insert_many_fraud_alerts(transactions)


def get_recent_alerts(limit: int = 100) -> List[Dict[str, Any]]:
    return get_client().get_recent_alerts(limit=limit)


def get_recent_flagged_transactions(limit: int = 100) -> List[Dict[str, Any]]:
    return get_recent_alerts(limit=limit)
