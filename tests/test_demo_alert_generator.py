"""Tests for dashboard demo alert generation paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.streaming import demo_alert_generator


class FakeMongoClient:
    def __init__(self) -> None:
        self.alerts: list[dict] = []

    def insert_many_fraud_alerts(self, alerts):
        self.alerts.extend(alerts)
        return len(alerts)


def write_seed(path: Path) -> None:
    records = [
        {
            "transaction_id": "seed-review",
            "merchant_name": "Metro Grocers",
            "amount": 24.58,
            "fraud_probability": 0.25,
            "risk_score": 25.0,
            "decision": "REVIEW",
            "recommended_action": "Route to review.",
            "event_time_utc": "2026-06-01T10:00:00+00:00",
            "stored_at_utc": "2026-06-01T10:01:00+00:00",
            "amount_source": "deployment_seed_alert",
            "top_shap_features": [],
        },
        {
            "transaction_id": "seed-flagged",
            "merchant_name": "Bluebird Electronics",
            "amount": 189.64,
            "fraud_probability": 0.72,
            "risk_score": 72.0,
            "decision": "FLAGGED",
            "recommended_action": "Hold for step-up authentication.",
            "event_time_utc": "2026-06-02T14:00:00+00:00",
            "stored_at_utc": "2026-06-02T14:01:00+00:00",
            "amount_source": "deployment_seed_alert",
            "top_shap_features": [],
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def make_local_test_dir(name: str) -> Path:
    base_dir = Path("tests") / "_tmp_demo_alert_generator" / name
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def test_demo_generation_falls_back_to_seed_when_creditcard_csv_is_missing(monkeypatch):
    test_dir = make_local_test_dir("missing_csv")
    try:
        seed_path = test_dir / "demo_alerts_seed.json"
        write_seed(seed_path)
        fake_client = FakeMongoClient()

        monkeypatch.setattr(demo_alert_generator, "DATA_PATH", test_dir / "missing_creditcard.csv")
        monkeypatch.setattr(demo_alert_generator, "DEMO_SEED_PATH", seed_path)
        monkeypatch.setattr(
            demo_alert_generator,
            "load_model_bundle",
            lambda: pytest.fail("Fallback seed path should not load the trained model."),
        )

        result = demo_alert_generator.generate_demo_alerts(
            alert_threshold=0.1,
            max_alerts=2,
            mongo_client=fake_client,
        )

        assert result["used_fallback_seed"] is True
        assert result["inserted"] == 2
        assert len(fake_client.alerts) == 2
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_no_file_not_found_is_raised_in_seed_fallback_mode(monkeypatch):
    test_dir = make_local_test_dir("no_file_not_found")
    try:
        seed_path = test_dir / "demo_alerts_seed.json"
        write_seed(seed_path)

        monkeypatch.setattr(demo_alert_generator, "DATA_PATH", test_dir / "missing_creditcard.csv")
        monkeypatch.setattr(demo_alert_generator, "DEMO_SEED_PATH", seed_path)

        demo_alert_generator.generate_demo_alerts(
            alert_threshold=0.1,
            max_alerts=1,
            mongo_client=FakeMongoClient(),
        )
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_inserted_fallback_alerts_have_required_dashboard_fields(monkeypatch):
    test_dir = make_local_test_dir("required_fields")
    try:
        seed_path = test_dir / "demo_alerts_seed.json"
        write_seed(seed_path)
        fake_client = FakeMongoClient()

        monkeypatch.setattr(demo_alert_generator, "DATA_PATH", test_dir / "missing_creditcard.csv")
        monkeypatch.setattr(demo_alert_generator, "DEMO_SEED_PATH", seed_path)

        demo_alert_generator.generate_demo_alerts(
            alert_threshold=0.1,
            max_alerts=1,
            mongo_client=fake_client,
        )

        alert = fake_client.alerts[0]
        assert alert["decision"]
        assert isinstance(alert["Amount"], float)
        assert 0.0 <= alert["fraud_probability"] <= 1.0
        assert 0.0 <= alert["risk_score"] <= 100.0
        assert alert["amount_source"] == "deployment_seed_alert"
        assert alert["transaction_id"] != "seed-review"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_existing_dataset_path_keeps_local_model_scoring_behavior(monkeypatch):
    test_dir = make_local_test_dir("local_dataset")
    try:
        csv_path = test_dir / "creditcard.csv"
        csv_path.write_text("Time," + ",".join(f"V{i}" for i in range(1, 29)) + ",Amount,Class\n", encoding="utf-8")

        monkeypatch.setattr(demo_alert_generator, "DATA_PATH", csv_path)

        def raise_local_path_marker():
            raise RuntimeError("local Kaggle scoring path reached")

        monkeypatch.setattr(demo_alert_generator, "load_model_bundle", raise_local_path_marker)

        with pytest.raises(RuntimeError, match="local Kaggle scoring path reached"):
            demo_alert_generator.generate_demo_alerts(mongo_client=FakeMongoClient())
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
