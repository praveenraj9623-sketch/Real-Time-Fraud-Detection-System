"""Basic smoke tests for the fraud detection project scaffold."""

from pathlib import Path

from src.processing.feature_engineering import (
    BASE_FEATURE_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    get_spark_schema,
)


def test_feature_column_count():
    assert len(BASE_FEATURE_COLUMNS) == 30
    assert len(ENGINEERED_FEATURE_COLUMNS) == 4
    assert len(FEATURE_COLUMNS) == 34
    assert FEATURE_COLUMNS[0] == "Time"
    assert FEATURE_COLUMNS[-1] == "amount_zscore"


def test_spark_schema_contains_target():
    schema = get_spark_schema(include_target=True)
    assert len(schema.fields) == 31
    assert schema.fields[-1].name == "Class"


def test_expected_project_files_exist():
    expected_files = [
        "requirements.txt",
        "docker-compose.yml",
        "src/modeling/train_batch.py",
        "src/streaming/transaction_producer.py",
        "src/streaming/fraud_consumer.py",
        "src/api/main.py",
        "app.py",
    ]
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Missing {file_path}"
