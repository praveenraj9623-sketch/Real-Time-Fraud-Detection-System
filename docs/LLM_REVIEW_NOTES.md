# LLM Review Notes - Real-Time Fraud Detection System

This package is intentionally trimmed for code review by another LLM.

## What To Review

- `app.py`: Streamlit dashboard with four tabs.
- `src/dashboard/helpers.py`: dashboard data prep, metric merging, confusion values.
- `src/utils/risk.py`: probability/risk clipping, decisions, timestamp parsing.
- `src/utils/demo_selection.py`: lightweight demo alert sampler.
- `src/api/main.py`: FastAPI scoring, SHAP fields, risk decisions.
- `src/storage/mongodb_client.py`: MongoDB inserts, recent alerts, hourly/merchant analytics.
- `src/streaming/demo_alert_generator.py`: local Windows-friendly alert generation.
- `src/modeling/threshold_tuner.py`: extended threshold tuning and business-cost metrics.
- `tests/`: fast tests for logic and regressions.

## Important Context

- Active threshold is `0.999`.
- Old `models/model_metrics.json` XGBoost row is from threshold `0.90`; the dashboard now overrides it with active-threshold metrics.
- `models/threshold_tuning_summary.json` contains:
  - best F1 threshold `0.999`
  - best business-cost threshold `0.996`
- Static confusion matrix images can reflect old training thresholds. `app.py` now displays a dynamic confusion matrix from active counts.
- Demo alerts have both:
  - `stored_at_utc`: actual Mongo insert time
  - `event_time_utc` / `transaction_time_utc`: simulated transaction time

## Excluded From This Review Zip

- `venv/`, `.venv/`
- `.env`
- `data/raw/creditcard.csv`
- `data/processed/fraud_features.parquet`
- `models/*.joblib`
- `models/validation_data.csv`
- `mlruns/`
- `dist/`
- `__pycache__/`

## Current Verification

```bat
venv\Scripts\python.exe -m compileall src app.py tests
venv\Scripts\python.exe -m pytest -q
```

Current result:

```text
15 passed
```

## Suggested LLM Review Questions

1. Is `app.py` too large, and should dashboard tabs be split into modules?
2. Is the active threshold logic robust if `threshold_tuning_summary.json` is missing?
3. Should the project prefer best F1 threshold or best business-cost threshold by default?
4. Is the demo alert sampler balancing amount, risk, and decision categories well enough?
5. Are Docker service dependencies robust enough for startup timing?
6. Should model binaries be excluded from GitHub or handled with Git LFS?
7. Are there missing tests around FastAPI `/score` and MongoDB aggregation?
