# Real-Time Fraud Detection System

Portfolio-grade real-time fraud detection system using PySpark, SMOTE, XGBoost,
Isolation Forest, MLflow, FastAPI, Kafka, Spark Structured Streaming, MongoDB,
and Streamlit.

## Dataset

Place the Kaggle credit card fraud dataset here:

```text
data/raw/creditcard.csv
```

Expected columns:

```text
Time, V1, V2, ..., V28, Amount, Class
```

`Class = 1` means fraud and `Class = 0` means legitimate.

## End-to-End Commands

Run from the project root.

### Beginner Command Prompt quick start

Use these commands in **Command Prompt** from:

```bat
C:\Users\admin\Desktop\Real-Time Fraud Detection System
```

Install or refresh dependencies:

```bat
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Build features, train the model, and tune the threshold:

```bat
python -m src.processing.feature_engineering
python -m src.modeling.train_batch
python -m src.modeling.threshold_tuner
```

Start the API in one Command Prompt:

```bat
venv\Scripts\activate.bat
cd /d "C:\Users\admin\Desktop\Real-Time Fraud Detection System"
uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload
```

Start the dashboard in another Command Prompt:

```bat
venv\Scripts\activate.bat
cd /d "C:\Users\admin\Desktop\Real-Time Fraud Detection System"
streamlit run app.py --server.port 8502
```

Open:

```text
API docs:  http://127.0.0.1:8001/docs
Dashboard: http://127.0.0.1:8502
```

If the Live Feed is empty, use the dashboard sidebar button **Generate Demo Alerts**.
It scores real rows from `data/raw/creditcard.csv` and stores a realistic mix of
review, flagged, and blocked alerts in MongoDB.

### Streamlit Cloud demo data

The full Kaggle `data/raw/creditcard.csv` file is intentionally not committed to
GitHub. In local development, **Generate Demo Alerts** uses that real CSV and the
trained model exactly as described above. On Streamlit Cloud, when the CSV is not
available, the button inserts packaged deployment seed alerts from:

```text
data/demo/demo_alerts_seed.json
```

Those seed alerts are safe dashboard demo records exported from prior local demo
alerts. They do not contain database credentials, secrets, or the full Kaggle
dataset.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create environment file

```powershell
Copy-Item .env.example .env
```

### 4. Start Kafka and MongoDB

```powershell
docker compose up -d zookeeper kafka mongodb
```

### 5. Build processed parquet features

```powershell
python -m src.processing.feature_engineering
```

This writes:

```text
data/processed/fraud_features.parquet
data/processed/feature_stats.json
```

### 6. Train and track batch models

```powershell
python -m src.modeling.train_batch
```

This trains XGBoost and Isolation Forest in separate MLflow runs, logs metrics
and plots, saves both models under `models/`, and registers the best model as
`FraudDetector` when the local MLflow registry is available.

### 7. Tune the XGBoost threshold

```powershell
python -m src.modeling.threshold_tuner
```

This writes:

```text
models/optimal_threshold.txt
models/threshold_metrics.csv
models/precision_recall_tradeoff.png
```

### 8. Start the FastAPI scoring service

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

Open:

```text
http://127.0.0.1:8001/docs
```

### 9. Start the Spark streaming fraud consumer

Open a second terminal with the virtual environment activated:

```powershell
python -m src.streaming.fraud_consumer --score-url http://127.0.0.1:8001/score
```

Spark needs Java installed. If Spark cannot start, install a JDK and set
`JAVA_HOME`.

### 10. Start the Kafka transaction producer

Open a third terminal with the virtual environment activated:

```powershell
python -m src.streaming.transaction_producer --delay 0.1
```

For a quick demo:

```powershell
python -m src.streaming.transaction_producer --delay 0.02 --max-transactions 5000
```

### 11. Start the Streamlit dashboard

Open a fourth terminal with the virtual environment activated:

```powershell
streamlit run app.py --server.port 8502
```

Open:

```text
http://127.0.0.1:8502
```

## Docker API and Dashboard

After training the models on the host, you can run the API and Streamlit
dashboard in Docker:

```powershell
docker compose up --build -d
```

Services:

```text
FastAPI:   http://127.0.0.1:8001/docs
Streamlit: http://127.0.0.1:8502
Kafka:     localhost:9092
MongoDB:   localhost:27017
```

The Spark consumer and Kafka producer are typically run from the host virtual
environment so Spark can download/use its Kafka connector cleanly.

## Main Files

- `src/processing/feature_engineering.py` creates Spark engineered features and parquet data.
- `src/modeling/train_batch.py` trains XGBoost and Isolation Forest with MLflow tracking.
- `src/modeling/threshold_tuner.py` optimizes the fraud threshold.
- `src/api/main.py` exposes `/health`, `/score`, and `/batch-score`.
- `src/streaming/transaction_producer.py` publishes CSV rows to Kafka.
- `src/streaming/fraud_consumer.py` scores Kafka events and stores alerts.
- `src/storage/mongodb_client.py` wraps MongoDB storage and analytics queries.
- `app.py` is the Streamlit monitoring dashboard.
- `docker-compose.yml` defines Kafka, MongoDB, FastAPI, and Streamlit services.

## Dashboard Notes

This is a local real-time simulation project. The production-style architecture
is Kafka producer -> Spark streaming consumer -> FastAPI scoring -> MongoDB
alerts -> Streamlit dashboard. For an easy Windows demo, the dashboard also has
a **Generate Demo Alerts** button that bypasses Kafka and writes scored example
alerts directly to MongoDB.

The dashboard displays:

- Live Feed: recent alerts, decision counts, average risk, blocked amount, and top risky transactions.
- Analytics: alert count by hour, amount distribution, merchant risk, and volume vs risk.
- Model Performance: XGBoost vs Isolation Forest metrics, threshold tradeoff, confusion matrix, PR curve, ROC curve, and business counts.
- Risk Investigation: transaction lookup, SHAP drivers, recommended action, and full alert JSON.

`V1` through `V28` are anonymized PCA features from the Kaggle dataset. That
means SHAP can explain which mathematical model features moved the score, but it
cannot name direct business reasons like IP address, device, city, or merchant
category.

Fraud is very rare, so accuracy is not the headline metric. AUPRC, recall,
precision, F1, false positives, and false negatives are more useful for judging
this project.
"# Real-Time-Fraud-Detection-System" 
