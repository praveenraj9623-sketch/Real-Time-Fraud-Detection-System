"""Streamlit dashboard for real-time fraud monitoring — enhanced UI."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dashboard.helpers import (
    ACTIVE_THRESHOLD_EXPLANATION,
    ANALYTICS_SCOPE_CAPTION,
    build_active_xgb_metrics,
    cumulative_alerted_amount_by_event_time,
    confusion_matrix_values,
    event_time_series,
    format_currency,
    format_percent,
    format_probability_display,
    format_risk,
    has_confusion_count_fields,
    merchant_risk_summary,
    metrics_from_confusion_counts,
    night_time_mask,
    normalize_artifact_path,
    prepare_alert_dataframe,
    short_transaction_id,
    summarize_alerts,
)
from src.processing.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from src.storage.mongodb_client import MongoDBClient
from src.streaming.demo_alert_generator import generate_demo_alerts
from src.utils.risk import DEFAULT_SAVE_THRESHOLD, clip_probability

load_dotenv()

MODEL_METRICS_PATH = Path("models/model_metrics.json")
THRESHOLD_METRICS_PATH = Path("models/threshold_metrics.csv")
THRESHOLD_SUMMARY_PATH = Path("models/threshold_tuning_summary.json")
TRAINING_STATS_PATH = Path("models/training_stats.json")
VALIDATION_DATA_PATH = Path("models/validation_data.csv")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fraud_classifier.joblib"))
DEFAULT_XGB_CONFUSION = Path("models/artifacts/xgboost_confusion_matrix.png")
API_HEALTH_URL = os.getenv("FASTAPI_HEALTH_URL", "http://127.0.0.1:8001/health")

DECISION_COLORS = {
    "APPROVED": "#16a34a",
    "REVIEW":   "#2563eb",
    "FLAGGED":  "#d97706",
    "BLOCKED":  "#dc2626",
}
DEBUG_PLOTLY = os.getenv("STREAMLIT_DEBUG_MODE", "0").strip().lower() in {"1", "true", "yes"}
PLOTLY_CONFIG = {"displayModeBar": DEBUG_PLOTLY}
ANALYTICS_ALERT_LIMIT = int(os.getenv("DASHBOARD_ANALYTICS_ALERT_LIMIT", "100000"))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main .block-container { padding-top: 1.2rem; max-width: 1560px; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #1a2e4a 0%, #0d1b2e 100%);
        border-radius: 12px;
        padding: 22px 26px 18px;
        margin-bottom: 18px;
        box-shadow: 0 6px 30px rgba(0,0,0,0.22);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(37,99,235,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        margin: 0 0 5px;
        font-size: 24px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }
    .hero p { margin: 0; color: #94a3b8; font-size: 13px; }
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
        margin: 8px 6px 0 0;
        font-size: 11px;
        background: rgba(255,255,255,0.07);
        color: #cbd5e1;
        font-family: 'IBM Plex Mono', monospace;
    }
    .status-pill.ok  { border-color: rgba(34,197,94,0.4); color: #86efac; }
    .status-pill.err { border-color: rgba(239,68,68,0.4);  color: #fca5a5; }

    /* ── Insight cards ── */
    .insight-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 14px 0 4px;
    }
    .insight-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .insight-card.danger  { border-left-color: #dc2626; }
    .insight-card.warning { border-left-color: #d97706; }
    .insight-card.success { border-left-color: #16a34a; }
    .insight-card.purple  { border-left-color: #7c3aed; }
    .insight-val {
        font-size: 24px;
        font-weight: 700;
        margin: 0 0 3px;
        color: #0f172a;
        font-family: 'IBM Plex Mono', monospace;
    }
    .insight-label { font-size: 12px; color: #64748b; margin: 0; }

    /* ── Section label ── */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 18px 0 10px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 6px;
    }

    /* ── Headline metric cards ── */
    .headline-metric {
        background: linear-gradient(135deg, #f8faff 0%, #eff6ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
    }
    .headline-val {
        font-size: 32px;
        font-weight: 700;
        color: #1e40af;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0;
        line-height: 1;
    }
    .headline-lbl { font-size: 12px; color: #64748b; margin: 6px 0 0; font-weight: 500; }
    .headline-sub { font-size: 11px; color: #94a3b8; margin: 2px 0 0; }

    /* ── Empty state ── */
    .empty-box {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 36px 28px;
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        color: #475569;
        text-align: center;
    }
    .empty-box h3 { color: #1e3a5f; margin: 0 0 8px; font-size: 17px; }
    .empty-box p  { margin: 0; font-size: 14px; }

    /* ── Decision pill in table ── */
    .decision-pill {
        display: inline-block;
        min-width: 80px;
        text-align: center;
        padding: 3px 9px;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.04em;
    }

    /* ── Download button ── */
    div[data-testid="stDownloadButton"] button {
        background: #1e3a5f !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 5px 14px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: #2563eb !important;
    }

    /* ── Metrics ── */
    div[data-testid="stMetricValue"] { font-size: 22px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_mongo_client() -> MongoDBClient:
    mongo_uri = get_config_value(
        secret_keys=("MONGODB_URI", "MONGO_URI"),
        env_keys=("MONGODB_URI", "MONGO_URI"),
        fallback="mongodb://localhost:27017",
    )
    mongo_db = get_config_value(
        secret_keys=("MONGODB_DB", "MONGO_DATABASE"),
        env_keys=("MONGODB_DB", "MONGO_DATABASE"),
        fallback="fraud_detection",
    )
    mongo_collection = get_config_value(
        secret_keys=("MONGODB_COLLECTION", "MONGO_COLLECTION"),
        env_keys=("MONGODB_COLLECTION", "MONGO_COLLECTION"),
        fallback="fraud_alerts",
    )
    return MongoDBClient(
        uri=mongo_uri,
        database_name=mongo_db,
        collection_name=mongo_collection,
    )


def get_streamlit_secret(*keys: str) -> str | None:
    """Return the first configured Streamlit secret from a list of names."""
    try:
        secrets = st.secrets
        for key in keys:
            value = secrets.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        return None
    return None


def get_config_value(
    *,
    secret_keys: tuple[str, ...],
    env_keys: tuple[str, ...],
    fallback: str,
) -> str:
    """Read Streamlit secrets first, then environment variables, then fallback."""
    secret_value = get_streamlit_secret(*secret_keys)
    if secret_value:
        return secret_value
    for key in env_keys:
        env_value = os.getenv(key)
        if env_value is not None and env_value.strip():
            return env_value.strip()
    return fallback


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False, ttl=15)
def get_api_health() -> Dict[str, Any]:
    try:
        response = requests.get(API_HEALTH_URL, timeout=1.5)
        return response.json() if response.ok else {"status": f"http_{response.status_code}"}
    except Exception:
        return {"status": "offline"}


@st.cache_data(show_spinner=False)
def compute_xgb_validation_metrics(threshold: float) -> Dict[str, Any]:
    if not MODEL_PATH.exists() or not VALIDATION_DATA_PATH.exists():
        return {}
    try:
        bundle = joblib.load(MODEL_PATH)
        predictor = bundle.get("model") or bundle.get("pipeline")
        feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        validation = pd.read_csv(VALIDATION_DATA_PATH)
        X = validation[feature_columns].astype(float)
        y_true = validation[TARGET_COLUMN].astype(int)
        scores = predictor.predict_proba(X)[:, 1]
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics = metrics_from_confusion_counts(int(tn), int(fp), int(fn), int(tp))
        metrics.update({
            "threshold":   float(threshold),
            "auc":         float(roc_auc_score(y_true, scores)),
            "auprc":       float(average_precision_score(y_true, scores)),
            "metric_source": "active_threshold_recomputed",
            "predicted_fraud_count": int(y_pred.sum()),
            "fraud_cases":           int((y_true == 1).sum()),
            "validation_rows":       int(len(validation)),
        })
        return metrics
    except Exception:
        return {}


def render_dynamic_confusion_matrix(metrics: Dict[str, Any], threshold: float) -> go.Figure:
    """Active-threshold confusion matrix in standard fraud layout.

    Rows are actual labels and columns are predicted labels:
    [[TN, FP], [FN, TP]].
    """
    tn = int(metrics.get("true_negatives", 0) or 0)
    fp = int(metrics.get("false_positives", 0) or 0)
    fn = int(metrics.get("false_negatives", 0) or 0)
    tp = int(metrics.get("true_positives", 0) or 0)
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Active Confusion Matrix — threshold {threshold:.3f}",
        height=380,
        margin=dict(l=25, r=25, t=55, b=35),
        xaxis_title="Prediction",
        yaxis_title="Actual label",
    )
    return fig


def metrics_from_threshold_summary(summary: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    for key in ("best_f1_threshold", "best_business_cost_threshold"):
        row = summary.get(key, {})
        if row and abs(float(row.get("threshold", -1.0)) - float(threshold)) <= 1e-9:
            fallback = dict(row)
            fallback["metric_source"] = f"{key}_summary"
            return fallback
    return {}


def render_risk_gauge(value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(min(value, 100.0), 0.0),
        number={"suffix": " / 100", "font": {"size": 22}},
        title={"text": "Avg Risk Score", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": "#2563eb", "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#dcfce7"},
                {"range": [40, 75], "color": "#fef3c7"},
                {"range": [75, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line":  {"color": "#dc2626", "width": 3},
                "thickness": 0.7,
                "value": 75,
            },
        },
    ))
    fig.update_layout(height=290, margin=dict(l=15, r=15, t=45, b=15))
    return fig


def artifact_path(metrics: Dict[str, Any], model: str, artifact: str, fallback: Path) -> Path:
    value = metrics.get("artifacts", {}).get(model, {}).get(artifact)
    return normalize_artifact_path(value, fallback)


def styled_decision_table(table: pd.DataFrame):
    def color_decision(value: Any) -> str:
        color = DECISION_COLORS.get(str(value).upper(), "#475569")
        return f"background-color: {color}; color: white; font-weight: 700;"
    if "decision" not in table.columns:
        return table.style
    return table.style.map(color_decision, subset=["decision"])


def format_timestamp_display(value: Any) -> str:
    """Format timestamps consistently for tables, captions, JSON, and CSV exports."""
    if value is None or value == "":
        return "N/A"
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return str(value)
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(value)


def sanitize_for_display(value: Any) -> Any:
    """Recursively convert timestamps and non-JSON objects to readable values."""
    if isinstance(value, dict):
        return {k: sanitize_for_display(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_display(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_display(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return format_timestamp_display(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return value


def format_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["event_time_utc", "transaction_time_utc", "stored_at_utc", "last_alert_time"]:
        if col in out.columns:
            out[col] = out[col].map(format_timestamp_display)
    return out


# ── New chart helpers ─────────────────────────────────────────────────────────

def render_decision_donut(summary: Dict[str, Any]) -> go.Figure:
    """Donut: REVIEW / FLAGGED / BLOCKED distribution."""
    labels = ["REVIEW", "FLAGGED", "BLOCKED"]
    values = [
        max(int(summary.get("review_count",   0)), 0),
        max(int(summary.get("flagged_count",  0)), 0),
        max(int(summary.get("blocked_count",  0)), 0),
    ]
    colors = [DECISION_COLORS["REVIEW"], DECISION_COLORS["FLAGGED"], DECISION_COLORS["BLOCKED"]]
    total  = sum(values)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent" if total > 0 else "none",
        direction="clockwise",
        hovertemplate="%{label}: <b>%{value}</b> alerts (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text="Decision Split", font=dict(size=13)),
        height=290,
        margin=dict(l=10, r=10, t=42, b=10),
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:11px'>alerts</span>",
            x=0.5, y=0.5,
            font=dict(size=15, color="#0f172a"),
            showarrow=False,
        )] if total > 0 else [],
    )
    return fig


def render_risk_histogram(alerts_df: pd.DataFrame) -> go.Figure:
    """Histogram of risk scores, coloured by decision."""
    fig = px.histogram(
        alerts_df,
        x="risk_score",
        color="decision",
        nbins=25,
        title="Risk Score Distribution",
        labels={"risk_score": "Risk Score (0 – 100)", "count": "Alerts"},
        color_discrete_map=DECISION_COLORS,
        opacity=0.82,
        barmode="overlay",
    )
    fig.update_xaxes(range=[0, 100])
    fig.update_layout(
        height=265,
        bargap=0.03,
        margin=dict(l=25, r=15, t=45, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def render_cumulative_timeline(alerts_df: pd.DataFrame) -> go.Figure:
    """Area chart — cumulative alerted amount over time."""
    df = cumulative_alerted_amount_by_event_time(alerts_df)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Cumulative Alerted Amount by Event Time",
            height=270,
            annotations=[dict(text="No timestamp data available", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#94a3b8"))],
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["_event_time"], y=df["cum_amount"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.10)",
        line=dict(color="#dc2626", width=2.5),
        name="Cumulative $",
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    blocked = df[df["decision"].str.upper() == "BLOCKED"]
    if not blocked.empty:
        fig.add_trace(go.Scatter(
            x=blocked["_event_time"], y=blocked["cum_amount"],
            mode="markers",
            marker=dict(color="#dc2626", size=8, symbol="circle-open",
                        line=dict(width=2, color="#dc2626")),
            name="Blocked event",
            hovertemplate="Blocked: $%{y:,.2f}<extra></extra>",
        ))
    fig.update_layout(
        title="Cumulative Alerted Amount by Event Time",
        xaxis_title="Simulated Transaction Event Time",
        yaxis_title="Cumulative Alerted Amount",
        yaxis_tickprefix="$",
        height=270,
        margin=dict(l=25, r=15, t=45, b=35),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def render_prob_vs_amount_scatter(alerts_df: pd.DataFrame) -> go.Figure:
    """Scatter: fraud probability vs transaction amount."""
    fig = px.scatter(
        alerts_df,
        x="Amount",
        y="fraud_probability",
        color="decision",
        title="Fraud Probability by Transaction Amount",
        labels={"fraud_probability": "Fraud Probability", "Amount": "Transaction Amount ($)"},
        color_discrete_map=DECISION_COLORS,
        hover_data=["short_transaction_id", "merchant_name", "risk_display"],
        opacity=0.70,
        size_max=10,
    )
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(tickprefix="$")
    fig.update_layout(
        height=270,
        margin=dict(l=25, r=15, t=45, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


# ── Load data ─────────────────────────────────────────────────────────────────

model_metrics   = load_json(MODEL_METRICS_PATH)
training_stats  = load_json(TRAINING_STATS_PATH)
threshold_summary = load_json(THRESHOLD_SUMMARY_PATH)

dataset_stats    = training_stats.get("dataset", {})
optimal_threshold = clip_probability(training_stats.get("optimal_threshold", 0.5))
save_threshold   = float(os.getenv("ALERT_SAVE_THRESHOLD", str(DEFAULT_SAVE_THRESHOLD)))
api_health       = get_api_health()
last_refresh     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    mongo_client = get_mongo_client()
except Exception:
    st.error("MongoDB is not reachable. Configure MongoDB URI in Streamlit secrets or start MongoDB locally.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Controls")
    recent_limit   = st.slider("Recent alerts to show", 20, 500, 100, 20)
    auto_refresh   = st.toggle("Auto refresh", value=False)
    refresh_seconds = st.slider("Refresh every (s)", 5, 60, 10, 5)
    live_chart_view = st.radio(
        "Live chart view",
        ["Highest Risk", "Recent Mixed Alerts", "Group by Decision"],
        index=0,
    )

    st.divider()
    st.subheader("Demo Data")
    st.caption("Scores real creditcard.csv rows with the trained model, then saves a varied alert sample.")
    demo_rows       = st.number_input("Rows to scan", 1000, 100000, 30000, 1000)
    demo_threshold  = st.slider("Store demo alert if probability ≥", 0.01, 0.99, 0.10, 0.01)
    demo_max_alerts = st.slider("Max demo alerts", 10, 200, 60, 10)

    if st.button("Generate Demo Alerts", type="primary", use_container_width=True):
        with st.spinner("Scoring transactions and saving realistic demo alerts…"):
            result = generate_demo_alerts(
                max_rows=int(demo_rows),
                alert_threshold=float(demo_threshold),
                max_alerts=int(demo_max_alerts),
                include_shap=True,
                mongo_client=mongo_client,
            )
        mix = result.get("decision_mix", {})
        st.success(
            f"Inserted {result['inserted']} alerts from {result['scanned']} scanned rows. "
            f"Review {mix.get('review', 0)}, Flagged {mix.get('flagged', 0)}, Blocked {mix.get('blocked', 0)}."
        )
        st.rerun()

    st.divider()
    confirm_clear = st.checkbox("Confirm clear alerts")
    if st.button("Clear Alerts", use_container_width=True, disabled=not confirm_clear):
        deleted = mongo_client.clear_fraud_alerts()
        st.warning(f"Deleted {deleted} alerts.")
        st.rerun()

# ── Fetch and prepare data ────────────────────────────────────────────────────

recent_records = mongo_client.get_recent_alerts(limit=recent_limit)
alerts_df = prepare_alert_dataframe(
    recent_records,
    model_threshold=optimal_threshold,
    save_threshold=save_threshold,
)
analytics_records = mongo_client.get_recent_alerts(limit=ANALYTICS_ALERT_LIMIT)
analytics_df = prepare_alert_dataframe(
    analytics_records,
    model_threshold=optimal_threshold,
    save_threshold=save_threshold,
)
summary = summarize_alerts(alerts_df)

api_status   = api_health.get("status", "unknown")
fraud_rate   = float(dataset_stats.get("fraud_rate", 0.0))
training_rows = int(dataset_stats.get("row_count", 0) or 0)
api_pill_class = "ok" if api_status not in ("offline", "unknown") else "err"

# ── Hero header ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero">
      <h1>⚡ Real-Time Fraud Detection Monitor</h1>
      <p>Scores credit-card transactions in a near-real-time fraud workflow, stores suspicious activity, and surfaces investigation-ready analytics.</p>
      <div style="margin-top:4px;">
        <span class="status-pill ok">MongoDB connected</span>
        <span class="status-pill {api_pill_class}">API: {api_status}</span>
        <span class="status-pill">block threshold: {optimal_threshold:.3f}</span>
        <span class="status-pill">alert save ≥ {save_threshold:.2f}</span>
        <span class="status-pill">{training_rows:,} training rows</span>
        <span class="status-pill">dataset fraud rate: {fraud_rate:.4%}</span>
        <span class="status-pill">refreshed {last_refresh}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

live_tab, analytics_tab, performance_tab, investigation_tab = st.tabs(
    ["⚡ Live Feed", "📊 Analytics", "🎯 Model Performance", "🔍 Risk Investigation"]
)

# ════════════════════════════════════════════════════════════════════════════════
# LIVE FEED
# ════════════════════════════════════════════════════════════════════════════════
with live_tab:

    # ── 6 KPI metrics ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Alerts",    f"{summary['alert_count']:,}")
    c2.metric("Review",          f"{summary['review_count']:,}")
    c3.metric("Flagged",         f"{summary['flagged_count']:,}")
    c4.metric("Blocked",         f"{summary['blocked_count']:,}")
    c5.metric("Alerted Amount",  format_currency(summary["alerted_amount"]))
    c6.metric("Avg Risk",        format_risk(summary["average_risk"]))

    # ── Insight cards (shown even when alerts exist) ──
    if not alerts_df.empty:
        total_alerts = max(summary["alert_count"], 1)
        blocked_rate = summary["blocked_count"] / total_alerts * 100
        night_mask   = night_time_mask(alerts_df)
        night_count  = int(night_mask.sum())
        night_pct    = night_count / total_alerts * 100
        max_risk     = float(alerts_df["risk_score"].max())

        blocked_cls = "danger" if blocked_rate > 30 else "warning" if blocked_rate > 10 else "success"
        night_cls   = "warning" if night_pct   > 30 else ""
        risk_cls    = "danger"  if max_risk    > 90 else "warning" if max_risk > 70 else "purple"

        st.markdown(
            f"""
            <div class="insight-row">
              <div class="insight-card {blocked_cls}">
                <p class="insight-val">{blocked_rate:.1f}%</p>
                <p class="insight-label">Blocked share of stored alerts &nbsp;·&nbsp;
                  {summary['blocked_count']} blocked of {total_alerts} alerts</p>
              </div>
              <div class="insight-card {risk_cls}">
                <p class="insight-val">{max_risk:.0f}</p>
                <p class="insight-label">Highest risk score in current alert batch</p>
              </div>
              <div class="insight-card {night_cls}">
                <p class="insight-val">{night_pct:.1f}%</p>
                <p class="insight-label">Night-time alert share · window: 10 PM to 6 AM.</p>
                <p class="insight-label">Night-Time Alerts: {night_count} of {total_alerts}</p>
                <p class="insight-label">Based on simulated transaction event time.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Empty state ──
    if alerts_df.empty:
        st.markdown(
            """
            <div class="empty-box">
              <h3>No alerts stored yet</h3>
              <p>Use <b>Generate Demo Alerts</b> in the sidebar.<br>
              It reads real rows from <b>data/raw/creditcard.csv</b>, scores them with the
              trained XGBoost model, and saves REVIEW / FLAGGED / BLOCKED records into MongoDB.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Gauge + Donut + Live bar chart ──
        gauge_col, donut_col, chart_col = st.columns([1, 1, 2])

        with gauge_col:
            st.plotly_chart(render_risk_gauge(float(summary["average_risk"])),
                            use_container_width=True, config=PLOTLY_CONFIG)
        with donut_col:
            st.plotly_chart(render_decision_donut(summary),
                            use_container_width=True, config=PLOTLY_CONFIG)
        with chart_col:
            if live_chart_view == "Recent Mixed Alerts":
                event_sort = event_time_series(alerts_df, include_stored_at=True)
                recent_mixed = (
                    alerts_df.assign(_event_time_sort=event_sort)
                    .sort_values("_event_time_sort", ascending=False)
                    .head(24)
                )
                fig = px.bar(
                    recent_mixed, x="risk_score", y="short_transaction_id",
                    color="decision", orientation="h",
                    title="Recent Mixed Alerts by Risk Score",
                    labels={"risk_score": "Risk Score", "short_transaction_id": "Transaction"},
                    hover_data=["transaction_id", "merchant_name", "amount_display", "probability_display"],
                    color_discrete_map=DECISION_COLORS,
                )
                fig.update_xaxes(range=[0, 100])
                fig.update_layout(height=310,
                                  yaxis={"categoryorder": "array",
                                         "categoryarray": recent_mixed["short_transaction_id"].tolist()[::-1]})
            elif live_chart_view == "Group by Decision":
                decision_summary = (
                    alerts_df.groupby("decision")
                    .agg(alert_count=("transaction_id", "count"),
                         avg_risk_score=("risk_score", "mean"),
                         total_amount=("Amount", "sum"))
                    .reset_index()
                )
                fig = px.bar(
                    decision_summary, x="decision", y="alert_count",
                    color="decision", title="Alerts Grouped by Decision",
                    hover_data=["avg_risk_score", "total_amount"],
                    color_discrete_map=DECISION_COLORS,
                )
                fig.update_layout(height=310, showlegend=False)
            else:
                top_alerts = alerts_df.sort_values(
                    ["risk_score", "Amount"], ascending=[False, False]
                ).head(20)
                fig = px.bar(
                    top_alerts, x="risk_score", y="short_transaction_id",
                    color="decision", orientation="h",
                    title="Highest-Risk Recent Transactions",
                    labels={"risk_score": "Risk Score", "short_transaction_id": "Transaction"},
                    hover_data={"transaction_id": True, "merchant_name": True,
                                "amount_display": True, "probability_display": True,
                                "risk_display": True, "risk_score": False},
                    color_discrete_map=DECISION_COLORS,
                )
                fig.update_xaxes(range=[0, 100])
                fig.update_layout(height=310,
                                  yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # ── Risk score distribution histogram ──
        st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(render_risk_histogram(alerts_df), use_container_width=True, config=PLOTLY_CONFIG)

        # ── Alert feed table ──
        st.markdown('<div class="section-label">Recent Alert Feed</div>', unsafe_allow_html=True)

        display_columns = [c for c in [
            "short_transaction_id", "merchant_name", "amount_display",
            "probability_display", "risk_display", "decision",
            "action_short", "event_time_utc", "stored_at_utc",
        ] if c in alerts_df.columns]

        export_cols = [c for c in [
            "transaction_id", "merchant_name", "Amount", "fraud_probability",
            "risk_score", "decision", "recommended_action",
            "hour_of_day", "stored_at_utc",
        ] if c in alerts_df.columns]
        export_df = format_time_columns(alerts_df[export_cols])
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        tbl_left, tbl_right = st.columns([6, 1])
        with tbl_right:
            st.download_button(
                label="⬇ Export CSV",
                data=csv_bytes,
                file_name=f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

        display_df = format_time_columns(
            alerts_df[display_columns].rename(columns={
                "short_transaction_id": "transaction_id",
                "amount_display":       "amount",
                "probability_display":  "fraud_probability",
                "risk_display":         "risk_score",
                "action_short":         "recommended_action",
            })
        )
        st.dataframe(
            styled_decision_table(display_df),
            use_container_width=True,
            hide_index=True,
        )

# ════════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with analytics_tab:

    hourly          = pd.DataFrame(mongo_client.get_fraud_rate_by_hour())
    merchant_summary = merchant_risk_summary(analytics_df)
    st.caption(ANALYTICS_SCOPE_CAPTION)

    if analytics_df.empty:
        st.info("No alert analytics yet. Generate demo alerts from the sidebar.")
    else:
        # ── Row 1: hourly bar + amount histogram ──
        left, right = st.columns(2)
        with left:
            if not hourly.empty:
                fig = px.bar(
                    hourly, x="hour_of_day", y="fraud_alerts",
                    color="avg_risk_score",
                    hover_data=["fraud_rate", "total_amount", "blocked_count", "flagged_count"],
                    title="Alert Count by Hour of Day",
                    labels={"hour_of_day": "Hour", "fraud_alerts": "Alerts",
                            "avg_risk_score": "Avg Risk"},
                    color_continuous_scale="YlOrRd",
                )
                fig.update_xaxes(tickmode="array", tickvals=list(range(24)))
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                active_hours = int((hourly["fraud_alerts"] > 0).sum())
                if active_hours <= 1:
                    st.info("All saved alerts are in one hour — generate a fresh demo batch to spread timestamps.")
        with right:
            use_log = st.checkbox("Log scale for amount axis", value=False)
            amount_df = analytics_df.copy()
            amount_df["Amount"] = amount_df["Amount"].clip(lower=0.01)
            fig = px.histogram(
                amount_df, x="Amount", color="decision", nbins=35,
                title="Alerted Amount Distribution",
                labels={"Amount": "Amount"},
                color_discrete_map=DECISION_COLORS,
            )
            fig.update_xaxes(tickprefix="$")
            if use_log:
                fig.update_xaxes(type="log", tickprefix="$")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption(
                "Amounts come from the original anonymized credit-card dataset; "
                "some fraud examples have very small transaction amounts."
            )

        # ── Row 2: cumulative timeline + prob vs amount ──
        st.markdown('<div class="section-label">Transaction Timeline & Amount Analysis</div>',
                    unsafe_allow_html=True)
        tl_left, tl_right = st.columns(2)
        with tl_left:
            st.plotly_chart(render_cumulative_timeline(analytics_df), use_container_width=True, config=PLOTLY_CONFIG)
        with tl_right:
            st.plotly_chart(render_prob_vs_amount_scatter(analytics_df), use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(
            "Amounts come from the original anonymized credit-card dataset; "
            "some fraud examples have very small transaction amounts."
        )

        # ── Row 3: merchant scatter ──
        if not merchant_summary.empty:
            st.markdown('<div class="section-label">Merchant Risk Intelligence</div>',
                        unsafe_allow_html=True)
            scatter_df = merchant_summary.copy()
            scatter_df["bubble_amount"] = scatter_df["total_amount"].clip(lower=1.0)
            fig = px.scatter(
                scatter_df,
                x="alert_count", y="avg_risk_score",
                size="bubble_amount", color="blocked_count",
                hover_name="merchant_name",
                hover_data=["alert_count", "avg_risk_score", "total_amount",
                            "review_count", "flagged_count", "blocked_count", "block_rate",
                            "median_risk_score"],
                title="Merchant Alert Volume vs Average Risk",
                labels={
                    "alert_count": "Alert Count",
                    "avg_risk_score": "Average Risk Score",
                    "block_rate": "Block Rate",
                },
                color_continuous_scale="Reds",
            )
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            st.subheader("High-Risk Merchants")
            merchant_table = pd.DataFrame(mongo_client.get_high_risk_merchants(limit=15))
            if not merchant_table.empty:
                display_merchants = merchant_table.copy()
                for column in ["avg_risk_score", "median_risk_score"]:
                    if column in display_merchants.columns:
                        display_merchants[column] = display_merchants[column].map(
                            lambda v: f"{float(v):.1f} / 100")
                if "block_rate" in display_merchants.columns:
                    display_merchants["block_rate"] = display_merchants["block_rate"].map(format_percent)
                if "avg_probability" in display_merchants.columns:
                    display_merchants["avg_probability"] = display_merchants["avg_probability"].map(format_percent)
                if "total_amount" in display_merchants.columns:
                    display_merchants["total_amount"] = display_merchants["total_amount"].map(format_currency)
                display_merchants = format_time_columns(display_merchants)
                preferred_cols = [c for c in [
                    "merchant_name", "alert_count", "review_count", "flagged_count", "blocked_count",
                    "block_rate", "avg_probability", "avg_risk_score", "median_risk_score",
                    "total_amount", "last_alert_time",
                ] if c in display_merchants.columns]
                st.dataframe(display_merchants[preferred_cols], use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with performance_tab:

    stored_xgb_metrics = dict(model_metrics.get("xgboost", {}))
    iforest_metrics    = dict(model_metrics.get("isolation_forest", {}))
    computed_counts    = (
        compute_xgb_validation_metrics(optimal_threshold)
        or metrics_from_threshold_summary(threshold_summary, optimal_threshold)
    )
    xgb_metrics = build_active_xgb_metrics(stored_xgb_metrics, computed_counts, optimal_threshold)

    # ── Headline metrics ──
    best_f1_row = threshold_summary.get("best_f1_threshold", {})
    _auc    = float(stored_xgb_metrics.get("auc",   0))
    _auprc  = float(stored_xgb_metrics.get("auprc", 0))
    _f1     = float(best_f1_row.get("f1",     0))
    _recall = float(best_f1_row.get("recall", 0))
    _thr    = float(best_f1_row.get("threshold", optimal_threshold))

    st.markdown('<div class="section-label">Champion Model — XGBoost Headlines</div>',
                unsafe_allow_html=True)
    hm1, hm2, hm3, hm4 = st.columns(4)
    for col, val, label, sub in [
        (hm1, f"{_auc:.4f}",    "AUC (ROC)",     "Higher = better ranking across thresholds"),
        (hm2, f"{_auprc:.4f}",  "AUPRC",         "Key metric for imbalanced fraud data"),
        (hm3, f"{_f1:.4f}",     f"F1 @ {_thr:.3f}", "Best F1 threshold chosen by tuner"),
        (hm4, f"{_recall:.4f}", "Recall",        "Fraction of fraud cases caught"),
    ]:
        col.markdown(
            f"""<div class="headline-metric">
                  <p class="headline-val">{val}</p>
                  <p class="headline-lbl">{label}</p>
                  <p class="headline-sub">{sub}</p>
                </div>""",
            unsafe_allow_html=True,
        )

    st.caption(ACTIVE_THRESHOLD_EXPLANATION)
    st.divider()

    # ── Comparison table ──
    comparison_rows = [
        {"model": "XGBoost", **xgb_metrics},
        {"model": "Isolation Forest", **iforest_metrics},
    ]
    comparison = pd.DataFrame(comparison_rows)
    st.subheader("Model Comparison")
    if not comparison.empty:
        preferred = [
            "model", "precision", "recall", "f1", "auc", "auprc", "threshold",
            "true_negatives", "false_positives", "false_negatives", "true_positives",
            "predicted_fraud_count", "metric_source",
        ]
        visible = [c for c in preferred if c in comparison.columns]
        st.dataframe(comparison[visible], use_container_width=True, hide_index=True)

    if computed_counts and has_confusion_count_fields(computed_counts):
        caught  = computed_counts["true_positives"]
        total_f = computed_counts.get("fraud_cases", caught + computed_counts["false_negatives"])
        false_a = computed_counts["false_positives"]
        missed  = computed_counts["false_negatives"]
        st.info(
            f"At threshold **{optimal_threshold:.3f}**, XGBoost catches **{caught}** of "
            f"**{total_f}** fraud cases, flags **{false_a}** legitimate transactions incorrectly, "
            f"and misses **{missed}** fraud cases."
        )
    elif computed_counts:
        st.info(
            f"At threshold **{optimal_threshold:.3f}**, active threshold metrics were loaded from "
            "the tuning summary. Validation confusion-matrix count fields are not available in the "
            "deployed artifacts, so count-based wording is hidden."
        )

    threshold_metrics = pd.DataFrame()
    if THRESHOLD_METRICS_PATH.exists():
        threshold_metrics = pd.read_csv(THRESHOLD_METRICS_PATH)

    if not threshold_metrics.empty:
        st.subheader("Threshold Tradeoffs")
        fig = px.line(
            threshold_metrics, x="threshold",
            y=["precision", "recall", "f1"],
            markers=False,
            title="Precision, Recall, F1 by Threshold",
        )
        best_f1_thr   = threshold_summary.get("best_f1_threshold", {}).get("threshold", optimal_threshold)
        best_cost_thr = threshold_summary.get("best_business_cost_threshold", {}).get("threshold")
        fig.add_vline(
            x=float(optimal_threshold), line_dash="solid", line_color="#dc2626",
            annotation_text=f"Active {float(optimal_threshold):.3f}",
            annotation_position="top left",
        )
        fig.add_vline(
            x=float(best_f1_thr), line_dash="dash", line_color="black",
            annotation_text=f"Best F1 {float(best_f1_thr):.3f}",
            annotation_position="top left",
        )
        if best_cost_thr is not None:
            fig.add_vline(
                x=float(best_cost_thr), line_dash="dot", line_color="#7c3aed",
                annotation_text=f"Best cost {float(best_cost_thr):.3f}",
                annotation_position="top right",
            )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        if "business_cost" in threshold_metrics.columns:
            cost_fig = px.line(
                threshold_metrics, x="threshold", y="business_cost",
                title="Business Cost by Threshold",
                labels={"business_cost": "FP cost + FN cost"},
            )
            cost_fig.add_vline(
                x=float(optimal_threshold), line_dash="solid", line_color="#dc2626",
                annotation_text=f"Active {float(optimal_threshold):.3f}",
                annotation_position="top right",
            )
            st.plotly_chart(cost_fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.warning("Run threshold tuning to generate precision-recall artifacts.")

    st.markdown('<div class="section-label">Active Threshold Decision Quality</div>', unsafe_allow_html=True)
    cm_col, artifact_col = st.columns(2)
    with cm_col:
        if computed_counts and has_confusion_count_fields(computed_counts):
            st.plotly_chart(
                render_dynamic_confusion_matrix(computed_counts, optimal_threshold),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            st.caption(
                "Rows: Actual Legit, Actual Fraud. Columns: Predicted Legit, Predicted Fraud. "
                "Counts are recomputed for the active decision threshold."
            )
        else:
            st.warning(
                "Could not compute the dynamic confusion matrix. Check validation_data.csv or "
                "include threshold summary artifacts with confusion count fields."
            )

    with artifact_col:
        confusion_path    = artifact_path(model_metrics, "xgboost", "confusion_matrix", DEFAULT_XGB_CONFUSION)
        artifact_threshold = stored_xgb_metrics.get("threshold")
        threshold_differs  = (artifact_threshold is not None and
                               abs(float(artifact_threshold) - float(optimal_threshold)) > 1e-9)
        if confusion_path.exists() and threshold_differs:
            with st.expander(f"Static training artifact confusion matrix at threshold {float(artifact_threshold):.3f}", expanded=False):
                st.image(
                    str(confusion_path),
                    caption="Static training artifact, not the active threshold.",
                    use_column_width=True,
                )
        elif confusion_path.exists():
            st.success("Training artifact matches the active-threshold matrix, so duplicate static image is hidden.")
        else:
            st.info("No static training confusion matrix artifact found.")

    st.markdown('<div class="section-label">Static Training Curve Artifacts</div>', unsafe_allow_html=True)
    with st.expander("Precision-recall and ROC curve images", expanded=False):
        curve_col1, curve_col2 = st.columns(2)
        with curve_col1:
            pr_path = artifact_path(model_metrics, "xgboost", "precision_recall_curve",
                                    Path("models/artifacts/xgboost_precision_recall_curve.png"))
            if pr_path.exists():
                st.image(
                    str(pr_path),
                    caption=f"XGBoost Precision-Recall Curve — active threshold {optimal_threshold:.3f}",
                    use_column_width=True,
                )
            else:
                st.warning("Precision-recall curve artifact missing.")
        with curve_col2:
            roc_path = artifact_path(model_metrics, "xgboost", "roc_curve",
                                     Path("models/artifacts/xgboost_roc_curve.png"))
            if roc_path.exists():
                st.image(
                    str(roc_path),
                    caption=f"XGBoost ROC Curve — active threshold {optimal_threshold:.3f}",
                    use_column_width=True,
                )
            else:
                st.warning("ROC curve artifact missing.")

    if threshold_summary:
        with st.expander("Full threshold tuning summary"):
            st.json(threshold_summary)

    st.markdown("---")
    st.subheader("Why AUPRC Matters for Fraud Detection")
    fraud_rate_text = f"{fraud_rate:.4%}" if fraud_rate > 0 else "a very small minority"
    st.write(
        f"Credit card fraud is rare ({fraud_rate_text} of transactions in this dataset), "
        "so accuracy is a misleading metric. AUC measures ranking quality across thresholds, "
        "while **AUPRC focuses on precision and recall for the rare fraud class** — the more "
        f"operationally meaningful metric here. The stored XGBoost AUPRC is **{_auprc:.3f}**. "
        "Use it as validation-set evidence, not as a guaranteed production fraud rate."
    )

# ════════════════════════════════════════════════════════════════════════════════
# RISK INVESTIGATION
# ════════════════════════════════════════════════════════════════════════════════
with investigation_tab:
    st.info(
        "V1-V28 are anonymized PCA features, so SHAP explains mathematical drivers "
        "rather than direct business meanings."
    )

    if alerts_df.empty or "transaction_id" not in alerts_df.columns:
        st.markdown(
            """
            <div class="empty-box">
              <h3>No transactions to investigate yet</h3>
              <p>Generate demo alerts from the sidebar — the dropdown will populate automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        option_rows = alerts_df[
            ["transaction_id", "short_transaction_id", "merchant_name", "risk_display", "decision"]
        ].to_dict("records")
        option_ids  = [r["transaction_id"] for r in option_rows]
        label_by_id = {
            r["transaction_id"]:
            f"{r['short_transaction_id']} | {r['merchant_name']} | {r['risk_display']} | {r['decision']}"
            for r in option_rows
        }

        selected_id = st.selectbox(
            "Choose a recent transaction",
            option_ids,
            format_func=lambda v: label_by_id.get(v, v),
        )
        manual_id   = st.text_input("Or paste a full transaction_id", placeholder="Paste here…")
        search_id   = manual_id.strip() or selected_id
        selected_alert = mongo_client.get_alert_by_transaction_id(search_id)

        if not selected_alert:
            st.warning("No alert found for that transaction_id. Copy the full ID from the Live Feed or use the dropdown.")
        else:
            normalized = prepare_alert_dataframe(
                [selected_alert],
                model_threshold=optimal_threshold,
                save_threshold=save_threshold,
            ).iloc[0]

            decision_val = str(normalized["decision"])
            dec_color    = DECISION_COLORS.get(decision_val.upper(), "#475569")

            st.markdown(
                f"""<div style="margin-bottom:12px;">
                  <span class="decision-pill" style="background:{dec_color};font-size:14px;
                        min-width:100px;padding:6px 16px;">{decision_val}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            cols = st.columns(6)
            cols[0].metric("Risk Score",   format_risk(normalized["risk_score"]))
            cols[1].metric("Probability",  format_probability_display(normalized["fraud_probability"]))
            cols[2].metric("Amount",       format_currency(normalized["Amount"]))
            cols[3].metric("Merchant",     str(normalized.get("merchant_name", "UNKNOWN")))
            cols[4].metric("Hour of Day",  str(int(normalized.get("hour_of_day", 0))))
            cols[5].metric("Night Txn",
                           "Yes" if int(normalized.get("hour_of_day", 12)) >= 22
                               or int(normalized.get("hour_of_day", 12)) <= 5 else "No")

            st.write(f"**Recommended action:** {selected_alert.get('recommended_action', '')}")
            event_time = format_timestamp_display(
                selected_alert.get('event_time_utc') or selected_alert.get('transaction_time_utc')
            )
            stored_time = format_timestamp_display(selected_alert.get('stored_at_utc'))
            st.caption(f"Event time: {event_time} | Stored: {stored_time}")

            shap_rows = selected_alert.get("top_shap_features", [])
            if shap_rows:
                shap_df = pd.DataFrame(shap_rows)
                if "feature_value" not in shap_df.columns and "value" in shap_df.columns:
                    shap_df["feature_value"] = shap_df["value"]
                shap_df["shap_value"]    = pd.to_numeric(shap_df["shap_value"],    errors="coerce").fillna(0.0)
                shap_df["feature_value"] = pd.to_numeric(shap_df["feature_value"], errors="coerce").round(4)
                shap_df["shap_value"]    = shap_df["shap_value"].round(4)
                shap_df["direction"]     = shap_df["shap_value"].map(
                    lambda v: "risk_increasing" if v >= 0 else "risk_decreasing")

                fig = px.bar(
                    shap_df.sort_values("shap_value"),
                    x="shap_value", y="feature", orientation="h",
                    color="direction",
                    title="SHAP Feature Contributions to This Score",
                    color_discrete_map={"risk_increasing": "#dc2626", "risk_decreasing": "#2563eb"},
                )
                fig.update_layout(
                    height=max(280, len(shap_df) * 32),
                    margin=dict(l=10, r=20, t=50, b=30),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                st.caption(
                    "SHAP values are local model-contribution scores for anonymized PCA features. "
                    "They explain why the model score moved up or down; they are not direct business causes."
                )

                increasing = shap_df[shap_df["shap_value"] >= 0].sort_values("shap_value", ascending=False)
                decreasing = shap_df[shap_df["shap_value"] <  0].sort_values("shap_value")
                inc_col, dec_col = st.columns(2)
                with inc_col:
                    st.subheader("🔴 Risk-Increasing Features")
                    if increasing.empty:
                        st.info("No risk-increasing features among the stored top SHAP drivers.")
                    else:
                        st.dataframe(
                            increasing[["feature", "feature_value", "shap_value", "direction"]],
                            use_container_width=True, hide_index=True,
                        )
                with dec_col:
                    st.subheader("🔵 Protective Features")
                    if decreasing.empty:
                        st.info("No protective features among the stored top SHAP drivers.")
                    else:
                        st.dataframe(
                            decreasing[["feature", "feature_value", "shap_value", "direction"]],
                            use_container_width=True, hide_index=True,
                        )
            else:
                st.info("No SHAP explanation was stored for this alert.")

            with st.expander("Full alert document (demo MongoDB record; includes Kaggle ground-truth only for validation)"):
                st.json(sanitize_for_display(selected_alert))

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
