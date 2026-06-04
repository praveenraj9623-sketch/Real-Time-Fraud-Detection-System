"""Streamlit dashboard for real-time fraud monitoring."""

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
    build_active_xgb_metrics,
    confusion_matrix_values,
    format_currency,
    format_percent,
    format_risk,
    merchant_risk_summary,
    metrics_from_confusion_counts,
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
    "REVIEW": "#2563eb",
    "FLAGGED": "#d97706",
    "BLOCKED": "#dc2626",
}

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1550px; }
    .hero {
        border: 1px solid #d9e2ef;
        border-radius: 8px;
        padding: 18px 22px;
        background: #f8fbff;
        margin-bottom: 14px;
    }
    .hero h1 { margin: 0 0 4px 0; font-size: 32px; letter-spacing: 0; }
    .hero p { margin: 0; color: #516070; }
    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid #cbd5e1;
        margin: 4px 8px 4px 0;
        font-size: 12px;
        background: #fff;
    }
    .empty-box {
        border: 1px dashed #b8c4d6;
        border-radius: 8px;
        padding: 18px;
        background: #fbfdff;
        color: #334155;
    }
    .decision-pill {
        display: inline-block;
        min-width: 84px;
        text-align: center;
        padding: 3px 9px;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 27px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_mongo_client() -> MongoDBClient:
    """Connect once to MongoDB."""
    return MongoDBClient()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def get_api_health() -> Dict[str, Any]:
    try:
        response = requests.get(API_HEALTH_URL, timeout=1.5)
        return response.json() if response.ok else {"status": f"http_{response.status_code}"}
    except Exception:
        return {"status": "offline"}


@st.cache_data(show_spinner=False)
def compute_xgb_validation_metrics(threshold: float) -> Dict[str, Any]:
    """Compute active-threshold metrics if the validation CSV and model exist."""
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
        metrics.update(
            {
                "threshold": float(threshold),
                "auc": float(roc_auc_score(y_true, scores)),
                "auprc": float(average_precision_score(y_true, scores)),
                "metric_source": "active_threshold_recomputed",
            }
        )
        metrics.update({
            "predicted_fraud_count": int(y_pred.sum()),
            "fraud_cases": int((y_true == 1).sum()),
            "validation_rows": int(len(validation)),
        })
        return metrics
    except Exception:
        return {}


def render_dynamic_confusion_matrix(metrics: Dict[str, Any], threshold: float) -> go.Figure:
    """Render a confusion matrix that matches the active-threshold table counts."""
    z = confusion_matrix_values(metrics)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["True Legit", "True Fraud"],
            text=z,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Dynamic Confusion Matrix at Active Threshold {threshold:.3f}",
        height=420,
        margin=dict(l=25, r=25, t=70, b=35),
    )
    return fig


def metrics_from_threshold_summary(summary: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Fallback active-threshold metrics from threshold_tuning_summary.json."""
    for key in ("best_f1_threshold", "best_business_cost_threshold"):
        row = summary.get(key, {})
        if row and abs(float(row.get("threshold", -1.0)) - float(threshold)) <= 1e-9:
            fallback = dict(row)
            fallback["metric_source"] = f"{key}_summary"
            return fallback
    return {}


def render_risk_gauge(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(min(value, 100.0), 0.0),
            number={"suffix": " / 100"},
            title={"text": "Average Alert Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 75], "color": "#fef3c7"},
                    {"range": [75, 100], "color": "#fecaca"},
                ],
            },
        )
    )
    fig.update_layout(height=310, margin=dict(l=15, r=15, t=45, b=15))
    return fig


def artifact_path(metrics: Dict[str, Any], model: str, artifact: str, fallback: Path) -> Path:
    value = metrics.get("artifacts", {}).get(model, {}).get(artifact)
    return normalize_artifact_path(value, fallback)


def styled_decision_table(table: pd.DataFrame):
    def color_decision(value: Any) -> str:
        color = DECISION_COLORS.get(str(value).upper(), "#475569")
        return f"background-color: {color}; color: white; font-weight: 700;"

    return table.style.applymap(color_decision, subset=["decision"])


model_metrics = load_json(MODEL_METRICS_PATH)
training_stats = load_json(TRAINING_STATS_PATH)
threshold_summary = load_json(THRESHOLD_SUMMARY_PATH)
dataset_stats = training_stats.get("dataset", {})
optimal_threshold = clip_probability(training_stats.get("optimal_threshold", 0.5))
save_threshold = float(os.getenv("ALERT_SAVE_THRESHOLD", str(DEFAULT_SAVE_THRESHOLD)))
api_health = get_api_health()
last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    mongo_client = get_mongo_client()
except Exception as exc:
    st.error("MongoDB is not reachable. Start MongoDB service, then refresh this page.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.subheader("Controls")
    recent_limit = st.slider("Recent alerts to show", min_value=20, max_value=500, value=100, step=20)
    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_seconds = st.slider("Refresh every seconds", min_value=5, max_value=60, value=10, step=5)
    live_chart_view = st.radio(
        "Live chart view",
        ["Highest Risk", "Recent Mixed Alerts", "Group by Decision"],
        index=0,
    )

    st.divider()
    st.subheader("Demo Data")
    st.caption("This scores real creditcard.csv rows with the trained model, then saves a varied alert sample.")
    demo_rows = st.number_input("Rows to scan", min_value=1000, max_value=100000, value=30000, step=1000)
    demo_threshold = st.slider("Save if probability >=", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
    demo_max_alerts = st.slider("Max demo alerts", min_value=10, max_value=200, value=60, step=10)

    if st.button("Generate Demo Alerts", type="primary", use_container_width=True):
        with st.spinner("Scoring transactions and saving realistic demo alerts..."):
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

recent_records = mongo_client.get_recent_alerts(limit=recent_limit)
alerts_df = prepare_alert_dataframe(
    recent_records,
    model_threshold=optimal_threshold,
    save_threshold=save_threshold,
)
summary = summarize_alerts(alerts_df)

api_status = api_health.get("status", "unknown")
fraud_rate = float(dataset_stats.get("fraud_rate", 0.0))
training_rows = int(dataset_stats.get("row_count", 0) or 0)

st.markdown(
    f"""
    <div class="hero">
      <h1>Real-Time Fraud Detection Monitor</h1>
      <p>Scores credit card transactions, stores suspicious activity, and gives analysts a practical investigation view.</p>
      <div style="margin-top: 12px;">
        <span class="status-pill">MongoDB connected</span>
        <span class="status-pill">API/model: {api_status}</span>
        <span class="status-pill">Block threshold: {optimal_threshold:.3f}</span>
        <span class="status-pill">Training rows: {training_rows:,}</span>
        <span class="status-pill">Dataset fraud rate: {fraud_rate:.4%}</span>
        <span class="status-pill">Last refresh: {last_refresh}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

live_tab, analytics_tab, performance_tab, investigation_tab = st.tabs(
    ["Live Feed", "Analytics", "Model Performance", "Risk Investigation"]
)

with live_tab:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Recent Alerts", f"{summary['alert_count']:,}")
    col2.metric("Review", f"{summary['review_count']:,}")
    col3.metric("Flagged", f"{summary['flagged_count']:,}")
    col4.metric("Blocked", f"{summary['blocked_count']:,}")
    col5.metric("Alerted Amount", format_currency(summary["alerted_amount"]))
    col6.metric("Avg Risk", format_risk(summary["average_risk"]))

    st.metric("Blocked Amount", format_currency(summary["blocked_amount"]))

    if alerts_df.empty:
        st.markdown(
            """
            <div class="empty-box">
            No alerts are stored yet. Use <b>Generate Demo Alerts</b> in the left sidebar.
            That button reads real rows from <b>data/raw/creditcard.csv</b>, scores them with the trained XGBoost model,
            saves review/flagged/blocked records into MongoDB, and then this table fills with transaction IDs.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        gauge_col, chart_col = st.columns([1, 2])
        with gauge_col:
            st.plotly_chart(render_risk_gauge(float(summary["average_risk"])), use_container_width=True)
        with chart_col:
            if live_chart_view == "Recent Mixed Alerts":
                recent_mixed = alerts_df.sort_values("event_time_utc" if "event_time_utc" in alerts_df else "stored_at_utc", ascending=False).head(24)
                fig = px.bar(
                    recent_mixed,
                    x="risk_score",
                    y="short_transaction_id",
                    color="decision",
                    orientation="h",
                    title="Recent Mixed Alerts by Risk Score",
                    labels={"risk_score": "Risk Score", "short_transaction_id": "Transaction"},
                    hover_data=["transaction_id", "merchant_name", "amount_display", "probability_display"],
                    color_discrete_map=DECISION_COLORS,
                )
                fig.update_xaxes(range=[0, 100])
                fig.update_layout(height=340, yaxis={"categoryorder": "array", "categoryarray": recent_mixed["short_transaction_id"].tolist()[::-1]})
            elif live_chart_view == "Group by Decision":
                decision_summary = (
                    alerts_df.groupby("decision")
                    .agg(alert_count=("transaction_id", "count"), avg_risk_score=("risk_score", "mean"), total_amount=("Amount", "sum"))
                    .reset_index()
                )
                fig = px.bar(
                    decision_summary,
                    x="decision",
                    y="alert_count",
                    color="decision",
                    title="Saved Alerts Grouped by Decision",
                    hover_data=["avg_risk_score", "total_amount"],
                    color_discrete_map=DECISION_COLORS,
                )
                fig.update_layout(height=340, showlegend=False)
            else:
                top_alerts = alerts_df.sort_values(["risk_score", "Amount"], ascending=[False, False]).head(20)
                fig = px.bar(
                    top_alerts,
                    x="risk_score",
                    y="short_transaction_id",
                    color="decision",
                    orientation="h",
                    title="Highest-Risk Recent Transactions",
                    labels={"risk_score": "Risk Score", "short_transaction_id": "Transaction"},
                    hover_data={
                        "transaction_id": True,
                        "merchant_name": True,
                        "amount_display": True,
                        "probability_display": True,
                        "risk_display": True,
                        "risk_score": False,
                    },
                    color_discrete_map=DECISION_COLORS,
                )
                fig.update_xaxes(range=[0, 100])
                fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        display_columns = [
            "short_transaction_id",
            "merchant_name",
            "amount_display",
            "probability_display",
            "risk_display",
            "decision",
            "action_short",
            "event_time_utc",
            "stored_at_utc",
        ]
        display_columns = [column for column in display_columns if column in alerts_df.columns]
        st.subheader("Recent Alert Feed")
        st.dataframe(
            styled_decision_table(alerts_df[display_columns].rename(
                columns={
                    "short_transaction_id": "transaction_id",
                    "amount_display": "amount",
                    "probability_display": "fraud_probability",
                    "risk_display": "risk_score",
                    "action_short": "recommended_action",
                }
            )),
            use_container_width=True,
            hide_index=True,
        )

with analytics_tab:
    hourly = pd.DataFrame(mongo_client.get_fraud_rate_by_hour())
    merchant_summary = merchant_risk_summary(alerts_df)

    if alerts_df.empty:
        st.info("No alert analytics yet. Generate demo alerts from the sidebar.")
    else:
        left, right = st.columns(2)
        with left:
            if not hourly.empty:
                fig = px.bar(
                    hourly,
                    x="hour_of_day",
                    y="fraud_alerts",
                    color="avg_risk_score",
                    hover_data=["fraud_rate", "total_amount", "blocked_count", "flagged_count"],
                    title="Alert Count by Hour of Day",
                    labels={"hour_of_day": "Hour of Day", "fraud_alerts": "Saved Alerts", "avg_risk_score": "Avg Risk"},
                    color_continuous_scale="YlOrRd",
                )
                fig.update_xaxes(tickmode="array", tickvals=list(range(24)))
                st.plotly_chart(fig, use_container_width=True)
                active_hours = int((hourly["fraud_alerts"] > 0).sum())
                if active_hours <= 1:
                    st.info("All saved alerts are in one hour. Generate a fresh demo batch to spread timestamps across the last 7 days.")
        with right:
            use_log = st.checkbox("Use log scale for amount distribution", value=False)
            amount_df = alerts_df.copy()
            amount_df["Amount"] = amount_df["Amount"].clip(lower=0.01)
            fig = px.histogram(
                amount_df,
                x="Amount",
                color="decision",
                nbins=35,
                title="Alerted Amount Distribution",
                labels={"Amount": "Amount"},
                color_discrete_map=DECISION_COLORS,
            )
            fig.update_xaxes(tickprefix="$")
            if use_log:
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)

        if not merchant_summary.empty:
            scatter_df = merchant_summary.copy()
            scatter_df["bubble_amount"] = scatter_df["total_amount"].clip(lower=1.0)
            fig = px.scatter(
                scatter_df,
                x="alert_count",
                y="avg_risk_score",
                size="bubble_amount",
                color="blocked_count",
                hover_name="merchant_name",
                hover_data=["alert_count", "avg_risk_score", "total_amount", "blocked_count", "flagged_count"],
                title="Merchant Alert Volume vs Average Fraud Risk",
                labels={"alert_count": "Alert Count", "avg_risk_score": "Average Risk Score"},
                color_continuous_scale="Reds",
            )
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("High-Risk Merchants")
            merchant_table = pd.DataFrame(mongo_client.get_high_risk_merchants(limit=15))
            if not merchant_table.empty:
                merchant_table["avg_risk_score"] = merchant_table["avg_risk_score"].map(lambda v: f"{float(v):.1f} / 100")
                merchant_table["max_risk_score"] = merchant_table["max_risk_score"].map(lambda v: f"{float(v):.1f} / 100")
                merchant_table["total_amount"] = merchant_table["total_amount"].map(format_currency)
                st.dataframe(merchant_table, use_container_width=True, hide_index=True)

with performance_tab:
    stored_xgb_metrics = dict(model_metrics.get("xgboost", {}))
    iforest_metrics = dict(model_metrics.get("isolation_forest", {}))
    computed_counts = compute_xgb_validation_metrics(optimal_threshold) or metrics_from_threshold_summary(threshold_summary, optimal_threshold)
    xgb_metrics = build_active_xgb_metrics(stored_xgb_metrics, computed_counts, optimal_threshold)

    comparison_rows = [
        {"model": "XGBoost", **xgb_metrics},
        {"model": "Isolation Forest", **iforest_metrics},
    ]
    comparison = pd.DataFrame(comparison_rows)

    st.subheader("Model Comparison")
    if not comparison.empty:
        preferred = [
            "model",
            "precision",
            "recall",
            "f1",
            "auc",
            "auprc",
            "threshold",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "predicted_fraud_count",
            "metric_source",
        ]
        visible = [column for column in preferred if column in comparison.columns]
        st.dataframe(comparison[visible], use_container_width=True, hide_index=True)

    if computed_counts:
        caught = computed_counts["true_positives"]
        total_fraud = computed_counts["fraud_cases"]
        false_alarms = computed_counts["false_positives"]
        missed = computed_counts["false_negatives"]
        st.info(
            f"At threshold {optimal_threshold:.3f}, XGBoost catches {caught} out of {total_fraud} fraud cases, "
            f"incorrectly flags {false_alarms} legitimate transactions, and misses {missed} fraud cases."
        )

    threshold_metrics = pd.DataFrame()
    if THRESHOLD_METRICS_PATH.exists():
        threshold_metrics = pd.read_csv(THRESHOLD_METRICS_PATH)

    left, right = st.columns(2)
    with left:
        if not threshold_metrics.empty:
            fig = px.line(
                threshold_metrics,
                x="threshold",
                y=["precision", "recall", "f1"],
                markers=False,
                title="Precision, Recall, and F1 by Threshold",
            )
            best_f1_threshold = threshold_summary.get("best_f1_threshold", {}).get("threshold", optimal_threshold)
            best_cost_threshold = threshold_summary.get("best_business_cost_threshold", {}).get("threshold")
            fig.add_vline(
                x=float(best_f1_threshold),
                line_dash="dash",
                line_color="black",
                annotation_text=f"Best F1 {float(best_f1_threshold):.3f}",
                annotation_position="top left",
            )
            if best_cost_threshold is not None:
                fig.add_vline(
                    x=float(best_cost_threshold),
                    line_dash="dot",
                    line_color="#7c3aed",
                    annotation_text=f"Best cost {float(best_cost_threshold):.3f}",
                    annotation_position="top right",
                )
            st.plotly_chart(fig, use_container_width=True)
            if "business_cost" in threshold_metrics.columns:
                cost_fig = px.line(
                    threshold_metrics,
                    x="threshold",
                    y="business_cost",
                    title="Business Cost by Threshold",
                    labels={"business_cost": "FP cost + FN cost"},
                )
                st.plotly_chart(cost_fig, use_container_width=True)
        else:
            st.warning("Run threshold tuning to generate precision-recall artifacts.")

    with right:
        if computed_counts:
            st.plotly_chart(render_dynamic_confusion_matrix(computed_counts, optimal_threshold), use_container_width=True)
        else:
            st.warning("Could not compute the dynamic confusion matrix. Check validation_data.csv and fraud_classifier.joblib.")

        confusion_path = artifact_path(model_metrics, "xgboost", "confusion_matrix", DEFAULT_XGB_CONFUSION)
        artifact_threshold = stored_xgb_metrics.get("threshold")
        threshold_differs = artifact_threshold is not None and abs(float(artifact_threshold) - float(optimal_threshold)) > 1e-9
        if confusion_path.exists() and threshold_differs:
            with st.expander(f"Training artifact confusion matrix at threshold {float(artifact_threshold):.3f}"):
                st.image(str(confusion_path), caption="Static training artifact, not the active threshold.")

    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        pr_path = artifact_path(model_metrics, "xgboost", "precision_recall_curve", Path("models/artifacts/xgboost_precision_recall_curve.png"))
        if pr_path.exists():
            st.image(str(pr_path), caption=f"XGBoost Precision-Recall Curve. Active decision threshold: {optimal_threshold:.3f}.")
        else:
            st.warning("Precision-recall curve artifact is missing.")
    with curve_col2:
        roc_path = artifact_path(model_metrics, "xgboost", "roc_curve", Path("models/artifacts/xgboost_roc_curve.png"))
        if roc_path.exists():
            st.image(str(roc_path), caption=f"XGBoost ROC Curve. Active decision threshold: {optimal_threshold:.3f}.")
        else:
            st.warning("ROC curve artifact is missing.")

    if threshold_summary:
        with st.expander("Threshold tuning summary"):
            st.json(threshold_summary)

    st.subheader("Why AUPRC Matters")
    st.write(
        "Credit card fraud is rare, so ordinary accuracy is misleading. AUC measures ranking quality across many "
        "false-positive rates, while AUPRC focuses on precision and recall for the rare fraud class. For this "
        "dataset, AUPRC, recall, precision, and F1 are the useful portfolio metrics."
    )

with investigation_tab:
    st.info(
        "V1-V28 are anonymized PCA features from the dataset, so SHAP explains model behavior mathematically, "
        "not direct business meaning like merchant category or device type."
    )

    if alerts_df.empty or "transaction_id" not in alerts_df.columns:
        st.markdown(
            """
            <div class="empty-box">
            No transaction IDs are available yet. Generate demo alerts from the sidebar, then come back here.
            The dropdown will automatically show recent transaction IDs.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        option_rows = alerts_df[["transaction_id", "short_transaction_id", "merchant_name", "risk_display", "decision"]].to_dict("records")
        option_ids = [row["transaction_id"] for row in option_rows]
        label_by_id = {
            row["transaction_id"]: f"{row['short_transaction_id']} | {row['merchant_name']} | {row['risk_display']} | {row['decision']}"
            for row in option_rows
        }
        selected_id = st.selectbox("Choose a recent transaction", option_ids, format_func=lambda value: label_by_id.get(value, value))
        manual_id = st.text_input("Or paste a full transaction_id", placeholder="Paste transaction_id here")
        search_id = manual_id.strip() or selected_id
        selected_alert = mongo_client.get_alert_by_transaction_id(search_id)

        if not selected_alert:
            st.warning("No alert found for that transaction_id. Copy the full ID from the Live Feed full document or use the dropdown.")
        else:
            normalized = prepare_alert_dataframe([selected_alert], model_threshold=optimal_threshold, save_threshold=save_threshold).iloc[0]
            cols = st.columns(6)
            cols[0].metric("Decision", str(normalized["decision"]))
            cols[1].metric("Risk", format_risk(normalized["risk_score"]))
            cols[2].metric("Probability", format_percent(normalized["fraud_probability"]))
            cols[3].metric("Amount", format_currency(normalized["Amount"]))
            cols[4].metric("Merchant", str(normalized.get("merchant_name", "UNKNOWN")))
            cols[5].metric("Hour", str(int(normalized.get("hour_of_day", 0))))
            st.write(f"Recommended action: **{selected_alert.get('recommended_action', '')}**")
            st.caption(
                "Event time: "
                f"{selected_alert.get('event_time_utc') or selected_alert.get('transaction_time_utc') or 'N/A'} | "
                f"Stored at: {selected_alert.get('stored_at_utc') or 'N/A'}"
            )

            shap_rows = selected_alert.get("top_shap_features", [])
            if shap_rows:
                shap_df = pd.DataFrame(shap_rows)
                if "feature_value" not in shap_df.columns and "value" in shap_df.columns:
                    shap_df["feature_value"] = shap_df["value"]
                shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"], errors="coerce").fillna(0.0)
                shap_df["direction"] = shap_df["shap_value"].map(lambda value: "risk_increasing" if value >= 0 else "risk_decreasing")
                shap_df["feature_value"] = pd.to_numeric(shap_df["feature_value"], errors="coerce").round(4)
                shap_df["shap_value"] = shap_df["shap_value"].round(4)

                fig = px.bar(
                    shap_df.sort_values("shap_value"),
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="direction",
                    title="Top SHAP Features Driving This Score",
                    color_discrete_map={"risk_increasing": "#dc2626", "risk_decreasing": "#2563eb"},
                )
                st.plotly_chart(fig, use_container_width=True)

                increasing = shap_df[shap_df["shap_value"] >= 0].sort_values("shap_value", ascending=False)
                decreasing = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value")
                inc_col, dec_col = st.columns(2)
                with inc_col:
                    st.subheader("Risk-Increasing Features")
                    st.dataframe(increasing[["feature", "feature_value", "shap_value", "direction"]], use_container_width=True, hide_index=True)
                with dec_col:
                    st.subheader("Protective Features")
                    st.dataframe(decreasing[["feature", "feature_value", "shap_value", "direction"]], use_container_width=True, hide_index=True)
            else:
                st.info("No SHAP explanation was stored for this alert.")

            with st.expander("Full alert document"):
                st.json(selected_alert)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
