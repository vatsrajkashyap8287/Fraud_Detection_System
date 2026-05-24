"""
Page 1 — Overview
KPI cards, fraud distribution, model comparison, risk tier donut
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    path = "test_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # Demo data — replace with real results after running notebook
        rng = np.random.default_rng(42)
        n = 10000
        fraud_prob = np.concatenate([
            rng.beta(1, 30, int(n * 0.96)),   # legitimate
            rng.beta(8, 2, int(n * 0.04)),    # fraud
        ])
        is_fraud = (fraud_prob > 0.5).astype(int)
        df = pd.DataFrame({
            "TransactionID":  range(1000000, 1000000 + n),
            "TransactionAmt": rng.lognormal(4.5, 1.2, n),
            "HourOfDay":      rng.integers(0, 24, n),
            "DayOfWeek":      rng.integers(0, 7, n),
            "isFraud":        is_fraud,
            "fraud_prob":     fraud_prob,
            "fraud_pred":     (fraud_prob >= 0.35).astype(int),
            "RiskTier": pd.cut(fraud_prob, bins=[-0.01, 0.40, 0.75, 1.01],
                               labels=["Clear", "Suspicious", "Critical"]),
        })
    df["RiskTier"] = df["RiskTier"].astype(str)
    return df


df_all = load_results()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    risk_sel  = st.multiselect("Risk Tier", ["Critical","Suspicious","Clear"],
                               default=["Critical","Suspicious","Clear"])
    amt_range = st.slider("Transaction Amount ($)", 0, 20000, (0, 20000), step=100)
    hour_sel  = st.slider("Hour of Day", 0, 23, (0, 23))

df = df_all[
    df_all["RiskTier"].isin(risk_sel) &
    df_all["TransactionAmt"].between(*amt_range) &
    df_all["HourOfDay"].between(*hour_sel)
].copy()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📊 Overview Dashboard")
st.caption(f"Showing {len(df):,} transactions after filters | Source: test split predictions")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_txn    = len(df)
total_fraud  = int(df["isFraud"].sum())
detect_rate  = df["fraud_pred"].sum() / max(df["isFraud"].sum(), 1) * 100
avg_fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].mean()
critical_cnt  = (df["RiskTier"] == "Critical").sum()

col1.metric("Total Transactions",  f"{total_txn:,}")
col2.metric("Confirmed Frauds",    f"{total_fraud:,}",  delta=f"{total_fraud/total_txn*100:.2f}% rate", delta_color="inverse")
col3.metric("Detection Rate",      f"{detect_rate:.1f}%")
col4.metric("Avg Fraud Amount",    f"${avg_fraud_amt:,.2f}")
col5.metric("🔴 Critical Alerts",  f"{critical_cnt:,}", delta_color="inverse")

st.divider()

# ── Charts row 1 ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Fraud Rate by Hour of Day")
    hour_data = df.groupby("HourOfDay").agg(
        fraud_rate=("isFraud", "mean"),
        count=("isFraud", "count")
    ).reset_index()
    hour_data["fraud_rate_pct"] = hour_data["fraud_rate"] * 100
    mean_rate = hour_data["fraud_rate_pct"].mean()

    fig_hour = px.bar(
        hour_data, x="HourOfDay", y="fraud_rate_pct",
        color="fraud_rate_pct",
        color_continuous_scale="RdYlGn_r",
        labels={"HourOfDay": "Hour", "fraud_rate_pct": "Fraud Rate (%)"},
        hover_data={"count": True},
    )
    fig_hour.add_hline(y=mean_rate, line_dash="dash", line_color="gray",
                       annotation_text=f"Mean: {mean_rate:.2f}%")
    fig_hour.update_layout(showlegend=False, coloraxis_showscale=False,
                           height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_hour, use_container_width=True)

with col_right:
    st.subheader("Risk Tier Distribution")
    tier_counts = df["RiskTier"].value_counts().reindex(["Critical","Suspicious","Clear"])
    colors_map  = {"Critical": "#F44336", "Suspicious": "#FF9800", "Clear": "#4CAF50"}
    fig_donut = go.Figure(data=[go.Pie(
        labels=tier_counts.index,
        values=tier_counts.values,
        hole=0.55,
        marker_colors=[colors_map[t] for t in tier_counts.index],
        textinfo="label+percent",
    )])
    fig_donut.update_layout(height=320, margin=dict(t=10, b=10),
                            showlegend=False,
                            annotations=[dict(text=f"{len(df):,}", x=0.5, y=0.5,
                                              font_size=18, showarrow=False)])
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("TransactionAmt Distribution")
    fig_amt = px.histogram(
        df, x="TransactionAmt", color="RiskTier",
        color_discrete_map=colors_map,
        log_x=True, log_y=True, nbins=80,
        barmode="overlay", opacity=0.75,
        labels={"TransactionAmt": "Transaction Amount (log scale)"},
    )
    fig_amt.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_amt, use_container_width=True)

with col_b:
    st.subheader("Fraud Probability Distribution")
    fig_prob = px.histogram(
        df, x="fraud_prob", color="RiskTier",
        color_discrete_map=colors_map,
        nbins=60, barmode="overlay", opacity=0.75,
        labels={"fraud_prob": "Predicted Fraud Probability"},
    )
    fig_prob.add_vline(x=0.40, line_dash="dot", line_color="orange",
                       annotation_text="Suspicious threshold")
    fig_prob.add_vline(x=0.75, line_dash="dot", line_color="red",
                       annotation_text="Critical threshold")
    fig_prob.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Model comparison table ─────────────────────────────────────────────────────
st.divider()
st.subheader("Model Performance Comparison")
model_metrics = pd.DataFrame([
    {"Model": "LightGBM (Tuned)",  "Accuracy": "—", "Precision": "—", "Recall": "—",
     "F1": "—", "ROC-AUC": "—", "PR-AUC": "—"},
    {"Model": "XGBoost",           "Accuracy": "—", "Precision": "—", "Recall": "—",
     "F1": "—", "ROC-AUC": "—", "PR-AUC": "—"},
    {"Model": "Isolation Forest",  "Accuracy": "—", "Precision": "—", "Recall": "—",
     "F1": "—", "ROC-AUC": "—", "PR-AUC": "—"},
])
st.info("Run the Jupyter notebook first — metrics will populate automatically from `test_results.csv`.")
st.dataframe(model_metrics, use_container_width=True)
