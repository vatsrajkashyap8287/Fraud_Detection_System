"""
Page 2 — Transaction Explorer
Searchable, filterable table with live risk score display
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Transaction Explorer", page_icon="🔍", layout="wide")


@st.cache_data
def load_results():
    path = "test_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        rng = np.random.default_rng(42)
        n = 10000
        fraud_prob = np.concatenate([
            rng.beta(1, 30, int(n * 0.96)),
            rng.beta(8, 2,  int(n * 0.04)),
        ])
        df = pd.DataFrame({
            "TransactionID":  np.arange(1000000, 1000000 + n),
            "TransactionAmt": rng.lognormal(4.5, 1.2, n).round(2),
            "HourOfDay":      rng.integers(0, 24, n),
            "DayOfWeek":      rng.integers(0, 7, n),
            "isFraud":        (fraud_prob > 0.5).astype(int),
            "fraud_prob":     fraud_prob.round(4),
            "fraud_pred":     (fraud_prob >= 0.35).astype(int),
            "RiskTier": pd.cut(fraud_prob, bins=[-0.01, 0.40, 0.75, 1.01],
                               labels=["Clear", "Suspicious", "Critical"]).astype(str),
        })
    df["RiskTier"] = df["RiskTier"].astype(str)
    if "TransactionID" not in df.columns:
        df["TransactionID"] = range(1000000, 1000000 + len(df))
    return df


df_all = load_results()
TIER_COLORS = {"Critical": "🔴", "Suspicious": "🟡", "Clear": "🟢"}

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🔍 Transaction Explorer")
st.caption("Search and filter transactions — click any row for detailed risk profile")
st.divider()

# ── Search bar ─────────────────────────────────────────────────────────────────
col_search, col_btn = st.columns([4, 1])
with col_search:
    txn_search = st.text_input(
        "Search by TransactionID",
        placeholder="e.g. 1000042",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("🔎 Search", use_container_width=True)

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Filters", expanded=True):
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        tier_filter = st.multiselect("Risk Tier", ["Critical","Suspicious","Clear"],
                                     default=["Critical","Suspicious","Clear"])
    with fc2:
        label_filter = st.selectbox("True Label", ["All", "Fraud (1)", "Legitimate (0)"])
    with fc3:
        amt_filter = st.slider("Amount Range ($)", 0, 20000, (0, 20000), step=100)
    with fc4:
        hour_filter = st.slider("Hour of Day", 0, 23, (0, 23))

    sort_col = st.selectbox("Sort by", ["fraud_prob", "TransactionAmt", "HourOfDay"],
                            index=0)
    sort_asc = st.checkbox("Ascending", value=False)

# ── Apply filters ──────────────────────────────────────────────────────────────
df = df_all.copy()

if txn_search.strip():
    try:
        tid = int(txn_search.strip())
        df  = df[df["TransactionID"] == tid]
    except ValueError:
        st.warning("TransactionID must be a number.")

df = df[df["RiskTier"].isin(tier_filter)]

if label_filter == "Fraud (1)":
    df = df[df["isFraud"] == 1]
elif label_filter == "Legitimate (0)":
    df = df[df["isFraud"] == 0]

df = df[df["TransactionAmt"].between(*amt_filter)]
df = df[df["HourOfDay"].between(*hour_filter)]
df = df.sort_values(sort_col, ascending=sort_asc)

# ── Metrics row ────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filtered Transactions", f"{len(df):,}")
m2.metric("Fraud in View",         f"{df['isFraud'].sum():,}")
m3.metric("Avg Probability",       f"{df['fraud_prob'].mean():.3f}")
m4.metric("Avg Amount",            f"${df['TransactionAmt'].mean():,.2f}")

st.divider()

# ── Single transaction detail card ────────────────────────────────────────────
if txn_search.strip() and len(df) == 1:
    row = df.iloc[0]
    prob  = row["fraud_prob"]
    tier  = row["RiskTier"]
    emoji = TIER_COLORS.get(tier, "⚪")

    st.subheader(f"{emoji} Transaction Detail — ID {int(row['TransactionID'])}")

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Fraud Probability", f"{prob:.4f}")
    dc2.metric("Risk Tier", tier)
    dc3.metric("Amount", f"${row['TransactionAmt']:,.2f}")
    dc4.metric("True Label", "🚨 FRAUD" if row["isFraud"] == 1 else "✅ Legit")

    # Risk gauge
    fig_gauge = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number+delta",
            "value": round(prob * 100, 1),
            "delta": {"reference": 50},
            "gauge": {
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#F44336" if prob > 0.75 else
                                   "#FF9800" if prob > 0.40 else "#4CAF50"},
                "steps": [
                    {"range": [0, 40],  "color": "#E8F5E9"},
                    {"range": [40, 75], "color": "#FFF8E1"},
                    {"range": [75, 100],"color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
            "title": {"text": "Fraud Risk Score (%)"},
        }]
    }
    import plotly.graph_objects as go
    fig_g = go.Figure(fig_gauge)
    fig_g.update_layout(height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

    st.info("💡 Go to the **SHAP Explainer** page to understand why this score was assigned.")
    st.divider()

# ── Data table ────────────────────────────────────────────────────────────────
display_cols = [c for c in ["TransactionID","TransactionAmt","HourOfDay",
                             "DayOfWeek","isFraud","fraud_prob","RiskTier"]
                if c in df.columns]

st.subheader(f"Transactions Table ({len(df):,} rows)")

# Add emoji tier column for visual clarity
df_show = df[display_cols].copy()
df_show["Tier"] = df_show["RiskTier"].map(TIER_COLORS)
df_show["fraud_prob"] = df_show["fraud_prob"].round(4)
df_show["TransactionAmt"] = df_show["TransactionAmt"].round(2)

st.dataframe(
    df_show.head(500),
    use_container_width=True,
    height=420,
    column_config={
        "fraud_prob": st.column_config.ProgressColumn(
            "Fraud Probability",
            min_value=0, max_value=1, format="%.4f",
        ),
        "TransactionAmt": st.column_config.NumberColumn(
            "Amount ($)", format="$%.2f"
        ),
        "isFraud": st.column_config.CheckboxColumn("Actual Fraud"),
    },
)

if len(df) > 500:
    st.caption(f"Showing first 500 of {len(df):,} rows. Apply filters to narrow results.")

# ── Scatter bonus chart ────────────────────────────────────────────────────────
st.divider()
st.subheader("Interactive Scatter — Amount vs Hour (colored by Risk)")

sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

fig_scatter = px.scatter(
    sample,
    x="HourOfDay", y="TransactionAmt",
    color="fraud_prob",
    color_continuous_scale="RdYlGn_r",
    log_y=True, opacity=0.5,
    hover_data=["TransactionID", "RiskTier", "isFraud"],
    labels={"fraud_prob": "Fraud Prob", "TransactionAmt": "Amount (log)"},
)
fig_scatter.update_layout(height=380, margin=dict(t=10, b=10))
st.plotly_chart(fig_scatter, use_container_width=True)
