"""
Fraud Detection Operations Dashboard
Streamlit multi-page app — Task 6

Run:  streamlit run app.py
"""

import os
import gdown

# ── Auto-download models from Google Drive ────────────────────────────────────
FILES = {
    "lgbm_model.pkl":     "13aM7g-xYMGOMRH_IGx6cVfOzIdJHJLxA",
    "shap_explainer.pkl": "1A34RIQlzKMiWxpv8YokvT5fSKMuvD2PC",
    "scaler.pkl":         "1mKFKXV3DXW7tc25uA72WC8IMvkfZl1G3",
    "feature_names.pkl":  "1Uq4GxkY1fdVry8md2zM56hZTrHTT3dRI",
    "test_results.csv":   "19fI7ji7Y8lhsb_VFKh9EmnYKMYc9OL0X",
}

for filename, file_id in FILES.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            filename,
            quiet=False
        )


import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=80)
    st.title("Fraud Ops Center")
    st.caption("IEEE-CIS Fraud Detection | LightGBM + SHAP")
    st.divider()

    st.subheader("Global Filters")
    risk_filter = st.multiselect(
        "Risk Tier",
        options=["Critical", "Suspicious", "Clear"],
        default=["Critical", "Suspicious", "Clear"],
        key="global_risk_filter",
    )
    amt_range = st.slider(
        "Transaction Amount ($)",
        min_value=0,
        max_value=20000,
        value=(0, 20000),
        step=100,
        key="global_amt_filter",
    )
    hour_range = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=(0, 23),
        key="global_hour_filter",
    )
    st.divider()
    st.caption("Navigate using the pages above ↑")
    st.caption("Built with LightGBM · SHAP · Optuna · Streamlit")

# ── Home / landing page content ────────────────────────────────────────────────
st.title("🛡️ Fraud Detection Operations Dashboard")
st.markdown("""
Welcome to the **Real-Time Fraud Detection System** — a capstone project built on the
[IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).

---

### Navigate to:

| Page | Description |
|------|-------------|
| **📊 Overview** | KPI cards, class distribution, model performance summary |
| **🔍 Transaction Explorer** | Searchable & filterable transaction table with live risk scores |
| **🧠 SHAP Explainer** | Enter a TransactionID to get a waterfall explanation + plain-English verdict |

---
""")

col1, col2, col3 = st.columns(3)
col1.info("**590,540** total transactions in dataset")
col2.warning("**3.5%** overall fraud rate")
col3.success("**LightGBM** best model (ROC-AUC > 0.95)")

st.markdown("---")
st.markdown("*Use the sidebar filters to refine results on the Explorer and Overview pages.*")
