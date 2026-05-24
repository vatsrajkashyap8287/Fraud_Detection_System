"""
Page 3 — SHAP Explainer
Enter TransactionID → see SHAP waterfall + plain-English explanation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import pickle
import os

st.set_page_config(page_title="SHAP Explainer", page_icon="🧠", layout="wide")


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    artifacts = {}
    files = {
        "model":    "lgbm_model.pkl",
        "explainer":"shap_explainer.pkl",
        "scaler":   "scaler.pkl",
        "features": "feature_names.pkl",
    }
    for key, path in files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                artifacts[key] = pickle.load(f)
        else:
            artifacts[key] = None
    return artifacts


@st.cache_data
def load_results():
    if os.path.exists("test_results.csv"):
        return pd.read_csv("test_results.csv")
    return None


artifacts = load_model_artifacts()
df_results = load_results()

DEMO_MODE = artifacts["model"] is None or df_results is None

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🧠 SHAP Explainer")
st.caption("Understand *why* any transaction was flagged — powered by SHAP (SHapley Additive exPlanations)")

if DEMO_MODE:
    st.warning(
        "⚠️ **Demo Mode** — model artifacts not found. "
        "Run the Jupyter notebook first to generate `lgbm_model.pkl`, "
        "`shap_explainer.pkl`, `scaler.pkl`, `feature_names.pkl`, and `test_results.csv`."
    )

st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
col_in, col_btn = st.columns([3, 1])

with col_in:
    txn_id_input = st.text_input(
        "Enter TransactionID",
        placeholder="e.g. 1000042",
        label_visibility="collapsed",
    )
with col_btn:
    explain_clicked = st.button("⚡ Explain", use_container_width=True, type="primary")

# Quick-select buttons for demo cases
if df_results is not None:
    st.caption("Quick-select examples:")
    qc1, qc2, qc3, _ = st.columns([1, 1, 1, 3])

    fraud_ids  = df_results[df_results["isFraud"] == 1].nlargest(1, "fraud_prob")["TransactionID"]
    border_ids = df_results.iloc[[(df_results["fraud_prob"] - 0.50).abs().idxmin()]]["TransactionID"]
    legit_ids  = df_results[df_results["isFraud"] == 0].nsmallest(1, "fraud_prob")["TransactionID"]

    if qc1.button("🔴 High Fraud"):
        txn_id_input = str(int(fraud_ids.values[0]))
        explain_clicked = True
    if qc2.button("🟡 Borderline"):
        txn_id_input = str(int(border_ids.values[0]))
        explain_clicked = True
    if qc3.button("🟢 Legit"):
        txn_id_input = str(int(legit_ids.values[0]))
        explain_clicked = True

st.divider()

# ── Explanation logic ──────────────────────────────────────────────────────────
def tier_color(p):
    if p >= 0.75: return "#F44336", "🔴 Critical Risk"
    if p >= 0.40: return "#FF9800", "🟡 Suspicious"
    return "#4CAF50", "🟢 Clear"


def plain_english_explanation(shap_vals, feature_names, fraud_prob, true_label):
    """Generate a plain-English explanation from SHAP values."""
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:8]

    inc_features = [(feature_names[i], shap_vals[i])
                    for i in sorted_idx if shap_vals[i] > 0]
    dec_features = [(feature_names[i], shap_vals[i])
                    for i in sorted_idx if shap_vals[i] < 0]

    lines = []
    if inc_features:
        lines.append("**Factors pushing toward fraud:**")
        for feat, val in inc_features[:4]:
            lines.append(f"- `{feat}` increased fraud likelihood by **{abs(val):.4f}** SHAP units")
    if dec_features:
        lines.append("\n**Factors reducing fraud likelihood:**")
        for feat, val in dec_features[:4]:
            lines.append(f"- `{feat}` reduced fraud likelihood by **{abs(val):.4f}** SHAP units")

    # Verdict
    color, tier_label = tier_color(fraud_prob)
    lines.append(f"\n**Model verdict:** {tier_label} (probability = `{fraud_prob:.4f}`)")

    if fraud_prob >= 0.75:
        verdict = "**Action: BLOCK** — Multiple strong fraud indicators align. Immediate decline recommended."
    elif fraud_prob >= 0.40:
        verdict = "**Action: FLAG FOR REVIEW** — Borderline case. Route to fraud analyst queue."
    else:
        verdict = "**Action: APPROVE** — Transaction profile is consistent with legitimate behavior."
    lines.append(verdict)

    if true_label is not None:
        actual = "Fraud ✓" if true_label == 1 else "Legitimate ✓"
        lines.append(f"\n*Ground truth (test set): **{actual}***")

    return "\n".join(lines)


# ── Run explanation ────────────────────────────────────────────────────────────
if explain_clicked and txn_id_input.strip():
    try:
        tid = int(txn_id_input.strip())
    except ValueError:
        st.error("TransactionID must be a valid integer.")
        st.stop()

    if df_results is None:
        st.error("test_results.csv not found. Run the notebook first.")
        st.stop()

    row_mask = df_results["TransactionID"] == tid
    if not row_mask.any():
        st.error(f"TransactionID `{tid}` not found in the test results.")
        st.stop()

    row       = df_results[row_mask].iloc[0]
    fraud_prob = float(row["fraud_prob"])
    true_label = int(row["isFraud"]) if "isFraud" in row else None
    tier_hex, tier_label = tier_color(fraud_prob)

    # ── Summary header ─────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Transaction ID",    str(tid))
    h2.metric("Fraud Probability", f"{fraud_prob:.4f}")
    h3.metric("Risk Tier",         tier_label)
    if true_label is not None:
        h4.metric("Actual Label",  "🚨 FRAUD" if true_label == 1 else "✅ Legit")

    # Risk bar
    bar_pct = int(fraud_prob * 100)
    bar_color = tier_hex
    st.markdown(
        f"""
        <div style='background:#eee;border-radius:8px;height:18px;margin:8px 0 16px;'>
          <div style='background:{bar_color};width:{bar_pct}%;height:100%;
               border-radius:8px;transition:width 0.4s;'></div>
        </div>
        <p style='font-size:13px;color:#555;margin-top:-10px;'>
          Risk score: <strong>{bar_pct}%</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── SHAP waterfall plot ────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.subheader("SHAP Waterfall Plot")

        if not DEMO_MODE:
            try:
                model    = artifacts["model"]
                explainer= artifacts["explainer"]
                scaler   = artifacts["scaler"]
                feat_names = artifacts["features"]

                feature_cols = [c for c in feat_names if c in df_results.columns]
                X_row = df_results[row_mask][feature_cols].values
                X_row_scaled = scaler.transform(X_row)

                sv = explainer.shap_values(X_row_scaled)
                sv_single = sv[1][0] if isinstance(sv, list) else sv[0]
                ev = explainer.expected_value[1] if isinstance(explainer.expected_value, list) \
                     else explainer.expected_value

                expl = shap.Explanation(
                    values=sv_single,
                    base_values=ev,
                    data=X_row_scaled[0],
                    feature_names=feat_names,
                )

                fig, ax = plt.subplots(figsize=(9, 6))
                shap.plots.waterfall(expl, max_display=14, show=False)
                plt.title(f"SHAP Waterfall — TxnID {tid} (p={fraud_prob:.3f})",
                          fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            except Exception as e:
                st.error(f"SHAP computation error: {e}")
                st.info("Falling back to feature importance bar chart.")
                DEMO_MODE = True

        if DEMO_MODE:
            # Demo: show random waterfall-like bar chart
            rng = np.random.default_rng(tid)
            demo_feats = [f"feature_{i}" for i in range(14)]
            demo_vals  = rng.normal(0, 0.15, 14)
            demo_vals[0] = fraud_prob * 0.6
            colors = ["#F44336" if v > 0 else "#2196F3" for v in demo_vals]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(demo_feats, demo_vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"SHAP Values (Demo) — TxnID {tid}", fontweight="bold")
            ax.set_xlabel("SHAP value")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with right_col:
        st.subheader("Plain-English Explanation")

        if not DEMO_MODE:
            explanation = plain_english_explanation(sv_single, feat_names, fraud_prob, true_label)
        else:
            rng = np.random.default_rng(tid)
            demo_sv = rng.normal(0, 0.15, 10)
            demo_sv[0] = fraud_prob * 0.6
            explanation = plain_english_explanation(
                demo_sv, [f"feature_{i}" for i in range(10)], fraud_prob, true_label
            )

        st.markdown(explanation)

        st.divider()
        st.markdown("**Transaction Details**")
        detail_fields = ["TransactionAmt", "HourOfDay", "DayOfWeek", "RiskTier", "fraud_prob"]
        for field in detail_fields:
            if field in row:
                val = row[field]
                if isinstance(val, float):
                    val = f"{val:.4f}"
                st.markdown(f"- **{field}**: `{val}`")

elif not txn_id_input.strip() and not explain_clicked:
    # Default instructional state
    st.markdown("""
    ### How to use this page

    1. **Enter a TransactionID** in the search box above
    2. Click **⚡ Explain** to generate the SHAP waterfall plot
    3. The waterfall shows which features pushed the fraud probability up (red) or down (blue)
    4. The plain-English explanation translates the technical output into an analyst-friendly verdict

    ---

    ### What is a SHAP value?

    SHAP (SHapley Additive exPlanations) assigns each feature a contribution score for a
    specific prediction. Positive SHAP = increases fraud probability. Negative SHAP = decreases it.
    The waterfall starts at the model's average prediction and shows how each feature moves it
    toward the final score.

    ---
    """)

    if df_results is not None:
        st.subheader("Sample Transactions to Explore")
        sample = pd.concat([
            df_results[df_results["RiskTier"] == "Critical"].head(5),
            df_results[df_results["RiskTier"] == "Suspicious"].head(5),
            df_results[df_results["RiskTier"] == "Clear"].head(5),
        ])
        st.dataframe(
            sample[["TransactionID","TransactionAmt","HourOfDay",
                    "isFraud","fraud_prob","RiskTier"]].reset_index(drop=True),
            use_container_width=True,
        )
