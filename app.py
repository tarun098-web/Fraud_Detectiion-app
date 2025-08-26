# app.py
# Minimal Streamlit dashboard with 3 dropdowns (models, fairness metrics, performance metrics)
# Bars only; supports multiple fairness metrics; fixes MACE fairness plots.

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Detection Results", page_icon="📊", layout="wide")
st.title("📊 Fairness‑Aware Fraud Detection — Results")

# =========================
# Data (edit here if needed)
# =========================

# Final (Non‑SMOTE / Weighted) performance
performance = pd.DataFrame(
    [
        ["Random Forest",        0.3983, 0.7650, 0.5239, 0.9815],
        ["XGBoost",              0.2551, 0.7506, 0.3807, 0.9654],
        ["Logistic Regression",  0.0144, 0.7506, 0.0283, 0.8783],
    ],
    columns=["Model", "Precision", "Recall", "F1", "ROC-AUC"]
)

# Fairness metrics (tidy format)
fairness_rows = [
    # metric, subgroup, model, value
    # SPD
    ("SPD", "Gender", "Random Forest",       -0.001858),
    ("SPD", "Gender", "XGBoost",             -0.002187),
    ("SPD", "Gender", "Logistic Regression", -0.054855),
    ("SPD", "Age",    "Random Forest",        0.000589),
    ("SPD", "Age",    "XGBoost",              0.000917),
    ("SPD", "Age",    "Logistic Regression",  0.021519),

    # DPD
    ("DPD", "Gender", "Random Forest",       -0.001858),
    ("DPD", "Gender", "XGBoost",             -0.002187),
    ("DPD", "Gender", "Logistic Regression", -0.054855),
    ("DPD", "Age",    "Random Forest",        0.002859),
    ("DPD", "Age",    "XGBoost",              0.007427),
    ("DPD", "Age",    "Logistic Regression", -0.031194),

    # EOD
    ("EOD", "Gender", "Random Forest",        0.195487),
    ("EOD", "Gender", "XGBoost",              0.192480),
    ("EOD", "Gender", "Logistic Regression",  0.150998),
    ("EOD", "Age",    "Random Forest",        0.002859),
    ("EOD", "Age",    "XGBoost",              0.007427),
    ("EOD", "Age",    "Logistic Regression", -0.031194),

    # Equalized Odds
    ("Equalized Odds", "Gender", "Random Forest",        0.169900),
    ("Equalized Odds", "Gender", "XGBoost",              0.098650),
    ("Equalized Odds", "Gender", "Logistic Regression",  0.007566),
    ("Equalized Odds", "Age",    "Random Forest",        0.02334),
    ("Equalized Odds", "Age",    "XGBoost",              0.02047),
    ("Equalized Odds", "Age",    "Logistic Regression",  0.00097),

    # MACE (overall) — ensure these show in fairness charts
    ("MACE (overall)", "Gender", "Random Forest",        0.1080),
    ("MACE (overall)", "Gender", "XGBoost",              0.149753),
    ("MACE (overall)", "Gender", "Logistic Regression",  0.494949),
    ("MACE (overall)", "Age",    "Random Forest",        0.191905),
    ("MACE (overall)", "Age",    "XGBoost",              0.167727),
    ("MACE (overall)", "Age",    "Logistic Regression",  0.740050),
]
fairness = pd.DataFrame(fairness_rows, columns=["Metric", "Subgroup", "Model", "Value"])

# =========================
# 3 Dropdowns
# =========================

models_available = performance["Model"].tolist()
perf_metrics_available = ["Precision", "Recall", "F1", "ROC-AUC"]

# Build fairness option labels as a list of (label, metric, subgroup) so parsing is robust
fairness_options_map = []
for metric in fairness["Metric"].unique():
    for subgroup in ["Gender", "Age"]:
        # include option only if we actually have rows
        if not fairness[(fairness["Metric"] == metric) & (fairness["Subgroup"] == subgroup)].empty:
            label = f"{metric} — {subgroup}"
            fairness_options_map.append((label, metric, subgroup))

with st.container():
    c1, c2, c3 = st.columns(3)

    # 1) Models (multi-select)
    sel_models = c1.multiselect("Select model(s)", models_available, default=models_available)

    # 2) Fairness metrics (multi-select now)
    fairness_labels = [lbl for (lbl, _, _) in fairness_options_map]
    sel_fairness_labels = c2.multiselect(
        "Select fairness metric(s)",
        fairness_labels,
        default=["SPD — Gender", "DPD — Gender", "EOD — Gender", "Equalized Odds — Gender", "MACE (overall) — Gender"]
    )

    # 3) Performance metrics (multi-select)
    sel_perf_metrics = c3.multiselect(
        "Select performance metrics",
        perf_metrics_available,
        default=["Precision", "Recall", "F1", "ROC-AUC"]
    )

st.markdown("---")

# =========================
# Performance (Bars only)
# =========================
st.subheader("Performance")

perf_df = performance[performance["Model"].isin(sel_models)].copy()

if len(sel_perf_metrics) == 0:
    st.info("Select at least one performance metric from the third dropdown.")
else:
    perf_long = perf_df.melt(id_vars="Model", value_vars=sel_perf_metrics,
                             var_name="Metric", value_name="Score")
    fig_bar = px.bar(
        perf_long, x="Model", y="Score", color="Metric", barmode="group",
        text="Score", height=420
    )
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bar.update_layout(yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# Fairness (Multi-metric; one chart per selection)
# =========================
st.subheader("Fairness")

if len(sel_fairness_labels) == 0:
    st.info("Select at least one fairness metric from the second dropdown.")
else:
    # create cols for a simple responsive grid: 2 charts per row
    for i, lbl in enumerate(sel_fairness_labels):
        metric = next(m for (L, m, s) in fairness_options_map if L == lbl)
        subgroup = next(s for (L, m, s) in fairness_options_map if L == lbl)

        sub_df = fairness[
            (fairness["Metric"] == metric) &
            (fairness["Subgroup"] == subgroup) &
            (fairness["Model"].isin(sel_models))
        ].copy()

        if i % 2 == 0:
            cols = st.columns(2)
        col = cols[i % 2]

        if sub_df.empty:
            col.info(f"No data for {lbl}.")
        else:
            sub_df = sub_df.sort_values("Model")
            title = f"{metric} ({subgroup})"
            fig = px.bar(sub_df, x="Model", y="Value", text="Value", title=title, height=380)
            # Show 4 decimal places for fairness, 3 for MACE depending on range
            if "MACE" in metric:
                fig.update_traces(texttemplate="%{text:.3f}")
            else:
                fig.update_traces(texttemplate="%{text:.4f}")
            fig.update_traces(textposition="outside")
            col.plotly_chart(fig, use_container_width=True)

# =========================
# Footer notes
# =========================
st.caption("""
**Notes**
- Performance metrics reflect final Non‑SMOTE (custom weighting) models.
- You can select multiple fairness metrics at once (e.g., SPD, DPD, EOD, Equalized Odds, and MACE for Gender/Age).
- For fairness: values near 0 are desirable for SPD/DPD/EOD/Equalized Odds; **lower** is better for MACE (overall).
""")
