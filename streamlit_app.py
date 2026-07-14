"""
Loan Default Prediction with Explainable AI (SHAP vs LIME)
Interactive Streamlit dashboard for exploring model predictions and
comparing explanation agreement between SHAP and LIME.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="Loan Default XAI",
    page_icon="\U0001F4B3",
    layout="wide"
)

RANDOM_STATE = 42
TOP_K = 5

FEATURE_LABELS = {
    "credit.policy": "Meets Credit Policy",
    "purpose_encoded": "Loan Purpose",
    "int.rate": "Interest Rate",
    "installment": "Monthly Installment",
    "log.annual.inc": "Log Annual Income",
    "dti": "Debt-to-Income Ratio",
    "fico": "FICO Score",
    "days.with.cr.line": "Days With Credit Line",
    "revol.bal": "Revolving Balance",
    "revol.util": "Revolving Utilization %",
    "inq.last.6mths": "Inquiries (Last 6mo)",
    "delinq.2yrs": "Delinquencies (2yr)",
    "pub.rec": "Public Records"
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    le = joblib.load("models/purpose_encoder.pkl")
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    with open("data/comparison_summary.json") as f:
        comparison_summary = json.load(f)
    comparison_df = pd.read_csv("data/shap_lime_comparison.csv")
    return model, le, X_train, X_test, y_test, metrics, comparison_summary, comparison_df


@st.cache_resource
def build_explainers(_model, _X_train):
    shap_explainer = shap.TreeExplainer(_model)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=_X_train.values,
        feature_names=_X_train.columns.tolist(),
        class_names=["Fully Paid", "Not Fully Paid"],
        mode="classification",
        random_state=RANDOM_STATE
    )
    return shap_explainer, lime_explainer


def jaccard_similarity(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def get_shap_top(shap_row, feature_names, k=TOP_K):
    abs_vals = np.abs(shap_row)
    idx = np.argsort(abs_vals)[::-1][:k]
    return [(feature_names[i], shap_row[i]) for i in idx]


def get_lime_top(lime_exp, feature_names, k=TOP_K):
    sorted_feats = sorted(lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True)[:k]
    cleaned = []
    for desc, weight in sorted_feats:
        matched = next((n for n in feature_names if n in desc), desc)
        cleaned.append((matched, weight, desc))
    return cleaned


def make_bar_chart(items, title, color):
    names = [FEATURE_LABELS.get(n[0], n[0]) for n in items][::-1]
    vals = [n[1] for n in items][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=[color if v >= 0 else "#d62728" for v in vals]
    ))
    fig.update_layout(
        title=title, height=280, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Contribution to prediction"
    )
    return fig


model, le, X_train, X_test, y_test, metrics, comparison_summary, comparison_df = load_artifacts()
shap_explainer, lime_explainer = build_explainers(model, X_train)
feature_names = X_train.columns.tolist()

st.title("\U0001F4B3 Loan Default Prediction with Explainable AI")
st.caption("Benchmarking SHAP vs. LIME agreement on individual loan default predictions")

tab1, tab2, tab3 = st.tabs(["Explore a Prediction", "SHAP vs LIME Benchmark", "About This Project"])


with tab1:
    st.subheader("Pick a loan from the test set")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        idx = st.number_input(
            "Test set row index", min_value=0, max_value=len(X_test) - 1,
            value=int(comparison_summary["min_jaccard_case"]["test_idx"]), step=1
        )
        row = X_test.iloc[[idx]]
        proba = model.predict_proba(row)[0, 1]
        actual = y_test.iloc[idx]

        st.metric("Predicted default probability", f"{proba:.1%}")
        st.metric("Model prediction", "Not Fully Paid" if proba >= 0.5 else "Fully Paid")
        st.metric("Actual outcome", "Not Fully Paid" if actual == 1 else "Fully Paid")

        st.markdown("**Loan details**")
        display_row = row.iloc[0].rename(index=FEATURE_LABELS)
        st.dataframe(display_row.to_frame("Value"), use_container_width=True)

    with col_b:
        shap_vals = shap_explainer.shap_values(row)[0]
        shap_top = get_shap_top(shap_vals, feature_names)

        lime_exp = lime_explainer.explain_instance(
            row.values[0],
            lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names)),
            num_features=TOP_K, labels=(1,)
        )
        lime_top_raw = get_lime_top(lime_exp, feature_names)
        lime_top = [(n, w) for n, w, _ in lime_top_raw]

        jaccard = jaccard_similarity(
            [n for n, _ in shap_top], [n for n, _ in lime_top]
        )

        st.plotly_chart(make_bar_chart(shap_top, "SHAP: Top 5 Contributing Features", "#1f77b4"),
                         use_container_width=True)
        st.plotly_chart(make_bar_chart(lime_top, "LIME: Top 5 Contributing Features", "#2ca02c"),
                         use_container_width=True)

        if jaccard < 0.3:
            st.error(f"**Jaccard similarity: {jaccard:.3f}** — SHAP and LIME largely disagree "
                     f"on why this prediction was made. Low agreement.")
        elif jaccard < 0.6:
            st.warning(f"**Jaccard similarity: {jaccard:.3f}** — partial agreement between "
                       f"SHAP and LIME on top features.")
        else:
            st.success(f"**Jaccard similarity: {jaccard:.3f}** — SHAP and LIME largely agree "
                       f"on top contributing features.")


with tab2:
    st.subheader("Explanation agreement across 30 sampled test cases")
    st.write(
        "Each case below compares the top-5 SHAP and LIME features for a single prediction, "
        "measuring overlap with **Jaccard similarity** (0 = no overlap, 1 = identical feature sets). "
        "Cases are split between maximally borderline predictions (probability \u2248 0.50) and "
        "high-confidence predictions."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall mean Jaccard", f"{comparison_summary['overall_mean_jaccard']:.3f}")
    c2.metric("Borderline cases mean Jaccard", f"{comparison_summary['borderline_mean_jaccard']:.3f}")
    c3.metric("Confident cases mean Jaccard", f"{comparison_summary['confident_mean_jaccard']:.3f}")

    st.markdown(
        f"**Lowest-agreement case:** test index "
        f"`{comparison_summary['min_jaccard_case']['test_idx']}`, predicted probability "
        f"`{comparison_summary['min_jaccard_case']['predicted_proba']:.3f}` "
        f"(maximally borderline) had a Jaccard similarity of just "
        f"**{comparison_summary['min_jaccard_case']['jaccard_similarity']:.3f}** — "
        f"SHAP and LIME nearly completely disagreed on which features drove this prediction."
    )

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=comparison_df[comparison_df["is_borderline"]]["jaccard_similarity"],
        name="Borderline (proba \u2248 0.5)", marker_color="#d62728"
    ))
    fig.add_trace(go.Box(
        y=comparison_df[~comparison_df["is_borderline"]]["jaccard_similarity"],
        name="Confident predictions", marker_color="#1f77b4"
    ))
    fig.update_layout(
        title="Jaccard Similarity Distribution: Borderline vs. Confident Predictions",
        yaxis_title="Jaccard similarity (top-5 features)", height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full comparison table**")
    st.dataframe(
        comparison_df[["test_idx", "predicted_proba", "is_borderline", "jaccard_similarity"]]
        .sort_values("jaccard_similarity"),
        use_container_width=True, hide_index=True
    )

