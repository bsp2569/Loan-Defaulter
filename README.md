# Loan Default Prediction with Explainable AI (SHAP vs. LIME)

## Overview
End-to-end loan default classification pipeline benchmarking two model
explainability methods — SHAP and LIME — against each other, to test how
reliable "black box" explanations are for high-stakes credit decisions.

## Dataset
LendingClub loan data (2007–2010), 9,578 loans, 14 features.
Target: `not.fully.paid` (1 = borrower did not fully repay). ~16% default rate.
Source: public LendingClub dataset, widely used in credit risk case studies
and ML tutorials (e.g. the "Decision Trees and Random Forest" case study).

**Download URL:**
`https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/loan_data.csv`

## Pipeline
1. **`01_train_model.py`** — Trains XGBoost and Random Forest classifiers on
   an 80/20 train/test split, handling class imbalance via `scale_pos_weight`
   / `class_weight="balanced"`. Saves both models and evaluation metrics.
2. **`02_explain_compare.py`** — For 30 test cases (15 maximally-borderline
   predictions near proba=0.5, 15 high-confidence predictions), generates:
   - SHAP explanations via `TreeExplainer` (exact for tree-based models)
   - LIME explanations via `LimeTabularExplainer`
   - Top-5 contributing feature sets from each method
   - **Jaccard similarity** between the two top-5 feature sets, as a
     quantitative measure of explanation agreement

## Results

| Metric | XGBoost | Random Forest |
|---|---|---|
| Test AUC | 0.655 | 0.660 |

Model performance is modest — expected and honest for this dataset; the
signal-to-noise ratio in unsecured consumer lending data is genuinely low,
which is itself part of the point: **when a model's confidence is low, do its
explanations even agree with themselves across methods?**

### Explanation agreement (SHAP vs. LIME, top-5 features, Jaccard similarity)

| Segment | Mean Jaccard |
|---|---|
| All 30 cases | 0.529 |
| Borderline predictions (proba ≈ 0.50) | 0.411 |
| High-confidence predictions | 0.648 |

**Key finding:** the most borderline prediction in the sample (proba = 0.500)
had a Jaccard similarity of just **0.111** between SHAP's and LIME's top-5
features — SHAP and LIME essentially disagreed on *why* the model made that
call. SHAP pointed to `credit.policy`, `days.with.cr.line`, and `fico`; LIME
pointed to `int.rate`, `installment`, and `revol.util`. Only `days.with.cr.line`
appeared in both.

This pattern holds directionally across the sample: **borderline predictions
show meaningfully lower explanation agreement than confident ones**
(0.41 vs. 0.65 mean Jaccard). That's a concrete, reproducible caveat for any
team relying on a single explainability method to justify a lending decision:
explanations are least trustworthy exactly where the model itself is least
certain.

## Files
- `data/loan_data.csv` — raw dataset
- `data/shap_lime_comparison.csv` — full per-case comparison results
- `data/comparison_summary.json` — aggregate statistics
- `models/*.pkl` — trained models
- `01_train_model.py`, `02_explain_compare.py` — pipeline scripts

## Tech Stack
Python, scikit-learn, XGBoost, SHAP, LIME, pandas, Streamlit, Plotly

## Running Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploying to Streamlit Community Cloud (free)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", select the repo, branch, and set the main file path to
   `streamlit_app.py`.
4. Deploy. You'll get a live URL like `your-app-name.streamlit.app`.

Note: `models/rf_model.pkl` (Random Forest) is excluded from this package to
keep it small — the app only uses the XGBoost model. If you want to include
the Random Forest model too, retrain it locally with
`python 01_train_model.py` before deploying.
