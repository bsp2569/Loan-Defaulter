"""
Loan Default Prediction - SHAP vs LIME Explainability Comparison
Benchmarks explanation agreement between SHAP and LIME on individual predictions
using Jaccard similarity of top contributing features.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import json
from pathlib import Path

RANDOM_STATE = 42
TOP_K = 5  # number of top features to compare


def jaccard_similarity(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def get_shap_top_features(shap_values_row, feature_names, k=TOP_K):
    abs_vals = np.abs(shap_values_row)
    top_idx = np.argsort(abs_vals)[::-1][:k]
    return [feature_names[i] for i in top_idx]


def get_lime_top_features(lime_exp, k=TOP_K):
    # lime_exp.as_list() returns [(feature_description, weight), ...]
    sorted_feats = sorted(lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True)
    top_feats = [f[0] for f in sorted_feats[:k]]
    # LIME feature descriptions include conditions (e.g. "fico <= 700") - extract base feature name
    cleaned = []
    for f in top_feats:
        for name in feature_names_global:
            if name in f:
                cleaned.append(name)
                break
    return cleaned


if __name__ == "__main__":
    model = joblib.load("models/xgb_model.pkl")
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    feature_names_global = X_train.columns.tolist()

    # --- SHAP explainer (TreeExplainer, exact for tree models) ---
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X_test)

    # --- LIME explainer ---
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names_global,
        class_names=["Fully Paid", "Not Fully Paid"],
        mode="classification",
        random_state=RANDOM_STATE
    )

    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names_global))

    # Sample a set of test cases to compare (mix of confident and borderline predictions)
    proba = model.predict_proba(X_test)[:, 1]
    X_test_reset = X_test.reset_index(drop=True)

    # Select: 10 confident-correct, 10 borderline (proba near 0.5)
    borderline_idx = np.argsort(np.abs(proba - 0.5))[:15]
    confident_idx = np.argsort(np.abs(proba - 0.5))[-15:]
    sample_idx = np.concatenate([borderline_idx, confident_idx])

    results = []
    for idx in sample_idx:
        row = X_test_reset.iloc[[idx]]
        shap_row = shap_values[idx]
        shap_top = get_shap_top_features(shap_row, feature_names_global, TOP_K)

        lime_exp = lime_explainer.explain_instance(
            row.values[0], predict_fn, num_features=TOP_K, labels=(1,)
        )
        lime_top = get_lime_top_features(lime_exp, TOP_K)

        jaccard = jaccard_similarity(shap_top, lime_top)

        results.append({
            "test_idx": int(idx),
            "predicted_proba": float(proba[idx]),
            "is_borderline": bool(idx in borderline_idx),
            "shap_top_features": shap_top,
            "lime_top_features": lime_top,
            "jaccard_similarity": float(jaccard)
        })
        print(f"idx={idx:4d}  proba={proba[idx]:.3f}  jaccard={jaccard:.3f}  "
              f"{'[BORDERLINE]' if idx in borderline_idx else '[CONFIDENT]'}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/shap_lime_comparison.csv", index=False)

    print("\n=== Summary ===")
    print(f"Overall mean Jaccard similarity: {results_df['jaccard_similarity'].mean():.3f}")
    print(f"Borderline cases mean Jaccard:   {results_df[results_df['is_borderline']]['jaccard_similarity'].mean():.3f}")
    print(f"Confident cases mean Jaccard:    {results_df[~results_df['is_borderline']]['jaccard_similarity'].mean():.3f}")

    min_row = results_df.loc[results_df["jaccard_similarity"].idxmin()]
    print(f"\nLowest agreement case: idx={min_row['test_idx']}, "
          f"proba={min_row['predicted_proba']:.3f}, jaccard={min_row['jaccard_similarity']:.3f}")
    print(f"  SHAP top features: {min_row['shap_top_features']}")
    print(f"  LIME top features: {min_row['lime_top_features']}")

    summary = {
        "overall_mean_jaccard": float(results_df["jaccard_similarity"].mean()),
        "borderline_mean_jaccard": float(results_df[results_df["is_borderline"]]["jaccard_similarity"].mean()),
        "confident_mean_jaccard": float(results_df[~results_df["is_borderline"]]["jaccard_similarity"].mean()),
        "min_jaccard_case": {
            "test_idx": int(min_row["test_idx"]),
            "predicted_proba": float(min_row["predicted_proba"]),
            "jaccard_similarity": float(min_row["jaccard_similarity"]),
            "shap_top_features": min_row["shap_top_features"],
            "lime_top_features": min_row["lime_top_features"]
        }
    }
    with open("data/comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved: data/shap_lime_comparison.csv, data/comparison_summary.json")
