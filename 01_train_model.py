"""
Loan Default Prediction - Model Training
Dataset: LendingClub loan data (2007-2010), 9,578 loans, 14 features
Target: not.fully.paid (1 = borrower did not fully repay)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import json

RANDOM_STATE = 42

def load_and_prepare_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)

    # Encode categorical purpose column
    le = LabelEncoder()
    df["purpose_encoded"] = le.fit_transform(df["purpose"])

    feature_cols = [
        "credit.policy", "purpose_encoded", "int.rate", "installment",
        "log.annual.inc", "dti", "fico", "days.with.cr.line",
        "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec"
    ]
    X = df[feature_cols]
    y = df["not.fully.paid"]

    return X, y, feature_cols, le, df


def train_models(X_train, y_train):
    # XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    # Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)

    return xgb_model, rf_model


def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Fully Paid", "Not Fully Paid"]))
    return auc


if __name__ == "__main__":
    X, y, feature_cols, le, df = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Default rate (train): {y_train.mean():.3f}")

    xgb_model, rf_model = train_models(X_train, y_train)

    xgb_auc = evaluate(xgb_model, X_test, y_test, "XGBoost")
    rf_auc = evaluate(rf_model, X_test, y_test, "Random Forest")

    # Save the better-performing model as primary, both for reference
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(le, "models/purpose_encoder.pkl")
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    X_train.to_csv("data/X_train.csv", index=False)

    metrics = {
        "xgb_auc": float(xgb_auc),
        "rf_auc": float(rf_auc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate": float(y.mean()),
        "feature_cols": feature_cols
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModels and data saved.")
