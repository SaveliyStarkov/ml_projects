import joblib
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
from db import load_training_data, save_training_run

MODEL_PATH = "model.pkl"
MODEL_VERSION = "v2"

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "total_late_payments",
    "DebtRatio",
    "age",
    "NumberOfOpenCreditLinesAndLoans",
    "MonthlyIncome",
    "income_per_dependent",
    "NumberOfTimes90DaysLate"
]


def train_model() -> dict:
    df = load_training_data()
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = round(roc_auc_score(y_test, y_prob), 4)
    pr_auc  = round(average_precision_score(y_test, y_prob), 4)

    joblib.dump(model, MODEL_PATH)
    save_training_run(
        model_version=MODEL_VERSION,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        train_rows=len(X_train),
        status="success"
    )

    return {
        "model_version": MODEL_VERSION,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }