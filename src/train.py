"""
train.py
Trains a Logistic Regression on processed features.

DVC  → tracks data/processed/ as input, models/lr_model.pkl as output
MLflow → logs params, metrics, model artifact, and git commit tag
"""

import os
import json
import pickle
import subprocess
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, classification_report,
)

PARAMS      = yaml.safe_load(open("params.yaml"))
TRAIN_PARAMS = PARAMS["train"]
FEATURES    = ["session_duration", "pages_viewed", "product_clicks",
               "cart_additions", "search_queries", "returning_user",
               "device_type", "time_of_day"]
TARGET      = "purchased"
MODEL_PATH  = Path("models/lr_model.pkl")
METRICS_PATH = Path("metrics/scores.json")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test  = pd.read_csv("data/processed/test.csv")
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]
    return X_train, y_train, X_test, y_test


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "auc":       round(roc_auc_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
    }


def main():
    X_train, y_train, X_test, y_test = load_data()

    mlflow.set_tracking_uri("https://dagshub.com/asishjose/mlops-dvc-mlflow-dagshub.mlflow")
    mlflow.set_experiment("buysignal-purchase-intent")

    with mlflow.start_run(run_name=f"lr-C{TRAIN_PARAMS['C']}"):

        # ── log hyperparams ────────────────────────────────────────────
        mlflow.log_params(TRAIN_PARAMS)
        mlflow.log_params(PARAMS["preprocess"])   # full reproducibility

        # ── train ──────────────────────────────────────────────────────
        model = LogisticRegression(
            C=TRAIN_PARAMS["C"],
            solver=TRAIN_PARAMS["solver"],
            max_iter=TRAIN_PARAMS["max_iter"],
            class_weight="balanced",              # handles class imbalance
            random_state=42,
        )
        model.fit(X_train, y_train)

        # ── metrics ────────────────────────────────────────────────────
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        print("\nMetrics:", metrics)
        print(classification_report(y_test, model.predict(X_test)))

        # ── log model to MLflow ────────────────────────────────────────
        mlflow.sklearn.log_model(
            model,
            artifact_path="lr_model",
            registered_model_name="buysignal-lr-model",
        )
        
        # ── tie MLflow run to exact Git + DVC state ────────────────────
        git_commit = get_git_commit()
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_params_hash", str(hash(str(PARAMS))))

        # ── save model binary for DVC to track ────────────────────────
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved → {MODEL_PATH}")

        # ── save metrics for DVC metrics diff ─────────────────────────
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        run_id = mlflow.active_run().info.run_id
        with open("metrics/run_id.txt", "w") as f:
            f.write(run_id)
        print(f"\nMLflow run_id: {run_id}")
        print(f"Git commit:    {git_commit}")


if __name__ == "__main__":
    main()
