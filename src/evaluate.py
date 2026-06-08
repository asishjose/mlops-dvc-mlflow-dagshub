"""
evaluate.py
Loads the DVC-tracked model binary and runs evaluation.
Logs results to MLflow under the same experiment for comparison.
Used as a separate DVC stage so evaluation is always reproducible
from the exact model artifact — not an in-memory object.
"""

import json
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report,
)

FEATURES = ["session_duration", "pages_viewed", "product_clicks",
            "cart_additions", "search_queries", "returning_user",
            "device_type", "time_of_day"]
TARGET   = "purchased"


def main():
    # load DVC-tracked model binary
    with open("models/lr_model.pkl", "rb") as f:
        model = pickle.load(f)

    test = pd.read_csv("data/processed/test.csv")
    X_test, y_test = test[FEATURES], test[TARGET]

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "eval_auc":       round(roc_auc_score(y_test, y_proba), 4),
        "eval_f1":        round(f1_score(y_test, y_pred), 4),
        "eval_recall":    round(recall_score(y_test, y_pred), 4),
        "eval_precision": round(precision_score(y_test, y_pred), 4),
    }

    cm = confusion_matrix(y_test, y_pred)

    print("\n── Evaluation Results ──────────────────────────")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", cm)
    print("Metrics:", metrics)

    # log to MLflow evaluation run
    mlflow.set_tracking_uri("https://dagshub.com/asishjose/mlops-dvc-mlflow-dagshub.mlflow")
    mlflow.set_experiment("buysignal-purchase-intent")
    with open("metrics/run_id.txt") as f:
        train_run_id = f.read().strip()
    with mlflow.start_run(run_id=train_run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("stage", "evaluation")

        # log confusion matrix as artifact
        cm_path = Path("metrics/confusion_matrix.json")
        cm_path.parent.mkdir(exist_ok=True)
        with open(cm_path, "w") as f:
            json.dump(cm.tolist(), f)
        mlflow.log_artifact(str(cm_path))

    print("\nEvaluation logged to MLflow.")


if __name__ == "__main__":
    main()
