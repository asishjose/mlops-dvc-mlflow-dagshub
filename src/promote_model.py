"""
promote_model.py
Promotes the latest model version in MLflow registry
from None → Staging → Production.

Usage:
    python src/promote_model.py --stage Staging
    python src/promote_model.py --stage Production
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "https://dagshub.com/asishjose/mlops-dvc-mlflow-dagshub.mlflow"
MODEL_NAME   = "buysignal-lr-model"


def promote(stage: str):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # get the latest version
    versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        print(f"No registered versions found for '{MODEL_NAME}'.")
        print("Run the pipeline first: dvc repro")
        return

    latest = max(versions, key=lambda v: int(v.version))
    print(f"Promoting version {latest.version} → {stage}")

    client.transition_model_version_stage(
        name    = MODEL_NAME,
        version = latest.version,
        stage   = stage,
    )
    print(f"Done. Model version {latest.version} is now in {stage}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["Staging", "Production"], required=True)
    promote(parser.parse_args().stage)