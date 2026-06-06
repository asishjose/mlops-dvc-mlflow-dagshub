# BuySignal MLOps — DVC + MLflow Integration

Purchase intent prediction pipeline demonstrating a production MLOps workflow
with full data versioning, experiment tracking, and model registry.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Git Commit                              │
│                                                                 │
│          dvc.lock ──────────────────── mlflow run_id            │
│             │                                │                  │
│             ↓                                ↓                  │
│     exact data hashes               metrics + params            │
│     exact model hash                experiment UI               │
│     pipeline that ran               model registry              │
└─────────────────────────────────────────────────────────────────┘
           DVC answers                    MLflow answers
       "what was the state?"         "what were the results?"
```

## Tool Responsibilities

| Concern                  | Tool    | Where                        |
|--------------------------|---------|------------------------------|
| Raw data version         | DVC     | `dvc.lock` hash              |
| Processed features       | DVC     | `dvc.lock` hash              |
| Model binary (.pkl)      | DVC     | `dvc.lock` hash              |
| Pipeline DAG             | DVC     | `dvc.yaml`                   |
| Hyperparameters          | Both    | `params.yaml` → MLflow log   |
| Metrics (AUC, F1, recall)| MLflow  | Experiment UI                |
| Experiment comparison    | MLflow  | `mlflow ui`                  |
| Model registry           | MLflow  | Staging → Production         |
| Rollback                 | Git+DVC | `git checkout` + `dvc checkout` |

---

## Project Structure

```
buysignal-mlops/
├── data/
│   ├── raw/
│   │   └── clickstream.csv        ← DVC tracked
│   └── processed/
│       ├── train.csv              ← DVC tracked
│       ├── test.csv               ← DVC tracked
│       └── features.csv           ← DVC tracked
├── models/
│   └── lr_model.pkl               ← DVC tracked
├── metrics/
│   ├── scores.json                ← DVC metrics (no cache)
│   └── confusion_matrix.json
├── src/
│   ├── generate_data.py           ← one-time data seed
│   ├── preprocess.py              ← DVC stage
│   ├── train.py                   ← DVC stage + MLflow logging
│   └── evaluate.py                ← DVC stage + MLflow logging
├── dvc.yaml                       ← pipeline DAG
├── dvc.lock                       ← exact hashes (commit this)
├── params.yaml                    ← single source of truth
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Git and DVC
```bash
git init
dvc init
git add .
git commit -m "chore: project scaffold"
```

### 3. (Optional) Add DVC remote for team sharing
```bash
# Local remote
dvc remote add -d local_remote /tmp/dvc-storage

# Or MinIO (S3-compatible, runs locally via Docker)
# docker run -p 9001:9000 minio/minio server /data
# dvc remote add -d minio s3://mybucket/dvc-cache
# dvc remote modify minio endpointurl http://localhost:9001
```

### 4. Generate raw data
```bash
python src/generate_data.py
dvc add data/raw/clickstream.csv
git add data/raw/clickstream.csv.dvc .gitignore
git commit -m "data: add raw clickstream v1 (5000 rows)"
```

### 5. Run the full pipeline
```bash
dvc repro
```

DVC will:
- Skip stages whose inputs haven't changed (content hash comparison)
- Re-run only what changed
- Record exact hashes in `dvc.lock`

### 6. Commit the pipeline state
```bash
dvc push                                    # push data to remote
git add dvc.lock params.yaml metrics/
git commit -m "run: baseline LR C=1.0 AUC=0.87"
git tag baseline-v1
```

### 7. View MLflow experiments
```bash
mlflow ui
# open http://localhost:5000
```

---

## Experiment Workflow — Changing Hyperparameters

```bash
# edit params.yaml — change C: 1.0 → C: 0.1
dvc repro                  # only train + evaluate re-run (preprocess skipped)
git add dvc.lock params.yaml metrics/
git commit -m "run: C=0.1 AUC=0.84"
```

Compare runs in MLflow UI or via CLI:
```bash
dvc metrics diff HEAD~1    # diff metrics between last two commits
```

---

## Rollback to Any Prior Run

```bash
git checkout baseline-v1   # restore params, dvc.lock, metrics
dvc checkout               # restore exact data + model binary from cache
```

This is the key capability — any git tag gives you back the **exact data,
exact model, and exact metrics** from that run. No manual version strings needed.

---

## Why Not HDFS for DVC?

This project uses local/S3 storage intentionally. DVC's caching relies on:
- Hardlinks/symlinks (not available on HDFS)
- Free file movement during checkout (HDFS is write-once)
- Spark-readable output paths (DVC cache uses hash-addressed paths)

For HDFS-scale pipelines, Delta Lake or Apache Iceberg handle data versioning
at the storage format layer — where Spark can read versioned data natively
without a copy step.

---

## Key Concepts Demonstrated

1. **DVC as pipeline orchestrator** — `dvc repro` runs only what changed
2. **DVC as data versioner** — `dvc.lock` records exact content hashes
3. **MLflow as experiment tracker** — compare runs across hyperparameters
4. **Git as the binding layer** — every commit ties data state to experiment results
5. **params.yaml as single source of truth** — DVC detects changes, MLflow logs values
6. **Rollback** — `git checkout <tag>` + `dvc checkout` = full state restore
