"""
preprocess.py
Reads data/raw/clickstream.csv → data/processed/{train,test,features}.csv
DVC tracks inputs and outputs. No MLflow here — this is pure data transformation.
"""

import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PARAMS  = yaml.safe_load(open("params.yaml"))["preprocess"]
RAW     = Path("data/raw/clickstream.csv")
OUT_DIR = Path("data/processed")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # encode categoricals
    le_device  = LabelEncoder()
    le_time    = LabelEncoder()
    df = df.copy()
    df["device_type"] = le_device.fit_transform(df["device_type"])
    df["time_of_day"] = le_time.fit_transform(df["time_of_day"])
    return df


def main():
    df = pd.read_csv(RAW)
    print(f"Loaded {len(df)} rows from {RAW}")

    df = preprocess(df)

    train, test = train_test_split(
        df,
        test_size=PARAMS["test_size"],
        random_state=PARAMS["random_state"],
        stratify=df["purchased"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR  / "test.csv",  index=False)
    df.to_csv(OUT_DIR    / "features.csv", index=False)

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
