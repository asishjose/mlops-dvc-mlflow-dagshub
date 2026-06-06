"""
generate_data.py
Generates a synthetic e-commerce clickstream CSV for BuySignal-MLOps.
Run once to seed data/raw/clickstream.csv — DVC then tracks it.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 5000
OUT = Path("data/raw/clickstream.csv")


def main():
    rng = np.random.default_rng(SEED)

    # --- features -------------------------------------------------------
    session_duration   = rng.exponential(scale=300, size=N).clip(10, 3600)
    pages_viewed       = rng.integers(1, 30, size=N)
    product_clicks     = rng.integers(0, 15, size=N)
    cart_additions     = rng.integers(0, 5,  size=N)
    search_queries     = rng.integers(0, 10, size=N)
    returning_user     = rng.integers(0, 2,  size=N)
    device_type        = rng.choice(["mobile", "desktop", "tablet"], size=N,
                                    p=[0.55, 0.35, 0.10])
    time_of_day        = rng.choice(["morning", "afternoon", "evening", "night"],
                                    size=N, p=[0.20, 0.35, 0.30, 0.15])

    # --- label (purchase intent) ----------------------------------------
    # weighted score → sigmoid → binary label  (≈15% positive rate)
    score = (
        0.003 * session_duration
        + 0.05  * pages_viewed
        + 0.10  * product_clicks
        + 0.40  * cart_additions
        + 0.08  * search_queries
        + 0.20  * returning_user
        + rng.normal(0, 0.5, size=N)
    )
    prob      = 1 / (1 + np.exp(-score + 2.5))
    purchased = (rng.random(size=N) < prob).astype(int)

    df = pd.DataFrame({
        "session_duration": session_duration.round(1),
        "pages_viewed":     pages_viewed,
        "product_clicks":   product_clicks,
        "cart_additions":   cart_additions,
        "search_queries":   search_queries,
        "returning_user":   returning_user,
        "device_type":      device_type,
        "time_of_day":      time_of_day,
        "purchased":        purchased,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Generated {len(df)} rows → {OUT}")
    print(f"Purchase rate: {purchased.mean():.2%}")


if __name__ == "__main__":
    main()
