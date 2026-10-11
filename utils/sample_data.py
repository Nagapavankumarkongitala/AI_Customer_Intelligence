"""Reproducible SYNTHETIC data generator for demos and automated tests.

Output is clearly synthetic (see data/sample/README.md) and must never be mixed
with a user's uploaded dataset without explicit user action in the UI.
Run:  python utils/sample_data.py  [--customers N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Allow `python utils/sample_data.py` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.constants import RANDOM_SEED

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Jaipur"]
STATES = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "West Bengal", "Telangana"]
COUNTRY = "India"
REGIONS = ["North", "South", "East", "West"]
GENDERS = ["Male", "Female", "Other"]


def generate(n_customers: int = 1200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a dirty synthetic transaction table (dirt is intentional for Phase 1 demos)."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2024-01-01")
    end = np.datetime64("2025-12-31")
    span_days = int((end - start).astype(int))

    rows: list[dict] = []
    for i in range(n_customers):
        cid = f"C{i + 1:05d}"
        gender = rng.choice(GENDERS + [None], p=[0.46, 0.46, 0.03, 0.05])
        age = rng.integers(18, 75)
        if rng.random() < 0.04:  # invalid ages
            age = rng.choice([250, -5])
        if rng.random() < 0.05:  # missing ages
            age = None
        city = rng.choice(CITIES)
        state = rng.choice(STATES)
        region = rng.choice(REGIONS)
        n_txn = int(rng.poisson(8)) + 1
        offsets = np.sort(rng.integers(0, span_days, size=n_txn))
        spend_base = rng.lognormal(mean=6.5, sigma=0.9)
        for off in offsets:
            amount = round(float(rng.lognormal(mean=np.log(spend_base), sigma=0.5)), 2)
            rows.append(
                {
                    "customer_id": cid,
                    "transaction_date": str(start + int(off)),
                    "amount_spent": amount,
                    "gender": gender,
                    "age": age,
                    "city": city,
                    "state": state,
                    "country": COUNTRY,
                    "region": region,
                }
            )
    df = pd.DataFrame(rows)

    # --- Intentional dirt (controlled, for cleaning demos) ---
    n = len(df)
    dup_idx = rng.choice(n, size=max(1, n // 50), replace=False)  # ~2% exact duplicates
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    n = len(df)
    neg_idx = rng.choice(n, size=max(1, n // 100), replace=False)  # negative refunds
    df.loc[neg_idx, "amount_spent"] = -df.loc[neg_idx, "amount_spent"]
    out_idx = rng.choice(n, size=max(1, n // 200), replace=False)  # monetary outliers
    df.loc[out_idx, "amount_spent"] = df.loc[out_idx, "amount_spent"] * 50
    bad_date_idx = rng.choice(n, size=10, replace=False)
    df.loc[bad_date_idx, "transaction_date"] = rng.choice(["not-a-date", "32/13/2024", ""])
    miss_id_idx = rng.choice(n, size=5, replace=False)
    df.loc[miss_id_idx, "customer_id"] = None
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    """Generate the CSV sample file."""
    parser = argparse.ArgumentParser(description="Generate synthetic demo transactions.")
    parser.add_argument("--customers", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    df = generate(n_customers=args.customers, seed=args.seed)
    dest = Path("data/sample/sample_customer_transactions.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"Wrote {len(df)} rows, {df['customer_id'].nunique()} customers -> {dest}")


if __name__ == "__main__":
    main()
