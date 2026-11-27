from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features consistently for training, batch prediction and demo."""
    result = df.copy()

    for column in BOOLEAN_COLUMNS:
        if column in result.columns:
            result[column] = (
                result[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0})
                .astype("float32")
            )

    for column in CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].fillna("Unknown").astype(str)

    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "amount_usd" in result.columns:
        result["log_amount"] = np.log1p(result["amount_usd"].clip(lower=0))
    if {"amount_usd", "account_balance_usd"}.issubset(result.columns):
        result["amount_to_balance_ratio"] = (
            result["amount_usd"] / result["account_balance_usd"].clip(lower=1.0)
        )
    if "txn_count_last_24h" in result.columns and "hours_since_last_txn" in result.columns:
        result["txn_density"] = result["txn_count_last_24h"] / (
            result["hours_since_last_txn"].clip(lower=0) + 1.0
        )

    if "time_of_day_hour" in result.columns:
        hour = result["time_of_day_hour"]

        result["night_transaction"] = np.where(
            hour.isna(),
            np.nan,
            ((hour >= 22) | (hour <= 5)).astype("float32")
        )

    return result
