from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import pandas as pd

from .config import ID_COLUMN, MODEL_FILE
from .features import create_features


def load_artifact(model_path: Union[str, Path] = MODEL_FILE) -> dict:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "models" not in artifact:
        raise ValueError("Tệp mô hình không đúng định dạng fraud_models.pkl")
    return artifact


def predict_fraud(
    data: Union[pd.DataFrame, dict],
    model_path: Union[str, Path] = MODEL_FILE,
) -> list[float]:
    """Return ensemble fraud probabilities for one transaction or a batch."""
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    artifact = load_artifact(model_path)

    frame = create_features(frame)
    feature_columns = artifact["feature_columns"]
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Thiếu các cột đầu vào: {missing_features}")

    frame = frame.reindex(columns=feature_columns)
    probabilities = [
        model.predict_proba(frame)[:, 1] for model in artifact["models"]
    ]
    return list(pd.DataFrame(probabilities).mean(axis=0).astype(float))


def predict_csv(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    model_path: Union[str, Path] = MODEL_FILE,
) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    probabilities = predict_fraud(data, model_path)
    ids = data[ID_COLUMN] if ID_COLUMN in data.columns else range(len(data))
    result = pd.DataFrame(
        {
            ID_COLUMN: ids,
            "is_fraud": probabilities,
        }
    )
    result.to_csv(output_csv, index=False)
    return result
