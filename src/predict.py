from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import pandas as pd

from .config import ID_COLUMN, MODEL_FILE
from .features import create_features


def load_artifact(model_path: Union[str, Path] = MODEL_FILE) -> dict:
    artifact = joblib.load(model_path)
    required_keys = {
        "models",
        "feature_columns",
        "model_weights",
    }

    if not isinstance(artifact, dict):
        raise ValueError("Tệp mô hình không đúng định dạng fraud_models.pkl")

    missing_keys = required_keys - artifact.keys()

    if missing_keys:
        raise ValueError(
            f"Model artifact thiếu các trường: {missing_keys}"
        )

    expected_models = {"xgboost", "catboost", "lightgbm"}

    if set(artifact["models"]) != expected_models:
        raise ValueError("Model artifact không chứa đủ 3 mô hình.")

    weights = artifact["model_weights"]

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Model weights không được âm.")

    if not abs(sum(weights.values()) - 1.0) < 1e-6:
        raise ValueError("Tổng model weights phải bằng 1.")

    if set(artifact["model_weights"]) != expected_models:
        raise ValueError("Model weights không khớp với 3 mô hình.")

    if any(len(artifact["models"][name]) == 0 for name in expected_models):
        raise ValueError("Một hoặc nhiều nhóm mô hình không có model.")
    
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
    models = artifact["models"]
    model_weights = artifact["model_weights"]

    model_predictions = {}

    for model_type, fold_models in models.items():
        fold_predictions = [
            model.predict_proba(frame)[:, 1]
            for model in fold_models
        ]

        model_predictions[model_type] = (
            pd.DataFrame(fold_predictions)
            .mean(axis=0)
            .to_numpy()
        )

    ensemble_prediction = sum(
        model_weights[model_type] * model_predictions[model_type]
        for model_type in model_predictions
    )

    return ensemble_prediction.astype(float).tolist()


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
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
