from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .config import (
    ID_COLUMN,
    MODEL_FILE,
    N_FOLDS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    SUBMISSION_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
    XGBOOST_PARAMS,
)
from .features import create_features


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    numeric_columns = NUMERIC_COLUMNS + [
        "log_amount",
        "amount_to_balance_ratio",
        "txn_density",
        "night_transaction",
    ]
    categorical_columns = [
        "merchant_category",
        "card_type",
        "auth_method",
        "channel",
        "device_type",
    ]
    # Include boolean columns in the numeric transformer.
    boolean_columns = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns + boolean_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    params = XGBOOST_PARAMS | {"scale_pos_weight": scale_pos_weight}
    classifier = XGBClassifier(**params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-estimators", type=int, default=XGBOOST_PARAMS["n_estimators"])
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_FILE,
        help="Tệp mô hình đầu ra; mặc định là models/fraud_models.pkl",
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=SUBMISSION_FILE,
        help="Tệp submission đầu ra; mặc định là outputs/submission.csv",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Tệp metrics JSON tùy chọn",
    )
    args = parser.parse_args()

    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction phải nằm trong khoảng (0, 1]")
    if args.folds < 2:
        parser.error("--folds phải lớn hơn hoặc bằng 2")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if args.sample_fraction < 1.0:
        train_df = train_df.sample(
            frac=args.sample_fraction,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    y = train_df[TARGET_COLUMN].astype(int)
    X = train_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    X_test = test_df.drop(columns=[ID_COLUMN])

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    oof_predictions = np.zeros(len(train_df), dtype=float)
    fold_scores = []
    skf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    models = []

    for fold, (train_index, valid_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        pipeline = build_pipeline(scale_pos_weight)
        pipeline.set_params(classifier__n_estimators=args.n_estimators)
        pipeline.fit(X_train, y_train)
        valid_prediction = pipeline.predict_proba(X_valid)[:, 1]
        oof_predictions[valid_index] = valid_prediction

        fold_auc = roc_auc_score(y_valid, valid_prediction)
        fold_scores.append(fold_auc)
        models.append(pipeline)
        print(f"Fold {fold}/{args.folds} ROC AUC: {fold_auc:.6f}")

    oof_auc = roc_auc_score(y, oof_predictions)
    print(f"OOF ROC AUC: {oof_auc:.6f}")

    test_predictions = np.mean(
        [model.predict_proba(X_test)[:, 1] for model in models],
        axis=0,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.submission_output.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "models": models,
        "feature_columns": list(X.columns),
        "target_column": TARGET_COLUMN,
        "oof_auc": oof_auc,
        "fold_scores": fold_scores,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
    }
    joblib.dump(artifact, args.model_output)

    submission = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            TARGET_COLUMN: test_predictions,
        }
    )
    submission.to_csv(args.submission_output, index=False)

    metrics_file = args.metrics_output or (
        args.submission_output.parent
        / f"{args.submission_output.stem}_metrics.json"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(
        json.dumps(
            {
                "oof_auc": oof_auc,
                "fold_scores": fold_scores,
                "n_samples": len(train_df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved models to {args.model_output}")
    print(f"Saved submission to {args.submission_output}")


if __name__ == "__main__":
    main()
