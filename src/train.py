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
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

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
        CATBOOST_PARAMS,
        CATEGORICAL_COLUMNS,
        LIGHTGBM_PARAMS,
        MODEL_WEIGHTS,
    )
    from .features import create_features


    def build_classifier(
        model_type: str,
        scale_pos_weight: float,
        n_estimators: int | None = None,
    ):
        if model_type == "xgboost":
            params = XGBOOST_PARAMS | {
                "scale_pos_weight": scale_pos_weight
            }
            if n_estimators is not None:
                params["n_estimators"] = n_estimators
            return XGBClassifier(**params)

        if model_type == "lightgbm":
            params = LIGHTGBM_PARAMS | {
                "scale_pos_weight": scale_pos_weight
            }
            if n_estimators is not None:
                params["n_estimators"] = n_estimators
            return LGBMClassifier(**params)

        if model_type == "catboost":
            params = CATBOOST_PARAMS | {
                "scale_pos_weight": scale_pos_weight
            }
            if n_estimators is not None:
                params["iterations"] = n_estimators
            return CatBoostClassifier(**params)

        raise ValueError(f"Mô hình không được hỗ trợ: {model_type}")


    def build_pipeline(scale_pos_weight: float, model_type: str, n_estimators: int | None = None,):
        numeric_columns = NUMERIC_COLUMNS + [
            "log_amount",
            "amount_to_balance_ratio",
            "txn_density",
            "night_transaction",
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

        if model_type == "catboost":
            return build_classifier(
                model_type,
                scale_pos_weight,
                n_estimators,
            )

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
                    CATEGORICAL_COLUMNS,
                ),
            ],
            remainder="drop",
        )

        classifier = build_classifier(model_type, scale_pos_weight, n_estimators,)
        return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--folds", type=int, default=N_FOLDS)
        parser.add_argument(
            "--n-estimators",
            type=int,
            default=None,
            help="Ghi đè số cây của cả ba mô hình",
        )
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

        model_types = ("xgboost", "catboost", "lightgbm")

        oof_predictions = {
            model_type: np.zeros(len(train_df), dtype=float)
            for model_type in model_types
        }

        test_predictions = {
            model_type: np.zeros(len(X_test), dtype=float)
            for model_type in model_types
        }

        fold_scores = {
            model_type: []
            for model_type in model_types
        }

        models = {
            model_type: []
            for model_type in model_types
        }

        skf = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for fold, (train_index, valid_index) in enumerate(
            skf.split(X, y),
            start=1,
        ):
            print(f"\n========== FOLD {fold}/{args.folds} ==========")

            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            negative_count_fold = int((y_train == 0).sum())
            positive_count_fold = int((y_train == 1).sum())

            scale_pos_weight_fold = (
                negative_count_fold / max(positive_count_fold, 1)
            )

            for model_type in model_types:
                print(f"\nTraining {model_type}...")

                model = build_pipeline(
                    scale_pos_weight=scale_pos_weight_fold,
                    model_type=model_type,
                    n_estimators=args.n_estimators,
                )

                if model_type == "catboost":
                    model.fit(
                        X_train,
                        y_train,
                        cat_features=CATEGORICAL_COLUMNS,
                    )
                else:
                    model.fit(X_train, y_train)

                valid_prediction = model.predict_proba(X_valid)[:, 1]

                oof_predictions[model_type][valid_index] = valid_prediction

                fold_auc = roc_auc_score(
                    y_valid,
                    valid_prediction,
                )

                fold_scores[model_type].append(float(fold_auc))

                test_prediction = model.predict_proba(X_test)[:, 1]

                test_predictions[model_type] += (
                    test_prediction / args.folds
                )

                models[model_type].append(model)

                print(
                    f"{model_type} Fold {fold}/{args.folds} "
                    f"ROC AUC: {fold_auc:.6f}"
                )

        oof_auc = {
            model_type: float(
                roc_auc_score(y, oof_predictions[model_type])
            )
            for model_type in model_types
        }

        print("\n========== INDIVIDUAL OOF RESULTS ==========")

        for model_type in model_types:
            print(
                f"{model_type}: "
                f"OOF ROC AUC = {oof_auc[model_type]:.6f}"
            )

        ensemble_oof = sum(
            MODEL_WEIGHTS[model_type] * oof_predictions[model_type]
            for model_type in model_types
        )

        ensemble_test = sum(
            MODEL_WEIGHTS[model_type] * test_predictions[model_type]
            for model_type in model_types
        )

        ensemble_auc = float(
            roc_auc_score(y, ensemble_oof)
        )

        print("\n========== ENSEMBLE RESULTS ==========")
        print(f"Ensemble OOF ROC AUC: {ensemble_auc:.6f}")

        # Final prediction for the test dataset.
        final_predictions = ensemble_test

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        args.submission_output.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "models": models,
            "model_types": model_types,
            "model_weights": MODEL_WEIGHTS,
            "feature_columns": list(X.columns),
            "categorical_columns": CATEGORICAL_COLUMNS,
            "target_column": TARGET_COLUMN,
            "oof_auc": oof_auc,
            "ensemble_oof_auc": ensemble_auc,
            "fold_scores": fold_scores,
            "oof_predictions": oof_predictions,
            "random_state": RANDOM_STATE,
        }
        joblib.dump(artifact, args.model_output)

        submission = pd.DataFrame(
            {
                ID_COLUMN: test_df[ID_COLUMN],
                TARGET_COLUMN: final_predictions,
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
                    "ensemble_oof_auc": ensemble_auc,
                    "fold_scores": fold_scores,
                    "model_weights": MODEL_WEIGHTS,
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
