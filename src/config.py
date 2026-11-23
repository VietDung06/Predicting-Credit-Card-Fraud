from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
MODEL_FILE = MODEL_DIR / "fraud_models.pkl"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

ID_COLUMN = "id"
TARGET_COLUMN = "is_fraud"
RANDOM_STATE = 42
N_FOLDS = 5

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

BOOLEAN_COLUMNS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

NUMERIC_COLUMNS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "random_state": RANDOM_STATE,
    "eval_metric": "auc",
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

SUPPORTED_MODELS = ("xgboost", "lightgbm")
