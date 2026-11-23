import unittest

import pandas as pd
from lightgbm import LGBMClassifier

from src.train import build_pipeline


class LightGBMTest(unittest.TestCase):
    def test_build_lightgbm_pipeline(self):
        pipeline = build_pipeline(scale_pos_weight=3.0, model_type="lightgbm")
        self.assertIsInstance(
            pipeline.named_steps["classifier"],
            LGBMClassifier,
        )

    def test_lightgbm_pipeline_fit_and_predict(self):
        frame = pd.DataFrame(
            {
                "amount_usd": [10.0, 20.0, 30.0, 40.0],
                "account_balance_usd": [100.0, 200.0, 300.0, 400.0],
                "merchant_category": ["A", "B", "A", "B"],
                "card_type": ["credit", "debit", "credit", "debit"],
                "auth_method": ["pin", "chip", "pin", "chip"],
                "channel": ["pos", "web", "pos", "web"],
                "device_type": ["phone", "pc", "phone", "pc"],
                "log_amount": [10.0, 20.0, 30.0, 40.0],
                "amount_to_balance_ratio": [0.1, 0.1, 0.1, 0.1],
                "txn_density": [1.0, 1.0, 1.0, 1.0],
                "night_transaction": [0, 1, 0, 1],
                "hours_since_last_txn": [1.0, 2.0, 3.0, 4.0],
                "txn_count_last_24h": [1, 2, 3, 4],
                "distance_from_home_km": [1.0, 2.0, 3.0, 4.0],
                "card_age_months": [12, 24, 36, 48],
                "customer_age": [20, 30, 40, 50],
                "cvv_retry_count": [0, 1, 0, 1],
                "velocity_score": [10.0, 20.0, 30.0, 40.0],
                "time_of_day_hour": [1, 12, 23, 3],
                "day_of_week": [0, 1, 2, 3],
                "merchant_risk_score": [10.0, 20.0, 30.0, 40.0],
                "prior_disputes": [0, 1, 0, 1],
                "is_foreign_transaction": [0, 1, 0, 1],
                "is_new_merchant": [0, 1, 0, 1],
                "used_vpn": [0, 1, 0, 1],
                "ip_country_mismatch": [0, 1, 0, 1],
                "billing_shipping_mismatch": [0, 1, 0, 1],
                "is_ai_generated_scam_attempt": [0, 1, 0, 1],
            }
        )
        target = pd.Series([0, 1, 0, 1])
        pipeline = build_pipeline(1.0, "lightgbm")
        pipeline.set_params(classifier__n_estimators=5)
        pipeline.fit(frame, target)
        probabilities = pipeline.predict_proba(frame)[:, 1]
        self.assertEqual(len(probabilities), len(frame))


if __name__ == "__main__":
    unittest.main()
