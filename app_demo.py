import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

from src.config import MODEL_FILE, TRAIN_FILE
from src.predict import predict_fraud


@st.cache_resource
def load_artifact(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_train_sample():
    return pd.read_csv(TRAIN_FILE).head(500)


st.set_page_config(page_title="Phát hiện gian lận thẻ tín dụng", layout="wide")
st.title("Demo phát hiện gian lận thẻ tín dụng")
model_files = sorted(Path(MODEL_FILE.parent).glob("fraud_models*.pkl"))
if not model_files:
    st.error("Chưa có mô hình. Hãy chạy `python -m src.train` trước.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Chọn mô hình",
    model_files,
    format_func=lambda path: path.name,
)
artifact = load_artifact(selected_model)
MODEL_NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}
model_name = MODEL_NAMES.get(artifact.get("model_type", "xgboost"), "XGBoost")
st.caption(f"Mô hình {model_name} ensemble với 5-fold StratifiedKFold")
st.sidebar.metric("OOF ROC AUC", f"{artifact.get('oof_auc', 0):.4f}")

train_sample = load_train_sample()
st.subheader("Nhập thông tin giao dịch")

def categorical_input(label: str, column: str):
    values = sorted(train_sample[column].dropna().unique().tolist())
    return st.selectbox(label, values)


col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Số tiền (USD)", 0.0, 1_000_000.0, 100.0)
    hours = st.number_input("Giờ kể từ giao dịch trước", 0.0, 1000.0, 5.0)
    txn_count = st.number_input("Số giao dịch 24 giờ", 0, 100, 3)
    distance = st.number_input("Khoảng cách từ nhà (km)", 0.0, 10000.0, 10.0)
    card_age = st.number_input("Tuổi thẻ (tháng)", 0, 1000, 36)
    age = st.number_input("Tuổi khách hàng", 18, 100, 40)

with col2:
    balance = st.number_input("Số dư tài khoản (USD)", 0.0, 10_000_000.0, 2000.0)
    cvv_retry = st.number_input("Số lần nhập lại CVV", 0, 20, 0)
    velocity = st.number_input("Velocity score", 0.0, 100.0, 20.0)
    hour = st.number_input("Giờ trong ngày", 0, 23, 12)
    day = st.number_input("Ngày trong tuần (0-6)", 0, 6, 2)
    merchant_risk = st.number_input("Merchant risk score", 0.0, 100.0, 20.0)

with col3:
    prior_disputes = st.number_input("Số tranh chấp trước", 0, 100, 0)
    merchant_category = categorical_input("Loại merchant", "merchant_category")
    card_type = categorical_input("Loại thẻ", "card_type")
    auth_method = categorical_input("Phương thức xác thực", "auth_method")
    channel = categorical_input("Kênh", "channel")
    device_type = categorical_input("Thiết bị", "device_type")

boolean_labels = {
    "is_foreign_transaction": "Giao dịch nước ngoài",
    "is_new_merchant": "Merchant mới",
    "used_vpn": "Dùng VPN",
    "ip_country_mismatch": "IP khác quốc gia",
    "billing_shipping_mismatch": "Địa chỉ thanh toán khác giao hàng",
    "is_ai_generated_scam_attempt": "Cố gắng lừa đảo AI",
}
st.subheader("Dấu hiệu rủi ro")
flags = {}
flag_columns = st.columns(3)
for index, (column, label) in enumerate(boolean_labels.items()):
    with flag_columns[index % 3]:
        flags[column] = st.checkbox(label)

transaction = {
    "amount_usd": amount,
    "hours_since_last_txn": hours,
    "txn_count_last_24h": txn_count,
    "distance_from_home_km": distance,
    "card_age_months": card_age,
    "customer_age": age,
    "account_balance_usd": balance,
    "cvv_retry_count": cvv_retry,
    "velocity_score": velocity,
    "time_of_day_hour": hour,
    "day_of_week": day,
    "merchant_risk_score": merchant_risk,
    "prior_disputes": prior_disputes,
    "merchant_category": merchant_category,
    "card_type": card_type,
    "auth_method": auth_method,
    "channel": channel,
    "device_type": device_type,
    **{column: int(value) for column, value in flags.items()},
}

st.subheader("Dự đoán")
if st.button("Dự đoán xác suất gian lận", type="primary"):
    probability = predict_fraud(transaction, model_path=selected_model)[0]
    st.metric("Xác suất gian lận", f"{probability:.2%}")
    if probability >= 0.5:
        st.error("CẢNH BÁO: Giao dịch có khả năng gian lận cao")
    elif probability >= 0.2:
        st.warning("Cần kiểm tra thêm giao dịch này")
    else:
        st.success("Giao dịch có rủi ro thấp")

st.divider()
uploaded = st.file_uploader("Hoặc dự đoán batch từ file CSV", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    if st.button("Dự đoán batch"):
        probabilities = predict_fraud(batch, model_path=selected_model)
        result = batch.copy()
        result["is_fraud_probability"] = probabilities
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Tải kết quả",
            result.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv",
            "text/csv",
        )
