# Credit Card Fraud - Machine Learning Project

Project Python cho cuộc thi Kaggle **Playground Series - Season 6, Episode 11: Predicting Credit Card Fraud**.

## Cấu trúc

```text
data/          train.csv, test.csv
models/        fraud_models.pkl
outputs/       submission.csv, biểu đồ EDA và training_metrics.json
src/
  config.py    đường dẫn, cột dữ liệu và tham số mô hình
  features.py  feature engineering dùng chung cho train và demo
  eda.py       tạo biểu đồ phân tích
  train.py     training 5-fold và xuất submission
  predict.py   load ensemble và dự đoán 1 dòng/batch
app_demo.py    ứng dụng Streamlit offline
```

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Đặt `train.csv` và `test.csv` trong thư mục `data/` (project đã tổ chức sẵn).

## Huấn luyện

```powershell
python -m src.train
```

Chạy nhanh thử nghiệm nhỏ:

```powershell
python -m src.train --folds 3 --n-estimators 50 --sample-fraction 0.1
```

Để thử 10% dữ liệu mà không ghi đè mô hình và submission chính:

```powershell
python -m src.train --folds 5 --n-estimators 100 --sample-fraction 0.1 --model-output models/fraud_models_demo10.pkl --submission-output outputs/submission_demo10.csv --metrics-output outputs/training_metrics_demo10.json
```

Output:
- `models/fraud_models.pkl`: 5 pipeline XGBoost fold
- `outputs/submission.csv`: file nộp Kaggle
- `outputs/training_metrics.json`: OOF AUC và AUC từng fold

## EDA

Sau khi training:

```powershell
python -m src.eda
```

Các biểu đồ được lưu tại `outputs/`: mất cân bằng lớp, phân phối biến quan trọng, tương quan và feature importance.

## Dự đoán batch

```powershell
python -c "from src.predict import predict_csv; print(predict_csv('data/test.csv', 'outputs/batch_predictions.csv').head())"
```

## Chạy Demo offline

```powershell
streamlit run app_demo.py
```

Trình duyệt mặc định mở giao diện tại `http://localhost:8501`. Demo hỗ trợ nhập một giao dịch và tải lên CSV để dự đoán batch.

## Chạy trên Kaggle

1. Upload project hoặc copy các file trong `src/` vào notebook.
2. Trỏ `TRAIN_FILE` và `TEST_FILE` sang `/kaggle/input/...` nếu cần.
3. Chạy `python -m src.train`.
4. Tải `outputs/submission.csv` lên Kaggle.
