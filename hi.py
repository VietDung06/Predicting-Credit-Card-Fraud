import pandas as pd
from src.predict import predict_fraud

test = pd.read_csv("data/test.csv").head(5)

predictions = predict_fraud(test)

print(predictions)