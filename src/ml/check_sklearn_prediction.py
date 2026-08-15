import joblib
import pandas as pd


MODEL_PATH = "models/fraud_random_forest_top10_sklearn.joblib"


transaction = {
    "V12": 0.0,
    "V17": 0.0,
    "V16": 0.0,
    "V7": 0.0,
    "V10": 0.0,
    "V14": 0.0,
    "V11": 0.0,
    "V4": 0.0,
    "V9": 0.0,
    "V3": 0.0,
}


bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

X = pd.DataFrame(
    [transaction],
    columns=features
)

probability = model.predict_proba(X)[0, 1]

prediction = int(
    probability >= threshold
)

label = (
    "FRAUD"
    if prediction == 1
    else "NORMAL"
)

print()
print("==========================")
print("SKLEARN PREDICTION")
print("==========================")
print(f"Probability: {probability}")
print(f"Threshold:   {threshold}")
print(f"Prediction:  {prediction}")
print(f"Label:       {label}")