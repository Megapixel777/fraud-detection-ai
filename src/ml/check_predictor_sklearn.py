from src.ml.predictor_sklearn import FraudPredictorSklearn


predictor = FraudPredictorSklearn()


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


result = predictor.predict(transaction)

print()
print("==========================")
print("PREDICTION")
print("==========================")
print(result)