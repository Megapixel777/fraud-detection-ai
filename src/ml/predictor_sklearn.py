import joblib
import pandas as pd


# ==========================
# Configuration
# ==========================

MODEL_PATH = "models/fraud_random_forest_top10_sklearn.joblib"


# ==========================
# Fraud Predictor
# ==========================

class FraudPredictorSklearn:

    def __init__(self, model_path=MODEL_PATH):

        print()
        print("==========================")
        print("LOADING SKLEARN FRAUD MODEL")
        print("==========================")
        print(f"Model path: {model_path}")

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.features = bundle["features"]

        print(f"Threshold:  {self.threshold}")
        print("Features:")

        for feature in self.features:
            print(f"  - {feature}")

        print("==========================")
        print("Model loaded successfully.")


    # ==========================
    # Prediction
    # ==========================

    def predict(self, transaction):

        transaction_df = pd.DataFrame(
            [transaction],
            columns=self.features
        )

        probability = self.model.predict_proba(
            transaction_df
        )[0, 1]

        final_prediction = int(
            probability >= self.threshold
        )

        prediction_label = (
            "FRAUD"
            if final_prediction == 1
            else "NORMAL"
        )

        return {
            "fraud_probability": float(
                probability
            ),
            "final_prediction": final_prediction,
            "prediction_label": prediction_label
        }