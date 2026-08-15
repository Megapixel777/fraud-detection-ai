import joblib

from src.ml.predictor_sklearn import FraudPredictorSklearn


MODEL_PATH = "models/fraud_random_forest_top10_sklearn.joblib"


# ==========================
# Fraud Prediction Tool
# ==========================

predictor = FraudPredictorSklearn()


def predict_fraud(transaction: dict) -> dict:
    """
    Predict whether a transaction is fraudulent
    using the existing scikit-learn model.
    """

    return predictor.predict(transaction)


# ==========================
# Feature Importance Tool
# ==========================

def get_feature_importance() -> list:
    """
    Return feature importance values from the
    deployed Random Forest model.
    """

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]

    importances = model.feature_importances_

    result = [
        {
            "feature": feature,
            "importance": float(importance)
        }
        for feature, importance in zip(features, importances)
    ]

    result.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return result

# ==========================
# Model Configuration Tool
# ==========================

def get_model_configuration() -> dict:
    """
    Return the configuration of the fraud detection model.
    """

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    threshold = bundle["threshold"]
    features = bundle["features"]

    return {
        "model": type(model).__name__,
        "threshold": float(threshold),
        "features": features
    }