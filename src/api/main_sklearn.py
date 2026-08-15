from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.fraud_agent import investigate
from src.agent.schemas import InvestigationResult
from src.ml.predictor_sklearn import FraudPredictorSklearn


# ==========================
# FastAPI
# ==========================

app = FastAPI(
    title="Fraud Detection API - scikit-learn",
    description="API for fraud detection using scikit-learn",
    version="2.0.0"
)


# ==========================
# Request Model
# ==========================

class Transaction(BaseModel):

    V12: float
    V17: float
    V16: float
    V7: float
    V10: float
    V14: float
    V11: float
    V4: float
    V9: float
    V3: float


# ==========================
# Fraud Predictor
# ==========================

predictor = FraudPredictorSklearn()


# ==========================
# Health Check
# ==========================

@app.get("/health")
def health_check():

    return {
        "status": "ok"
    }


# ==========================
# Prediction
# ==========================

@app.post("/predict")
def predict(transaction: Transaction):

    result = predictor.predict(
        transaction.model_dump()
    )

    print(
        f"Prediction: "
        f"probability={result['fraud_probability']:.6f}, "
        f"prediction={result['prediction_label']}"
    )

    return result

# ==========================
# Fraud Investigation
# ==========================

@app.post(
    "/investigate",
    response_model=InvestigationResult
)
def investigate_transaction(transaction: Transaction):

    return investigate(
        transaction.model_dump()
    )