import os
import sys

# ==========================
# Force Spark to use the
# current Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from src.ml.predictor import FraudPredictor


# ==========================
# FastAPI
# ==========================

app = FastAPI(
    title="Fraud Detection API",
    description="API for fraud detection using Machine Learning",
    version="1.0.0"
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
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionAPI")
    .config("spark.sql.shuffle.partitions", "2")
    .config(
        "spark.sql.execution.pyspark.udf.faulthandler.enabled",
        "true"
    )
    .config(
        "spark.python.worker.faulthandler.enabled",
        "true"
    )
    .getOrCreate()
)


# ==========================
# Fraud Predictor
# ==========================

predictor = FraudPredictor(spark)


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

    transaction_data = transaction.model_dump()

    transaction_df = spark.createDataFrame(
        [transaction_data]
    )

    prediction_df = predictor.predict(
        transaction_df
    )

    result = prediction_df.select(
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    ).first()

    print(
        f"Prediction: "
        f"probability={result['fraud_probability']:.6f}, "
        f"prediction={result['prediction_label']}"
    )

    return {
        "fraud_probability": float(
            result["fraud_probability"]
        ),
        "final_prediction": int(
            result["final_prediction"]
        ),
        "prediction_label": result["prediction_label"]
    }