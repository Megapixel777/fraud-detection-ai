import os
import sys

# ==========================
# Force Spark to use the
# current Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql.functions import col, when
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array


# ==========================
# Configuration
# ==========================

MODEL_PATH = "models/fraud_random_forest_top10"

TOP_10_FEATURES = [
    "V12",
    "V17",
    "V16",
    "V7",
    "V10",
    "V14",
    "V11",
    "V4",
    "V9",
    "V3"
]

THRESHOLD = 0.20


# ==========================
# Fraud Predictor
# ==========================

class FraudPredictor:

    def __init__(self, spark):

        self.spark = spark

        print()
        print("==========================")
        print("LOADING FRAUD MODEL")
        print("==========================")
        print(f"Model path: {MODEL_PATH}")
        print(f"Threshold:  {THRESHOLD}")
        print("Features:")

        for feature in TOP_10_FEATURES:
            print(f"  - {feature}")

        print("==========================")

        self.model = RandomForestClassificationModel.load(
            MODEL_PATH
        )

        self.assembler = VectorAssembler(
            inputCols=TOP_10_FEATURES,
            outputCol="features_top10"
        )

        print("Model loaded successfully.")


    # ==========================
    # Prediction
    # ==========================

    def predict(self, transaction_df):

        # Create feature vector
        features_df = self.assembler.transform(
            transaction_df
        )

        # Generate Random Forest prediction
        prediction_df = self.model.transform(
            features_df
        )

        # Extract fraud probability
        prediction_df = prediction_df.withColumn(
            "fraud_probability",
            vector_to_array(
                col("probability")
            )[1]
        )

        # Apply validated threshold
        prediction_df = prediction_df.withColumn(
            "final_prediction",
            when(
                col("fraud_probability") >= THRESHOLD,
                1
            ).otherwise(0)
        )

        # Human-readable prediction
        prediction_df = prediction_df.withColumn(
            "prediction_label",
            when(
                col("final_prediction") == 1,
                "FRAUD"
            ).otherwise("NORMAL")
        )

        return prediction_df