import os
import sys

# ==========================
# Force Spark to use the
# current Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array


# ==========================
# Configuration
# ==========================

MODEL_PATH = "../models/fraud_random_forest_top10"

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

THRESHOLD = 0.35


# ==========================
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionInference")
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
# Load Model
# ==========================

model = RandomForestClassificationModel.load(
    MODEL_PATH
)


# ==========================
# Feature Assembler
# ==========================

assembler = VectorAssembler(
    inputCols=TOP_10_FEATURES,
    outputCol="features_top10"
)


# ==========================
# Fraud Prediction
# ==========================

def predict_fraud(transaction_df):

    # Select only the features required by the model
    transaction_features = transaction_df.select(
        *TOP_10_FEATURES
    )

    # Create feature vector
    features_df = assembler.transform(
        transaction_features
    )

    # Generate prediction
    prediction_df = model.transform(
        features_df
    )

    # Extract fraud probability
    prediction_df = prediction_df.withColumn(
        "fraud_probability",
        vector_to_array(col("probability"))[1]
    )

    # Apply optimized threshold
    prediction_df = prediction_df.withColumn(
        "final_prediction",
        when(
            col("fraud_probability") >= THRESHOLD,
            1.0
        ).otherwise(0.0)
    )

    return prediction_df


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    # Example transaction
    transaction = {
        "V12": -2.312227,
        "V17": -2.808987,
        "V16": -1.378318,
        "V7": -0.881974,
        "V10": -1.120993,
        "V14": -4.289254,
        "V11": -2.770089,
        "V4": 1.309969,
        "V9": -0.392049,
        "V3": 1.165455
    }

    # Create Spark DataFrame
    transaction_df = (
        spark.createDataFrame([transaction])
        .repartition(1)
    )

    # Generate prediction
    result = predict_fraud(
        transaction_df
    )

    # Get result
    prediction = result.select(
        "fraud_probability",
        "final_prediction"
    ).first()

    fraud_probability = prediction["fraud_probability"]
    final_prediction = prediction["final_prediction"]

    # Display result
    print()
    print("==========================")
    print("Fraud Detection Result")
    print("==========================")
    print(f"Fraud probability: {fraud_probability:.4f}")

    if final_prediction == 1.0:
        print("Prediction: FRAUD")
    else:
        print("Prediction: NORMAL")

    print("==========================")

    # Stop Spark
    spark.stop()

    # Force process termination on Windows
    os._exit(0)