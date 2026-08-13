import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel


# ==========================
# Configuration
# ==========================

MODEL_PATH = "../models/fraud_random_forest_top10"

FEATURES = [
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


# ==========================
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionFeatureImportance")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)


# ==========================
# Load model
# ==========================

print()
print("==========================")
print("Loading model")
print("==========================")

model = RandomForestClassificationModel.load(
    MODEL_PATH
)

print(f"Model: {MODEL_PATH}")


# ==========================
# Feature importance
# ==========================

importance_values = model.featureImportances.toArray()


results = list(
    zip(FEATURES, importance_values)
)

results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ==========================
# Display results
# ==========================

print()
print("==========================")
print("Feature Importance")
print("==========================")

print(
    f"{'Rank':<8}"
    f"{'Feature':<12}"
    f"{'Importance':<15}"
)

print("-" * 35)


for rank, (feature, importance) in enumerate(
    results,
    start=1
):

    print(
        f"{rank:<8}"
        f"{feature:<12}"
        f"{importance:<15.6f}"
    )


# ==========================
# Stop Spark
# ==========================

spark.stop()

os._exit(0)