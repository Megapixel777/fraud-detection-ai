import os
import sys

# ==========================
# Force Spark to use the
# current Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# ==========================
# Configuration
# ==========================

INPUT_PATH = "../data/bronze/transaction.csv"
OUTPUT_PATH = "../data/silver/transactions_clean"


# ==========================
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionDataPreparation")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)


# ==========================
# Read Bronze
# ==========================

print(f"Reading Bronze data from: {INPUT_PATH}")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(INPUT_PATH)
)


# ==========================
# Required features
# ==========================

required_features = [
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
# Validate columns
# ==========================

missing_columns = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_columns:
    print("ERROR: Missing required columns:")
    print(missing_columns)

    spark.stop()
    os._exit(1)


# ==========================
# Select required columns
# ==========================

silver_df = df.select(
    *required_features
)


# ==========================
# Convert features to double
# ==========================

for feature in required_features:
    silver_df = silver_df.withColumn(
        feature,
        col(feature).cast("double")
    )


# ==========================
# Remove invalid rows
# ==========================

silver_df = silver_df.dropna(
    subset=required_features
)


# ==========================
# Show Silver data
# ==========================

print()
print("==========================")
print("Silver Data")
print("==========================")

silver_df.show(
    truncate=False
)


# ==========================
# Save Silver
# ==========================

(
    silver_df
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(OUTPUT_PATH)
)


# ==========================
# Final message
# ==========================

print()
print("==========================")
print("Silver layer created")
print("==========================")
print(f"Input:  {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")
print("==========================")


# ==========================
# Stop Spark
# ==========================

spark.stop()
os._exit(0)