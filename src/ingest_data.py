import os
import sys


# ==========================
# Force Spark to use the
# current Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession


# ==========================
# Configuration
# ==========================

INPUT_PATH = "../data/raw/creditcard.csv"
OUTPUT_PATH = "../data/bronze/creditcard"


# ==========================
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionBronzeIngestion")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)


# ==========================
# Read Raw data
# ==========================

print()
print("==========================")
print("Reading Raw data")
print("==========================")
print(f"Input: {INPUT_PATH}")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(INPUT_PATH)
)


# ==========================
# Show Bronze data
# ==========================

print()
print("==========================")
print("Raw Data Preview")
print("==========================")

df.show(5, truncate=False)


# ==========================
# Show schema
# ==========================

print()
print("==========================")
print("Raw Schema")
print("==========================")

df.printSchema()


# ==========================
# Write Bronze
# ==========================

(
    df.write
    .mode("overwrite")
    .option("header", True)
    .csv(OUTPUT_PATH)
)


# ==========================
# Final message
# ==========================

print()
print("==========================")
print("Bronze layer created")
print("==========================")
print(f"Input:  {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")
print("==========================")


# ==========================
# Stop Spark
# ==========================

spark.stop()

# Force process termination on Windows
os._exit(0)