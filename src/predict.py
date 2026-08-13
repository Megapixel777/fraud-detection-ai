import os
import sys


# ==========================
# Force Spark to use the
# current Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, monotonically_increasing_id
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

THRESHOLD = 0.20

OUTPUT_PATH = "../data/gold/fraud_predictions"


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
# Model Information
# ==========================

print()
print("==========================")
print("MODEL")
print("==========================")
print(f"Model path: {MODEL_PATH}")
print(f"Threshold:  {THRESHOLD}")
print("Features:")

for feature in TOP_10_FEATURES:
    print(f"  - {feature}")

print("==========================")


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

    # -------------------------------------------------
    # Create an internal row identifier.
    #
    # This prevents duplicated rows when joining the
    # predictions back to the original transactions.
    # -------------------------------------------------

    original_df = transaction_df.withColumn(
        "_prediction_id",
        monotonically_increasing_id()
    )


    # -------------------------------------------------
    # Select only the features required by the model
    # -------------------------------------------------

    transaction_features = original_df.select(
        "_prediction_id",
        *TOP_10_FEATURES
    )


    # -------------------------------------------------
    # Create feature vector
    # -------------------------------------------------

    features_df = assembler.transform(
        transaction_features
    )


    # -------------------------------------------------
    # Generate Random Forest prediction
    # -------------------------------------------------

    prediction_df = model.transform(
        features_df
    )


    # -------------------------------------------------
    # Extract fraud probability
    # -------------------------------------------------

    prediction_df = prediction_df.withColumn(
        "fraud_probability",
        vector_to_array(
            col("probability")
        )[1]
    )


    # -------------------------------------------------
    # Apply validated threshold
    # -------------------------------------------------

    prediction_df = prediction_df.withColumn(
        "final_prediction",
        when(
            col("fraud_probability") >= THRESHOLD,
            1.0
        ).otherwise(0.0)
    )


    # -------------------------------------------------
    # Add human-readable prediction
    # -------------------------------------------------

    prediction_df = prediction_df.withColumn(
        "prediction_label",
        when(
            col("final_prediction") == 1.0,
            "FRAUD"
        ).otherwise("NORMAL")
    )


    # -------------------------------------------------
    # Keep only the prediction information needed
    # -------------------------------------------------

    prediction_results = prediction_df.select(
        "_prediction_id",
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    )


    # -------------------------------------------------
    # Join predictions back to original transactions
    # using the internal row identifier.
    # -------------------------------------------------

    result = (
        original_df
        .join(
            prediction_results,
            on="_prediction_id",
            how="inner"
        )
        .drop("_prediction_id")
    )


    return result


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    # ==========================
    # Check input argument
    # ==========================

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python predict.py <input_csv_or_spark_path>"
        )

        print()
        print("Example:")
        print(
            "python predict.py "
            "../data/silver/transactions_clean"
        )

        spark.stop()
        os._exit(1)


    input_path = sys.argv[1]


    # ==========================
    # Load input data
    # ==========================

    print()
    print(
        f"Loading transactions from: {input_path}"
    )

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )


    # ==========================
    # Input information
    # ==========================

    print()
    print("==========================")
    print("INPUT DATA")
    print("==========================")

    print(f"Columns: {len(transactions.columns)}")

    transactions.printSchema()


    # ==========================
    # Generate predictions
    # ==========================

    result = predict_fraud(
        transactions
    )


    # ==========================
    # Display predictions
    # ==========================

    print()
    print("==========================")
    print("FIRST 20 FRAUD PREDICTIONS")
    print("==========================")

    result.select(
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    ).show(
        20,
        truncate=False
    )


    # ==========================
    # Gold result
    # ==========================

    gold_result = result.select(
        *transactions.columns,
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    )


    # ==========================
    # Save Gold
    # ==========================

    print()
    print("==========================")
    print("SAVING GOLD")
    print("==========================")

    (
        gold_result
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
    print("Prediction completed")
    print("==========================")
    print(f"Model:     {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Input:     {input_path}")
    print(f"Output:    {OUTPUT_PATH}")
    print("==========================")


    # ==========================
    # Stop Spark
    # ==========================

    spark.stop()

    # Force process termination on Windows
    os._exit(0)