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

print("Fraud detection model loaded successfully.")
print(f"Model path: {MODEL_PATH}")
print(f"Threshold: {THRESHOLD}")
print(f"Features: {TOP_10_FEATURES}")


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

    # Keep the original transaction data
    original_df = transaction_df

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

    # Apply threshold
    prediction_df = prediction_df.withColumn(
        "final_prediction",
        when(
            col("fraud_probability") >= THRESHOLD,
            1.0
        ).otherwise(0.0)
    )

    # Add readable prediction
    prediction_df = prediction_df.withColumn(
        "prediction_label",
        when(
            col("final_prediction") == 1.0,
            "FRAUD"
        ).otherwise("NORMAL")
    )

    # Return original columns plus predictions
    #
    # The prediction dataframe contains the TOP 10 features,
    # so we join the prediction results back to the original
    # transaction dataframe.
    prediction_results = prediction_df.select(
        *TOP_10_FEATURES,
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    )

    result = original_df.join(
        prediction_results,
        on=TOP_10_FEATURES,
        how="inner"
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

    print()
    print(
        f"Loading transactions from: {input_path}"
    )


    # ==========================
    # Read Silver data
    # ==========================

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )


    # ==========================
    # Generate predictions
    # ==========================

    result = predict_fraud(
        transactions
    )


    # ==========================
    # Display results
    # ==========================

    print()
    print("==========================")
    print("Fraud Predictions")
    print("==========================")

    result.select(
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    ).show(
        truncate=False
    )


    # ==========================
    # Gold output
    # ==========================

    output_path = (
        "../data/gold/fraud_predictions"
    )


    # Select original Silver columns
    # plus model output columns
    gold_result = result.select(
        *transactions.columns,
        "fraud_probability",
        "final_prediction",
        "prediction_label"
    )


    # ==========================
    # Save Gold
    # ==========================

    (
        gold_result
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(output_path)
    )


    # ==========================
    # Final message
    # ==========================

    print()
    print("==========================")
    print("Prediction completed")
    print("==========================")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("==========================")


    # ==========================
    # Stop Spark
    # ==========================

    spark.stop()

    # Force process termination on Windows
    os._exit(0)