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
DATA_PATH = "../data/silver/transactions_clean"

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

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


# ==========================
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionThresholdOptimization")
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
# Load Silver data
# ==========================

print()
print("==========================")
print("Loading Silver data")
print("==========================")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

print(f"Data: {DATA_PATH}")


# ==========================
# Prepare data
# ==========================

evaluation_df = df.select(
    *TOP_10_FEATURES,
    "Class"
)


# ==========================
# Assemble features
# ==========================

assembler = VectorAssembler(
    inputCols=TOP_10_FEATURES,
    outputCol="features_top10"
)

features_df = assembler.transform(
    evaluation_df
)


# ==========================
# Generate model predictions
# ==========================

predictions = model.transform(
    features_df
)


# ==========================
# Extract fraud probability
# ==========================

predictions = predictions.withColumn(
    "fraud_probability",
    vector_to_array(col("probability"))[1]
)


# ==========================
# Cache predictions
# ==========================

predictions.cache()

# Force evaluation once
predictions.count()


# ==========================
# Threshold optimization
# ==========================

results = []


print()
print("==========================")
print("Threshold Optimization")
print("==========================")


for threshold in THRESHOLDS:

    threshold_predictions = predictions.withColumn(
        "final_prediction",
        when(
            col("fraud_probability") >= threshold,
            1
        ).otherwise(0)
    )

    # True Positives
    tp = threshold_predictions.filter(
        (col("Class") == 1) &
        (col("final_prediction") == 1)
    ).count()

    # False Positives
    fp = threshold_predictions.filter(
        (col("Class") == 0) &
        (col("final_prediction") == 1)
    ).count()

    # False Negatives
    fn = threshold_predictions.filter(
        (col("Class") == 1) &
        (col("final_prediction") == 0)
    ).count()

    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # Recall
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # F1
    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    results.append(
        (
            threshold,
            tp,
            fp,
            fn,
            precision,
            recall,
            f1
        )
    )


# ==========================
# Display results
# ==========================

print()

print(
    f"{'Threshold':<12}"
    f"{'TP':<10}"
    f"{'FP':<10}"
    f"{'FN':<10}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 78)


for row in results:

    (
        threshold,
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    ) = row

    print(
        f"{threshold:<12.2f}"
        f"{tp:<10}"
        f"{fp:<10}"
        f"{fn:<10}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ==========================
# Find best threshold
# ==========================

best_result = max(
    results,
    key=lambda x: x[6]
)


(
    best_threshold,
    best_tp,
    best_fp,
    best_fn,
    best_precision,
    best_recall,
    best_f1
) = best_result


# ==========================
# Display best threshold
# ==========================

print()
print("==========================")
print("Best Threshold")
print("==========================")

print(f"Threshold: {best_threshold:.2f}")
print(f"TP:        {best_tp}")
print(f"FP:        {best_fp}")
print(f"FN:        {best_fn}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall:    {best_recall:.4f}")
print(f"F1:        {best_f1:.4f}")


# ==========================
# Stop Spark
# ==========================

predictions.unpersist()

spark.stop()

os._exit(0)