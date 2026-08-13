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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.functions import vector_to_array


# ==========================
# Configuration
# ==========================

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

FEATURES_WITH_AMOUNT = TOP_10_FEATURES + ["Amount"]

THRESHOLD = 0.45


# ==========================
# Spark Session
# ==========================

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FraudDetectionAmountComparison")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)


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

df = df.select(
    *(TOP_10_FEATURES + ["Amount", "Class"])
)

df = df.dropna()

print()
print(f"Number of records: {df.count()}")


# ==========================
# Function to train/evaluate
# ==========================

def evaluate_model(feature_list, model_name):

    print()
    print("==========================")
    print(f"Training: {model_name}")
    print("==========================")

    # ----------------------
    # Assemble features
    # ----------------------

    assembler = VectorAssembler(
        inputCols=feature_list,
        outputCol="features"
    )

    assembled_df = assembler.transform(df)

    # ----------------------
    # Random Forest
    # ----------------------

    rf = RandomForestClassifier(
        labelCol="Class",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=100,
        seed=42
    )

    model = rf.fit(assembled_df)

    # ----------------------
    # Predictions
    # ----------------------

    predictions = model.transform(assembled_df)

    predictions = predictions.withColumn(
        "fraud_probability",
        vector_to_array(
            col("probability")
        )[1]
    )

    # ----------------------
    # Apply threshold
    # ----------------------

    predictions = predictions.withColumn(
        "final_prediction",
        when(
            col("fraud_probability") >= THRESHOLD,
            1.0
        ).otherwise(0.0)
    )

    # ----------------------
    # Confusion matrix
    # ----------------------

    tp = predictions.filter(
        (col("Class") == 1) &
        (col("final_prediction") == 1)
    ).count()

    fp = predictions.filter(
        (col("Class") == 0) &
        (col("final_prediction") == 1)
    ).count()

    fn = predictions.filter(
        (col("Class") == 1) &
        (col("final_prediction") == 0)
    ).count()

    # ----------------------
    # Metrics
    # ----------------------

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    # ----------------------
    # PR-AUC
    # ----------------------

    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    pr_auc = pr_evaluator.evaluate(
        predictions
    )

    # ----------------------
    # ROC-AUC
    # ----------------------

    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    roc_auc = roc_evaluator.evaluate(
        predictions
    )

    # ----------------------
    # Display
    # ----------------------

    print()
    print(f"Features: {feature_list}")
    print(f"Threshold: {THRESHOLD}")

    print()
    print(f"TP:        {tp}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return {
        "model": model_name,
        "features": len(feature_list),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }


# ==========================
# Model 1
# ==========================

top10_results = evaluate_model(
    TOP_10_FEATURES,
    "Top 10"
)


# ==========================
# Model 2
# ==========================

amount_results = evaluate_model(
    FEATURES_WITH_AMOUNT,
    "Top 10 + Amount"
)


# ==========================
# Comparison
# ==========================

print()
print("==========================")
print("MODEL COMPARISON")
print("==========================")

print()

print(
    f"{'Model':<20}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'PR-AUC':<12}"
    f"{'ROC-AUC':<12}"
)

print("-" * 78)

for result in [
    top10_results,
    amount_results
]:

    print(
        f"{result['model']:<20}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
        f"{result['pr_auc']:<12.4f}"
        f"{result['roc_auc']:<12.4f}"
    )


# ==========================
# Determine best model
# ==========================

if amount_results["f1"] > top10_results["f1"]:

    best_model = "Top 10 + Amount"

elif amount_results["f1"] < top10_results["f1"]:

    best_model = "Top 10"

else:

    # If F1 is identical, use PR-AUC
    if amount_results["pr_auc"] > top10_results["pr_auc"]:
        best_model = "Top 10 + Amount"
    else:
        best_model = "Top 10"


print()
print("==========================")
print("BEST MODEL")
print("==========================")

print(f"Model: {best_model}")


# ==========================
# Stop Spark
# ==========================

spark.stop()

os._exit(0)