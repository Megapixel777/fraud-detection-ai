import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator


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

LABEL_COLUMN = "Class"

RANDOM_SEED = 42

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
    .appName("FraudDetectionStratifiedValidation")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)


# ==========================
# Load Silver
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
    .select(*(TOP_10_FEATURES + [LABEL_COLUMN]))
    .dropna()
)

total_records = df.count()

print(f"Dataset: {DATA_PATH}")
print(f"Total records: {total_records}")


# ==========================
# Show original distribution
# ==========================

print()
print("==========================")
print("Original Class Distribution")
print("==========================")

df.groupBy(LABEL_COLUMN).count().orderBy(
    LABEL_COLUMN
).show()

total_fraud = df.filter(
    col(LABEL_COLUMN) == 1
).count()

fraud_rate = total_fraud / total_records

print(f"Fraud records: {total_fraud}")
print(f"Fraud rate:    {fraud_rate:.4%}")


# ==========================
# Stratified split
# ==========================
#
# We split each class independently:
#
# Class 0:
#   80% Train
#   10% Validation
#   10% Test
#
# Class 1:
#   80% Train
#   10% Validation
#   10% Test
#
# This keeps approximately the
# same fraud ratio in every set.
# ==========================

print()
print("==========================")
print("Creating Stratified Split")
print("==========================")

normal_df = df.filter(
    col(LABEL_COLUMN) == 0
)

fraud_df = df.filter(
    col(LABEL_COLUMN) == 1
)


# --------------------------
# Split NORMAL
# --------------------------

normal_train, normal_validation, normal_test = (
    normal_df.randomSplit(
        [0.80, 0.10, 0.10],
        seed=RANDOM_SEED
    )
)


# --------------------------
# Split FRAUD
# --------------------------

fraud_train, fraud_validation, fraud_test = (
    fraud_df.randomSplit(
        [0.80, 0.10, 0.10],
        seed=RANDOM_SEED
    )
)


# --------------------------
# Recombine
# --------------------------

train_df = normal_train.unionByName(
    fraud_train
)

validation_df = normal_validation.unionByName(
    fraud_validation
)

test_df = normal_test.unionByName(
    fraud_test
)


# ==========================
# Dataset sizes
# ==========================

train_count = train_df.count()
validation_count = validation_df.count()
test_count = test_df.count()

print()
print("==========================")
print("Dataset Sizes")
print("==========================")

print(f"Train:      {train_count}")
print(f"Validation: {validation_count}")
print(f"Test:       {test_count}")


# ==========================
# Distribution function
# ==========================

def show_distribution(name, dataset):

    total = dataset.count()

    fraud = dataset.filter(
        col(LABEL_COLUMN) == 1
    ).count()

    normal = dataset.filter(
        col(LABEL_COLUMN) == 0
    ).count()

    print()
    print(f"{name}")
    print("-" * 30)
    print(f"Total:  {total}")
    print(f"Normal: {normal}")
    print(f"Fraud:  {fraud}")
    print(f"Fraud rate: {fraud / total:.4%}")


# ==========================
# Verify stratification
# ==========================

print()
print("==========================")
print("Stratification Check")
print("==========================")

show_distribution(
    "TRAIN",
    train_df
)

show_distribution(
    "VALIDATION",
    validation_df
)

show_distribution(
    "TEST",
    test_df
)


# ==========================
# Assemble features
# ==========================

assembler = VectorAssembler(
    inputCols=TOP_10_FEATURES,
    outputCol="features"
)

train_features = assembler.transform(
    train_df
)

validation_features = assembler.transform(
    validation_df
)

test_features = assembler.transform(
    test_df
)


# ==========================
# Train Random Forest
# ==========================

print()
print("==========================")
print("Training Random Forest")
print("==========================")

rf = RandomForestClassifier(
    labelCol=LABEL_COLUMN,
    featuresCol="features",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    numTrees=100,
    seed=RANDOM_SEED
)

model = rf.fit(train_features)

print("Training completed.")


# ==========================
# Validation predictions
# ==========================

validation_predictions = model.transform(
    validation_features
)

validation_predictions = validation_predictions.withColumn(
    "fraud_probability",
    vector_to_array(
        col("probability")
    )[1]
)


# ==========================
# Threshold optimization
# ==========================

print()
print("==========================")
print("Threshold Optimization")
print("Validation Set")
print("==========================")

validation_results = []


for threshold in THRESHOLDS:

    current_predictions = validation_predictions.withColumn(
        "final_prediction",
        when(
            col("fraud_probability") >= threshold,
            1.0
        ).otherwise(0.0)
    )

    tp = current_predictions.filter(
        (col(LABEL_COLUMN) == 1) &
        (col("final_prediction") == 1)
    ).count()

    fp = current_predictions.filter(
        (col(LABEL_COLUMN) == 0) &
        (col("final_prediction") == 1)
    ).count()

    fn = current_predictions.filter(
        (col(LABEL_COLUMN) == 1) &
        (col("final_prediction") == 0)
    ).count()

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

    validation_results.append(
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
# Display validation results
# ==========================

print()

print(
    f"{'Threshold':<12}"
    f"{'TP':<8}"
    f"{'FP':<8}"
    f"{'FN':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 72)

for result in validation_results:

    (
        threshold,
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    ) = result

    print(
        f"{threshold:<12.2f}"
        f"{tp:<8}"
        f"{fp:<8}"
        f"{fn:<8}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )


# ==========================
# Best threshold
# ==========================

best_result = max(
    validation_results,
    key=lambda x: x[6]
)

(
    best_threshold,
    validation_tp,
    validation_fp,
    validation_fn,
    validation_precision,
    validation_recall,
    validation_f1
) = best_result


print()
print("==========================")
print("Best Validation Threshold")
print("==========================")

print(f"Threshold: {best_threshold:.2f}")
print(f"TP:        {validation_tp}")
print(f"FP:        {validation_fp}")
print(f"FN:        {validation_fn}")
print(f"Precision: {validation_precision:.4f}")
print(f"Recall:    {validation_recall:.4f}")
print(f"F1:        {validation_f1:.4f}")


# ==========================
# Test predictions
# ==========================

print()
print("==========================")
print("Final Test Evaluation")
print("==========================")

test_predictions = model.transform(
    test_features
)

test_predictions = test_predictions.withColumn(
    "fraud_probability",
    vector_to_array(
        col("probability")
    )[1]
)


# ==========================
# Apply validation threshold
# ==========================

test_predictions = test_predictions.withColumn(
    "final_prediction",
    when(
        col("fraud_probability") >= best_threshold,
        1.0
    ).otherwise(0.0)
)


# ==========================
# Confusion Matrix
# ==========================

print()
print("Confusion Matrix")

confusion_matrix = (
    test_predictions
    .groupBy(
        LABEL_COLUMN,
        "final_prediction"
    )
    .count()
    .orderBy(
        LABEL_COLUMN,
        "final_prediction"
    )
)

confusion_matrix.show()


# ==========================
# Test metrics
# ==========================

test_tp = test_predictions.filter(
    (col(LABEL_COLUMN) == 1) &
    (col("final_prediction") == 1)
).count()

test_fp = test_predictions.filter(
    (col(LABEL_COLUMN) == 0) &
    (col("final_prediction") == 1)
).count()

test_fn = test_predictions.filter(
    (col(LABEL_COLUMN) == 1) &
    (col("final_prediction") == 0)
).count()


if test_tp + test_fp > 0:
    test_precision = (
        test_tp / (test_tp + test_fp)
    )
else:
    test_precision = 0.0


if test_tp + test_fn > 0:
    test_recall = (
        test_tp / (test_tp + test_fn)
    )
else:
    test_recall = 0.0


if test_precision + test_recall > 0:
    test_f1 = (
        2 * test_precision * test_recall
        / (test_precision + test_recall)
    )
else:
    test_f1 = 0.0


# ==========================
# PR-AUC
# ==========================

pr_evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COLUMN,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

test_pr_auc = pr_evaluator.evaluate(
    test_predictions
)


# ==========================
# ROC-AUC
# ==========================

roc_evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COLUMN,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

test_roc_auc = roc_evaluator.evaluate(
    test_predictions
)


# ==========================
# Final metrics
# ==========================

print()
print("==========================")
print("FINAL TEST METRICS")
print("==========================")

print(f"Threshold:   {best_threshold:.2f}")
print(f"TP:          {test_tp}")
print(f"FP:          {test_fp}")
print(f"FN:          {test_fn}")
print(f"Precision:   {test_precision:.4f}")
print(f"Recall:      {test_recall:.4f}")
print(f"F1 Score:    {test_f1:.4f}")
print(f"PR-AUC:      {test_pr_auc:.4f}")
print(f"ROC-AUC:     {test_roc_auc:.4f}")


# ==========================
# Stop Spark
# ==========================

spark.stop()

os._exit(0)