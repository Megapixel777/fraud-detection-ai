import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

# ==========================
# Force Spark to use current
# Python environment
# ==========================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForest
from pyspark.ml.functions import vector_to_array


# ==========================
# Configuration
# ==========================

DATA_PATH = "data/silver/transactions_clean"

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
    "V3",
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
    0.50,
]


# ==========================
# Load data
# ==========================

print()
print("==========================")
print("Loading data")
print("==========================")

csv_files = sorted(
    [
        os.path.join(DATA_PATH, file)
        for file in os.listdir(DATA_PATH)
        if file.endswith(".csv")
    ]
)

print(f"CSV files found: {len(csv_files)}")

for file in csv_files:
    print(f"  - {file}")


df = pd.concat(
    [
        pd.read_csv(
            file,
            usecols=TOP_10_FEATURES + [LABEL_COLUMN],
        )
        for file in csv_files
    ],
    ignore_index=True,
)

df = df.dropna()

print(f"Records: {len(df):,}")


# ==========================
# Create ONE common
# stratified split
# ==========================

print()
print("==========================")
print("Creating common split")
print("==========================")

train_parts = []
validation_parts = []
test_parts = []

for class_value in [0, 1]:

    class_df = df[
        df[LABEL_COLUMN] == class_value
    ].sample(
        frac=1,
        random_state=RANDOM_SEED,
    )

    n = len(class_df)

    train_end = int(n * 0.80)
    validation_end = int(n * 0.90)

    train_parts.append(
        class_df.iloc[:train_end]
    )

    validation_parts.append(
        class_df.iloc[
            train_end:validation_end
        ]
    )

    test_parts.append(
        class_df.iloc[validation_end:]
    )


train_df = pd.concat(
    train_parts
).sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


validation_df = pd.concat(
    validation_parts
).sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


test_df = pd.concat(
    test_parts
).sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")


# ==========================
# Prepare sklearn data
# ==========================

X_train = train_df[TOP_10_FEATURES]
y_train = train_df[LABEL_COLUMN]

X_validation = validation_df[TOP_10_FEATURES]
y_validation = validation_df[LABEL_COLUMN]

X_test = test_df[TOP_10_FEATURES]
y_test = test_df[LABEL_COLUMN]


# ==========================
# Train sklearn model
# ==========================

print()
print("==========================")
print("Training scikit-learn")
print("==========================")

sklearn_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    criterion="gini",
    max_features="sqrt",
    bootstrap=True,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

sklearn_model.fit(
    X_train,
    y_train,
)

print("scikit-learn training completed.")


# ==========================
# Train Spark model
# ==========================

print()
print("==========================")
print("Training Spark")
print("==========================")

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("SparkVsSklearnComparison")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)


spark_train = spark.createDataFrame(train_df)
spark_validation = spark.createDataFrame(validation_df)
spark_test = spark.createDataFrame(test_df)


assembler = VectorAssembler(
    inputCols=TOP_10_FEATURES,
    outputCol="features",
)


spark_train_features = assembler.transform(
    spark_train
)

spark_validation_features = assembler.transform(
    spark_validation
)

spark_test_features = assembler.transform(
    spark_test
)


spark_model = SparkRandomForest(
    labelCol=LABEL_COLUMN,
    featuresCol="features",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    numTrees=100,
    seed=RANDOM_SEED,
)

spark_model = spark_model.fit(
    spark_train_features
)

print("Spark training completed.")


# ==========================
# Validation probabilities
# ==========================

sklearn_validation_probability = (
    sklearn_model.predict_proba(
        X_validation
    )[:, 1]
)


spark_validation_predictions = (
    spark_model.transform(
        spark_validation_features
    )
    .withColumn(
        "fraud_probability",
        vector_to_array(
            col("probability")
        )[1],
    )
)


spark_validation_probability = [
    row["fraud_probability"]
    for row in (
        spark_validation_predictions
        .select("fraud_probability")
        .collect()
    )
]


# ==========================
# Find best threshold
# ==========================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return precision, recall, f1


print()
print("==========================")
print("Threshold comparison")
print("==========================")


print()
print("Spark:")

best_spark_threshold = None
best_spark_f1 = -1

for threshold in THRESHOLDS:

    precision, recall, f1 = (
        calculate_metrics(
            y_validation.to_numpy(),
            pd.Series(
                spark_validation_probability
            ).to_numpy(),
            threshold,
        )
    )

    print(
        f"Threshold={threshold:.2f} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f}"
    )

    if f1 > best_spark_f1:
        best_spark_f1 = f1
        best_spark_threshold = threshold


print()
print("scikit-learn:")

best_sklearn_threshold = None
best_sklearn_f1 = -1

for threshold in THRESHOLDS:

    precision, recall, f1 = (
        calculate_metrics(
            y_validation.to_numpy(),
            sklearn_validation_probability,
            threshold,
        )
    )

    print(
        f"Threshold={threshold:.2f} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f}"
    )

    if f1 > best_sklearn_f1:
        best_sklearn_f1 = f1
        best_sklearn_threshold = threshold


# ==========================
# Test predictions
# ==========================

print()
print("==========================")
print("Final test comparison")
print("==========================")


# ---------- sklearn ----------

sklearn_test_probability = (
    sklearn_model.predict_proba(
        X_test
    )[:, 1]
)


sklearn_test_prediction = (
    sklearn_test_probability
    >= best_sklearn_threshold
).astype(int)


# ---------- Spark ----------

spark_test_predictions = (
    spark_model.transform(
        spark_test_features
    )
    .withColumn(
        "fraud_probability",
        vector_to_array(
            col("probability")
        )[1],
    )
)


spark_test_probability = pd.Series(
    [
        row["fraud_probability"]
        for row in (
            spark_test_predictions
            .select("fraud_probability")
            .collect()
        )
    ]
).to_numpy()


spark_test_prediction = (
    spark_test_probability
    >= best_spark_threshold
).astype(int)


# ==========================
# Metrics
# ==========================

sklearn_precision = precision_score(
    y_test,
    sklearn_test_prediction,
    zero_division=0,
)

sklearn_recall = recall_score(
    y_test,
    sklearn_test_prediction,
    zero_division=0,
)

sklearn_f1 = f1_score(
    y_test,
    sklearn_test_prediction,
    zero_division=0,
)

sklearn_pr_auc = average_precision_score(
    y_test,
    sklearn_test_probability,
)

sklearn_roc_auc = roc_auc_score(
    y_test,
    sklearn_test_probability,
)


spark_precision = precision_score(
    y_test,
    spark_test_prediction,
    zero_division=0,
)

spark_recall = recall_score(
    y_test,
    spark_test_prediction,
    zero_division=0,
)

spark_f1 = f1_score(
    y_test,
    spark_test_prediction,
    zero_division=0,
)

spark_pr_auc = average_precision_score(
    y_test,
    spark_test_probability,
)

spark_roc_auc = roc_auc_score(
    y_test,
    spark_test_probability,
)


# ==========================
# Display
# ==========================

print()
print("==========================")
print("SPARK")
print("==========================")

print(f"Threshold: {best_spark_threshold:.2f}")
print(f"Precision: {spark_precision:.4f}")
print(f"Recall:    {spark_recall:.4f}")
print(f"F1:        {spark_f1:.4f}")
print(f"PR-AUC:    {spark_pr_auc:.4f}")
print(f"ROC-AUC:   {spark_roc_auc:.4f}")


print()
print("==========================")
print("SCIKIT-LEARN")
print("==========================")

print(f"Threshold: {best_sklearn_threshold:.2f}")
print(f"Precision: {sklearn_precision:.4f}")
print(f"Recall:    {sklearn_recall:.4f}")
print(f"F1:        {sklearn_f1:.4f}")
print(f"PR-AUC:    {sklearn_pr_auc:.4f}")
print(f"ROC-AUC:   {sklearn_roc_auc:.4f}")


# ==========================
# Prediction agreement
# ==========================

agreement = (
    spark_test_prediction
    == sklearn_test_prediction
).mean()


print()
print("==========================")
print("PREDICTION AGREEMENT")
print("==========================")

print(
    f"Agreement: {agreement:.4%}"
)


# ==========================
# Stop Spark
# ==========================

spark.stop()