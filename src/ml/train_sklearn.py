import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ==========================
# Configuration
# ==========================

DATA_PATH = "data/silver/transactions_clean"

MODEL_PATH = "models/fraud_random_forest_top10_sklearn.joblib"

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
    glob.glob(
        os.path.join(DATA_PATH, "*.csv")
    )
)

print(f"CSV files found: {len(csv_files)}")

for file in csv_files:
    print(f"  - {file}")

df = pd.concat(
    [
        pd.read_csv(
            file,
            usecols=TOP_10_FEATURES + [LABEL_COLUMN]
        )
        for file in csv_files
    ],
    ignore_index=True
)

df = df.dropna()

print(f"Records: {len(df):,}")


# ==========================
# Stratified split
# ==========================

print()
print("==========================")
print("Creating stratified split")
print("==========================")

train_parts = []
validation_parts = []
test_parts = []

for class_value in [0, 1]:

    class_df = df[
        df[LABEL_COLUMN] == class_value
    ].sample(
        frac=1,
        random_state=RANDOM_SEED
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
    random_state=RANDOM_SEED
)

validation_df = pd.concat(
    validation_parts
).sample(
    frac=1,
    random_state=RANDOM_SEED
)

test_df = pd.concat(
    test_parts
).sample(
    frac=1,
    random_state=RANDOM_SEED
)

print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")


# ==========================
# Features / labels
# ==========================

X_train = train_df[TOP_10_FEATURES]
y_train = train_df[LABEL_COLUMN]

X_validation = validation_df[TOP_10_FEATURES]
y_validation = validation_df[LABEL_COLUMN]

X_test = test_df[TOP_10_FEATURES]
y_test = test_df[LABEL_COLUMN]


# ==========================
# Train Random Forest
# ==========================

print()
print("==========================")
print("Training Random Forest")
print("==========================")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    criterion="gini",
    max_features="sqrt",
    bootstrap=True,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==========================
# Validation
# ==========================

validation_probability = model.predict_proba(
    X_validation
)[:, 1]


print()
print("==========================")
print("Threshold Optimization")
print("==========================")

best_threshold = None
best_f1 = -1

for threshold in THRESHOLDS:

    prediction = (
        validation_probability >= threshold
    ).astype(int)

    tp = (
        (y_validation == 1) &
        (prediction == 1)
    ).sum()

    fp = (
        (y_validation == 0) &
        (prediction == 1)
    ).sum()

    fn = (
        (y_validation == 1) &
        (prediction == 0)
    ).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print(
        f"Threshold={threshold:.2f} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print()
print("==========================")
print("Best threshold")
print("==========================")

print(f"Threshold: {best_threshold:.2f}")
print(f"F1:        {best_f1:.4f}")


# ==========================
# Final test
# ==========================

test_probability = model.predict_proba(
    X_test
)[:, 1]

test_prediction = (
    test_probability >= best_threshold
).astype(int)

tp = (
    (y_test == 1) &
    (test_prediction == 1)
).sum()

fp = (
    (y_test == 0) &
    (test_prediction == 1)
).sum()

fn = (
    (y_test == 1) &
    (test_prediction == 0)
).sum()

precision = (
    tp / (tp + fp)
    if tp + fp > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if tp + fn > 0
    else 0.0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0.0
)


print()
print("==========================")
print("FINAL TEST METRICS")
print("==========================")

print(f"Threshold: {best_threshold:.2f}")
print(f"TP:        {tp}")
print(f"FP:        {fp}")
print(f"FN:        {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ==========================
# Save model
# ==========================

print()
print("==========================")
print("Saving model")
print("==========================")

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": TOP_10_FEATURES,
    },
    MODEL_PATH
)

print(f"Model saved to: {MODEL_PATH}")