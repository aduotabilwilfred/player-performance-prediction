import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(repo_root, "configs"))
from read_config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load all config values
data_path = get_config("data_path")
processed_data_dir = get_config("processed_data_dir")
train_ratio = get_config("train_ratio")
val_ratio = get_config("val_ratio")
test_ratio = get_config("test_ratio")
random_seed = get_config("random_seed")
remove_zero_ratings = get_config("remove_zero_ratings")
dedup_column = get_config("dedup_column")
target = get_config("target")
numeric_features = get_config("numeric_features")
categorical_features = get_config("categorical_features")
engineered_features = get_config("engineered_features")


def main(data_path):
    OUTPUT_DIR = os.path.join(repo_root, processed_data_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use mutable copy so we can extend with engineered features
    NUMERIC_FEATURES = list(numeric_features)
    CATEGORICAL_FEATURES = list(categorical_features)
    ENGINEERED_FEATURES = list(engineered_features)
    TARGET = target

    logger.info("Feature Definition:")
    logger.info(f"  Numeric: {len(NUMERIC_FEATURES)} features")
    logger.info(f"  Categorical: {len(CATEGORICAL_FEATURES)} features")
    logger.info(f"  Target: {TARGET}")

    # Load and deduplicate data
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=dedup_column)

    if remove_zero_ratings:
        df = df[df[TARGET] > 0].reset_index(drop=True)
    logger.info(f"Dataset loaded: {len(df)} rows after dedup and zero-rating filter")

    # === MISSING VALUES CHECK ===
    missing_numeric = df[NUMERIC_FEATURES].isnull().sum()
    if missing_numeric.sum() > 0:
        logger.info(f"Numeric missing:\n{missing_numeric[missing_numeric > 0]}")
    else:
        logger.info("No missing numeric values")

    missing_cat = df[CATEGORICAL_FEATURES].isnull().sum()
    if missing_cat.sum() > 0:
        logger.info(f"Categorical missing:\n{missing_cat[missing_cat > 0]}")
    else:
        logger.info("No missing categorical values")

    # === FEATURE ENGINEERING ===
    logger.info("Feature Engineering")
    df["pass_accuracy"] = (df["successful_passes"] / (df["total_passes"] + 1)) * 100
    df["shot_accuracy"] = (df["shots_on_target"] / (df["shots"] + 1)) * 100
    df["dribble_success"] = (df["successful_dribbles"] / (df["dribbles_attempted"] + 1)) * 100
    df["goals_per_minute"] = (df["goals"] / (df["minutes_played"] + 1)) * 90
    df["distance_per_minute"] = df["distance_covered_km"] / (df["minutes_played"] + 1)

    NUMERIC_FEATURES.extend(ENGINEERED_FEATURES)
    logger.info(f"  Created: {len(ENGINEERED_FEATURES)} engineered features")
    logger.info(f"  Total numeric: {len(NUMERIC_FEATURES)}")

    # === DATA SPLITTING ===
    logger.info("Splitting data (before any preprocessing)")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=random_seed, shuffle=True
    )
    val_test_split = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_split, random_state=random_seed, shuffle=True
    )

    logger.info("Split Results:")
    logger.info(f"  Train: {len(X_train):6,} samples ({len(X_train) / len(X) * 100:.1f}%)")
    logger.info(f"  Val:   {len(X_val):6,} samples ({len(X_val) / len(X) * 100:.1f}%)")
    logger.info(f"  Test:  {len(X_test):6,} samples ({len(X_test) / len(X) * 100:.1f}%)")
    logger.info(f"  Total: {len(X):6,} samples")

    # === NUMERICAL PREPROCESSING ===
    logger.info("Preprocessing NUMERICAL Features")

    # Impute missing values using train medians
    numeric_medians = X_train[NUMERIC_FEATURES].median()
    X_train[NUMERIC_FEATURES] = X_train[NUMERIC_FEATURES].fillna(numeric_medians)
    X_val[NUMERIC_FEATURES] = X_val[NUMERIC_FEATURES].fillna(numeric_medians)
    X_test[NUMERIC_FEATURES] = X_test[NUMERIC_FEATURES].fillna(numeric_medians)
    logger.info("  Imputed missing values with TRAIN medians")

    # Scale using train statistics only
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_FEATURES])
    X_train[NUMERIC_FEATURES] = scaler.transform(X_train[NUMERIC_FEATURES])
    X_val[NUMERIC_FEATURES] = scaler.transform(X_val[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])
    logger.info("  StandardScaler fitted on TRAIN, applied to all splits")

    # === CATEGORICAL PREPROCESSING ===
    logger.info("Preprocessing CATEGORICAL Features (One-Hot Encoding)")

    X_train_cat = pd.get_dummies(
        X_train[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False, prefix=["pos", "stage", "result"]
    )
    cat_columns = X_train_cat.columns.tolist()

    X_val_cat = pd.get_dummies(
        X_val[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False, prefix=["pos", "stage", "result"]
    )
    X_test_cat = pd.get_dummies(
        X_test[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False, prefix=["pos", "stage", "result"]
    )

    # Align columns across splits (handle unseen categories)
    for col in cat_columns:
        if col not in X_val_cat.columns:
            X_val_cat[col] = 0
        if col not in X_test_cat.columns:
            X_test_cat[col] = 0

    X_val_cat = X_val_cat[cat_columns]
    X_test_cat = X_test_cat[cat_columns]
    logger.info(f"  One-hot encoded: {len(cat_columns)} categorical columns")

    # === COMBINE FEATURES ===
    logger.info("Combining Numerical + Categorical Features")

    X_train_processed = pd.concat(
        [X_train.drop(CATEGORICAL_FEATURES, axis=1).reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1
    )
    X_val_processed = pd.concat(
        [X_val.drop(CATEGORICAL_FEATURES, axis=1).reset_index(drop=True), X_val_cat.reset_index(drop=True)], axis=1
    )
    X_test_processed = pd.concat(
        [X_test.drop(CATEGORICAL_FEATURES, axis=1).reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1
    )

    logger.info(f"  Train: {X_train_processed.shape} | Val: {X_val_processed.shape} | Test: {X_test_processed.shape}")

    # Add target variable
    train_df = pd.concat([X_train_processed.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val_processed.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_processed.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # === DATA QUALITY CHECKS ===
    logger.info("Data Quality Checks")
    train_missing = train_df.isnull().sum().sum()
    val_missing = val_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()
    logger.info(f"  Missing values — Train: {train_missing} | Val: {val_missing} | Test: {test_missing}")
    logger.info(
        f"  Target mean — Train: {train_df[TARGET].mean():.3f} | Val: {val_df[TARGET].mean():.3f} | Test: {test_df[TARGET].mean():.3f}"
    )
    logger.info("All quality checks passed!")

    # === SAVE OUTPUTS ===
    logger.info("Saving Processed Datasets")
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "validation.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Train saved:      {train_path} | Shape: {train_df.shape} | Size: {os.path.getsize(train_path) / 1024**2:.2f} MB"
    )
    logger.info(
        f"Validation saved: {val_path} | Shape: {val_df.shape} | Size: {os.path.getsize(val_path) / 1024**2:.2f} MB"
    )
    logger.info(
        f"Test saved:       {test_path} | Shape: {test_df.shape} | Size: {os.path.getsize(test_path) / 1024**2:.2f} MB"
    )


if __name__ == "__main__":
    main(data_path)
