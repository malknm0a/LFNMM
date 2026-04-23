# src/preprocessing.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import *

def load_raw_data():
    """Load and combine all CSV files"""
    import glob, os
    csv_files      = glob.glob(os.path.join(RAW_PATH, "*.csv"))
    labeled_list   = []
    unlabeled_list = []

    for file in csv_files:
        df       = pd.read_csv(file, low_memory=False)
        filename = os.path.basename(file)
        df["SOURCE_FILE"] = filename
        if TARGET in df.columns:
            labeled_list.append(df)
        else:
            unlabeled_list.append(df)

    labeled_df   = pd.concat(labeled_list,   ignore_index=True)
    unlabeled_df = pd.concat(unlabeled_list, ignore_index=True)

    print(f"✅ Loaded {len(labeled_list)} labeled files")
    print(f"✅ Loaded {len(unlabeled_list)} unlabeled files")
    return labeled_df, unlabeled_df


def drop_columns(df):
    """Drop irrelevant columns"""
    cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols, inplace=True)
    print(f"✅ Dropped {len(cols)} columns → {df.shape[1]} remaining")
    return df


def drop_corrupted_rows(df):
    """Drop missing targets and corrupted Unknown well"""
    before = len(df)

    # Drop missing targets
    df.dropna(subset=[TARGET], inplace=True)

    # Apply class mapping
    df[TARGET_CLEAN] = df[TARGET].map(LITHO_MAP)
    df.drop(columns=[TARGET], inplace=True)

    # Fill null WELL with Unknown
    df["WELL"]      = df["WELL"].fillna("Unknown")
    df["GROUP"]     = df["GROUP"].fillna("Unknown")
    df["FORMATION"] = df["FORMATION"].fillna("Unknown")

    # Drop entire Unknown well — corrupted data!
    unknown_rows = (df["WELL"] == "Unknown").sum()
    df = df[df["WELL"] != "Unknown"]
    print(f"✅ Dropped null targets  : 693 rows")
    print(f"✅ Dropped Unknown well  : {unknown_rows:,} corrupted rows")
    print(f"   Clean rows           : {len(df):,}")
    return df

def impute_missing(df):
    """Impute missing values per well"""
    # Numerical → median per well
    df[NUMERICAL_FEATURES] = df.groupby("WELL")[NUMERICAL_FEATURES]\
        .transform(lambda x: x.fillna(x.median()))
    df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES]\
        .fillna(df[NUMERICAL_FEATURES].median())

    # Categorical → mode per well
    for col in ["FORMATION", "GROUP"]:
        df[col] = df.groupby("WELL")[col]\
            .transform(lambda x: x.fillna(
                x.mode()[0] if not x.mode().empty else "Unknown"))

    print(f"✅ Missing values after imputation : {df.isnull().sum().sum()}")
    return df


def log_transform(df):
    """Log transform skewed features"""
    for col in LOG_TRANSFORM_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(np.abs(df[col])) * np.sign(df[col])

    # Fix DRHO outliers
    p1  = df["DRHO"].quantile(0.01)
    p99 = df["DRHO"].quantile(0.99)
    df["DRHO"] = df["DRHO"].clip(p1, p99)

    print(f"✅ Log transform applied to : {LOG_TRANSFORM_FEATURES}")
    return df


def encode_features(df):
    """Encode categorical features and target"""
    encoders = {}

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    df["TARGET"] = le_target.fit_transform(df[TARGET_CLEAN])
    encoders["target"] = le_target

    # Save encoders
    joblib.dump(encoders,   f"{MODEL_PATH}/encoders.pkl")
    joblib.dump(le_target,  f"{MODEL_PATH}/le_target.pkl")

    print(f"✅ Encoded : {CATEGORICAL_FEATURES}")
    print(f"✅ Target classes : {list(le_target.classes_)}")
    return df, encoders


def split_data(df):
    """Split into train and test sets"""
    X = df[FEATURES_13]
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    joblib.dump((X_train, X_test, y_train, y_test),
                f"{MODEL_PATH}/train_test_split.pkl")

    print(f"✅ Train : {len(X_train):,} | Test : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def run_preprocessing():
    """Run full preprocessing pipeline"""
    print("=" * 50)
    print("   🔧 PREPROCESSING PIPELINE")
    print("=" * 50)

    # Load
    labeled_df, unlabeled_df = load_raw_data()

    # Process
    df = drop_columns(labeled_df)
    df = drop_corrupted_rows(df)
    df = impute_missing(df)
    df = log_transform(df)
    df, encoders = encode_features(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Save cleaned data
    df.to_csv(CLEANED_FILE, index=False)
    print(f"\n💾 Saved → cleaned_data.csv")

    print("\n✅ Preprocessing complete!")
    print("=" * 50)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
