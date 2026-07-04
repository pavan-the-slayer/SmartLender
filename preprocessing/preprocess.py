"""
Data Pre-processing stage of the SmartLender pipeline:

    Exploratory Data Analysis -> One-Hot Encoding -> Feature Selection -> Scaling Transform

Produces:
  - artifacts/scaler_transform.pkl   (fitted StandardScaler)
  - artifacts/feature_columns.json   (final feature column order, needed by the predict API)
  - artifacts/X_train.npy / X_test.npy / y_train.npy / y_test.npy (for training.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "loan_data.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"No dataset found at {RAW_CSV}. Run "
            "`python data/generate_synthetic_data.py` first, or drop your own "
            "loan_data.csv there."
        )
    return pd.read_csv(RAW_CSV)


def run_eda(df: pd.DataFrame) -> dict:
    """Lightweight EDA summary — printed and returned as a dict for logging."""
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_values": df.isna().sum().to_dict(),
        "target_balance": df[TARGET_COL].value_counts(normalize=True).to_dict(),
        "numeric_describe": df[NUMERIC_COLS].describe().to_dict(),
    }
    print("=== EDA Summary ===")
    print(f"Rows: {report['n_rows']}, Cols: {report['n_cols']}")
    print("Missing values per column:")
    for col, n in report["missing_values"].items():
        if n:
            print(f"  {col}: {n}")
    print("Target balance (Loan_Status):", report["target_balance"])
    return report


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Selection Engineering: drop the ID column and any single-value /
    zero-variance columns produced by one-hot encoding.
    """
    df = df.drop(columns=[ID_COL], errors="ignore")
    nunique = df.nunique()
    zero_variance = nunique[nunique <= 1].index.tolist()
    if zero_variance:
        print(f"Dropping zero-variance columns: {zero_variance}")
        df = df.drop(columns=zero_variance)
    return df


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def run() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    df = load_raw()
    run_eda(df)

    df = impute_missing(df)
    df = one_hot_encode(df)
    df = select_features(df)

    y = (df[TARGET_COL] == "Y").astype(int)
    X = df.drop(columns=[TARGET_COL])

    feature_columns = X.columns.tolist()
    with open(ARTIFACTS / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, ARTIFACTS / "scaler_transform.pkl")
    np.save(ARTIFACTS / "X_train.npy", X_train_scaled)
    np.save(ARTIFACTS / "X_test.npy", X_test_scaled)
    np.save(ARTIFACTS / "y_train.npy", y_train.to_numpy())
    np.save(ARTIFACTS / "y_test.npy", y_test.to_numpy())

    print(f"\nSaved scaler_transform.pkl and train/test splits to {ARTIFACTS}")
    print(f"Final feature count: {len(feature_columns)}")


if __name__ == "__main__":
    run()
