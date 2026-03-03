"""
Preprocess Credit Risk dataset (Give Me Some Credit: cs-training.csv or similar).
Target: SeriousDlqin2yrs (default in 2 years). Handles missing MonthlyIncome, NumberOfDependents.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config import DATA_DIR, OUTPUT_DIR, CREDIT_RISK_CSV, RANDOM_STATE, TEST_SIZE


# Common column names for Give Me Some Credit (Kaggle)
DEFAULT_TARGET = "SeriousDlqin2yrs"
EXPECTED_COLS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_credit_data(csv_path=None):
    path = csv_path or os.path.join(DATA_DIR, CREDIT_RISK_CSV)
    if not os.path.isfile(path):
        # Try alternate names from zip
        for name in ["cs-training.csv", "GiveMeSomeCredit.csv", "cs_train.csv"]:
            p = os.path.join(DATA_DIR, name)
            if os.path.isfile(p):
                path = p
                break
    df = pd.read_csv(path)
    return df


def preprocess_credit(df: pd.DataFrame, target_col=None):
    df = df.copy()
    target_col = target_col or DEFAULT_TARGET
    if target_col not in df.columns:
        # Try first column as target if it looks binary
        cand = df.columns[0]
        if df[cand].nunique() <= 2:
            target_col = cand
        else:
            raise ValueError(f"Target column {DEFAULT_TARGET} not found. Columns: {list(df.columns)}")

    y = df[target_col].astype(int)
    df = df.drop(columns=[target_col])

    # Drop Unnamed:0 if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Fill missing numeric
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())

    X = df
    feature_names = X.columns.tolist()
    return X, y, feature_names


def run(output_subdir="credit_risk"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out, exist_ok=True)

    df = load_credit_data()
    X, y, feature_names = preprocess_credit(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save(os.path.join(out, "X_train.npy"), X_train)
    np.save(os.path.join(out, "y_train.npy"), y_train.values)
    np.save(os.path.join(out, "X_test.npy"), X_test)
    np.save(os.path.join(out, "y_test.npy"), y_test.values)
    joblib.dump(scaler, os.path.join(out, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(out, "feature_names.pkl"))

    print(f"Credit risk preprocessing done. Train {X_train.shape[0]}, test {X_test.shape[0]}, features {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run()
