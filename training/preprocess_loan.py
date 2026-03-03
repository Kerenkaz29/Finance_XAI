"""
Preprocess Loan Approval dataset (loan_data_set.csv).
Target: Loan_Status (Y/N) -> binary. Handles missing values and encoding.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from config import DATA_DIR, OUTPUT_DIR, LOAN_APPROVAL_CSV, RANDOM_STATE, TEST_SIZE


def load_loan_data(csv_path=None):
    path = csv_path or os.path.join(DATA_DIR, LOAN_APPROVAL_CSV)
    df = pd.read_csv(path)
    return df


def preprocess_loan(df: pd.DataFrame):
    df = df.copy()
    # Drop ID
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Target
    target_col = "Loan_Status"
    if target_col not in df.columns:
        raise ValueError(f"Expected column {target_col}")
    y = (df[target_col].str.strip().str.upper() == "Y").astype(int)
    df = df.drop(columns=[target_col])

    # Fill numeric missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    # Fill categorical with mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if len(df[c].mode()) else "")

    # Encode categoricals
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_dict[c] = le

    X = df
    feature_names = X.columns.tolist()
    return X, y, feature_names, le_dict


def run(output_subdir="loan"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out, exist_ok=True)

    df = load_loan_data()
    X, y, feature_names, le_dict = preprocess_loan(df)

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
    joblib.dump(le_dict, os.path.join(out, "label_encoders.pkl"))

    print(f"Loan preprocessing done. Train {X_train.shape[0]}, test {X_test.shape[0]}, features {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names, scaler, le_dict


if __name__ == "__main__":
    run()
