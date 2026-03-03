"""
Preprocess American Bankruptcy dataset (american_bankruptcy.csv).
Target: status_label (alive/bankrupt). Uses latest year per company or aggregate.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config import DATA_DIR, OUTPUT_DIR, BANKRUPTCY_CSV, RANDOM_STATE, TEST_SIZE


def load_bankruptcy_data(csv_path=None):
    path = csv_path or os.path.join(DATA_DIR, BANKRUPTCY_CSV)
    df = pd.read_csv(path)
    return df


# 10 key financial ratios for bankruptcy prediction (based on Altman Z-score + research)
BANKRUPTCY_FEATURES = [
    "Working Capital to Total Assets",           # Altman Z1: short-term liquidity cushion
    "Retained Earnings to Total Assets",          # Altman Z2: cumulative profitability
    "ROA(C) before interest and depreciation before interest",  # Altman Z3: operating efficiency
    "Net worth/Assets",                           # Altman Z4: equity vs assets
    "Total Asset Turnover",                       # Altman Z5: asset efficiency / revenue generation
    "Debt ratio %",                               # Financial leverage (key bankruptcy signal)
    "Cash Flow to Total Assets",                  # Cash generation ability
    "Interest Coverage Ratio (Interest expense to EBIT)",  # Ability to service debt
    "Current Ratio",                              # Short-term solvency
    "Borrowing dependency",                       # Reliance on external debt
]


def preprocess_bankruptcy(df: pd.DataFrame, use_latest_only=True):
    df = df.copy()
    # Strip leading/trailing spaces from column names first
    df.columns = [c.strip() for c in df.columns]

    # Support both old (status_label) and new (Bankrupt?) column names
    if "status_label" in df.columns:
        target_col = "status_label"
        is_binary = False
    elif "Bankrupt?" in df.columns:
        target_col = "Bankrupt?"
        is_binary = True
    else:
        raise ValueError("Expected column 'status_label' or 'Bankrupt?' in dataset")

    if use_latest_only and "company_name" in df.columns and "year" in df.columns:
        df = df.sort_values("year", ascending=False).groupby("company_name", as_index=False).first()

    # Drop identifiers
    for drop_col in ["company_name", "year"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    if is_binary:
        y = df[target_col].astype(int)
    else:
        y = (df[target_col].str.strip().str.lower() == "bankrupt").astype(int)
    df = df.drop(columns=[target_col])

    # Keep only selected features (those present in the dataset)
    selected = [f for f in BANKRUPTCY_FEATURES if f in df.columns]
    if len(selected) < 3:
        # Fallback: use all numeric columns if selected features not found
        selected = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[selected]

    # Fill NaN with column median
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())

    X = df
    feature_names = X.columns.tolist()
    return X, y, feature_names


def run(output_subdir="bankruptcy"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out, exist_ok=True)

    df = load_bankruptcy_data()
    # If dataset has company_name, use latest per company
    if "company_name" in df.columns:
        X, y, feature_names = preprocess_bankruptcy(df, use_latest_only=True)
    else:
        X, y, feature_names = preprocess_bankruptcy(df, use_latest_only=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Warn if one split has only one class (some classifiers will fail)
    for name, yy in [("train", y_train), ("test", y_test)]:
        n_classes = yy.nunique()
        if n_classes < 2:
            print(f"  Warning: {name} set has only {n_classes} class(es). Training may skip or fail for this dataset.")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save(os.path.join(out, "X_train.npy"), X_train)
    np.save(os.path.join(out, "y_train.npy"), y_train.values)
    np.save(os.path.join(out, "X_test.npy"), X_test)
    np.save(os.path.join(out, "y_test.npy"), y_test.values)
    joblib.dump(scaler, os.path.join(out, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(out, "feature_names.pkl"))

    print(f"Bankruptcy preprocessing done. Train {X_train.shape[0]}, test {X_test.shape[0]}, features {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run()
