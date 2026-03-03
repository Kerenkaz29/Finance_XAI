"""
FastAPI application: prediction + XAI endpoints.
Run: python -m uvicorn main:app --reload --port 8000
"""
import os
import traceback
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder

from config import DATASETS, CORS_ORIGINS, EXPERTISE_LEVELS, DEFAULT_MODEL_TYPE, TRAINING_OUTPUT
from model_loader import (
    get_model,
    get_scaler,
    get_feature_names,
    predict,
    predict_proba,
)
from xai_services import (
    get_shap_explanation,
    get_lime_explanation,
    get_dice_counterfactuals,
)
from xai_plots import save_shap_plot, save_lime_plot
from config import BASE_DIR, BASE_URL

# ----- Auto-download model weights from Google Drive if missing -----

_models_ready = False
_download_progress = {"done": 0, "total": 0, "current": ""}

# Hardcoded file IDs extracted from the shared Drive folder
_DRIVE_FILES = {
    "bankruptcy": {
        "feature_names.pkl": "10uSsrH_P8IZq_0tqrU-gwVWDm70TBh4Y",
        "model_gb.pkl":      "1LCrROg7vW4_bOjgJGddf3ow5TN6oGFMe",
        "model_lr.pkl":      "17gtDesvEcYEB_FD2Q0h9VOYX4gi06GLk",
        "model_mlp.pt":      "1MwRdEb6HB_qWcyAfa18jhDh4gHmMd5Up",
        "model_rf.pkl":      "1vi0lSyrvFX61gFPJrzEa-KtfLfyF7ilq",
        "scaler.pkl":        "1nSX8ygZd0UtrIiHxBf-xJigX3GQNWIeN",
        "X_test.npy":        "1yI5oQK8Jijo7K9syxc4vvyB_8hpk98Ds",
        "X_train.npy":       "14C3Un1L85LWFhgKaTnJuhSbtKeZMrIn6",
        "y_test.npy":        "15P1d4v_0I97uac_NEUm8I9pTpK1SWD6a",
        "y_train.npy":       "1rtuoPIqZQqycSYAtgPM1AamBf7jhPWue",
    },
    "credit_risk": {
        "feature_names.pkl": "1v3cviIXZUfHKqMWHonm6Zl-Pq-LGMTO4",
        "model_gb.pkl":      "1mlO8wAyyWjzbTg6bBPpdp6Ewei0krr8v",
        "model_lr.pkl":      "1cNAFemBQYtPMEfw_U5K8TAf-hUBkl3RW",
        "model_mlp.pt":      "1DwB9EdMUVyflYIjot9Sc2XLKiRo3Xqq1",
        "model_rf.pkl":      "1lyBj--X5wGLqZUPLtRCar9zlKZZmTJAM",
        "scaler.pkl":        "1308UOJa51zWDGz9Y0jWN5bCzUsRSwLht",
        "X_test.npy":        "1SgpCVsoJl4qXlX7avOfo4pnkGU7iv129",
        "X_train.npy":       "1jF48tJlATPYxCJE2DLEGyIu53euNgAqk",
        "y_test.npy":        "18ojaQHwlB5ImYUu9w6VCp1dME7z-xxXS",
        "y_train.npy":       "1gZWCHV-UncvAv8TehilzBp9-w1jsLHjM",
    },
    "loan": {
        "feature_names.pkl":  "1DvrivGO-Siow7NL4w67OKVHzOqrpnWV0",
        "label_encoders.pkl": "150jdQ0mfkiqdZelu_H0d4QEqH2woFic6",
        "model_gb.pkl":       "1zd28l7Hhl3BNq_IHGx8kZ9j7glOqbtMX",
        "model_lr.pkl":       "1G-S-2RUSQUDnuZlzvzg1y_NfPra1IdbY",
        "model_mlp.pt":       "136dM672DkrN0f9bQz235HL6gwD78E5U7",
        "model_rf.pkl":       "1TBqgybOxjMVFHCFop8aXKfPK5xgPDqev",
        "scaler.pkl":         "1syJSPr6sKctAmLlGZoW6tI9C39svnm-2",
        "X_test.npy":         "1y_su0v8_NiFoFkTCJZsEDb333eq7Tk6T",
        "X_train.npy":        "1CSXYToDnbLJxqaq1WDg_fPjKvpwAO95L",
        "y_test.npy":         "1izq-twbLJDm5vRJa_DdPBQkyzUsRGt5r",
        "y_train.npy":        "16ilkoFU-o0um_Z8c81XLZC0IZUQ0yEUo",
    },
}


def _gdrive_download(file_id: str, dest: str):
    """Download a single Google Drive file by ID using requests (no auth needed for shared files)."""
    import requests
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    with requests.Session() as s:
        r = s.get(url, stream=True, timeout=60)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def _download_models_worker():
    """Runs in a background thread — downloads all model files then sets _models_ready."""
    global _models_ready, _download_progress
    output_dir = os.path.abspath(TRAINING_OUTPUT)
    sentinel = os.path.join(output_dir, "loan", "model_rf.pkl")

    if os.path.isfile(sentinel):
        print("[startup] Model weights already present — skipping download.")
        _models_ready = True
        return

    print("[startup] Downloading model weights from Google Drive...")
    try:
        total = sum(len(v) for v in _DRIVE_FILES.values())
        done = 0
        _download_progress = {"done": 0, "total": total, "current": ""}
        for dataset, files in _DRIVE_FILES.items():
            for filename, file_id in files.items():
                dest = os.path.join(output_dir, dataset, filename)
                if os.path.isfile(dest):
                    done += 1
                    _download_progress = {"done": done, "total": total, "current": f"{dataset}/{filename}"}
                    continue
                _download_progress = {"done": done, "total": total, "current": f"{dataset}/{filename}"}
                print(f"[startup] ({done+1}/{total}) {dataset}/{filename} ...")
                _gdrive_download(file_id, dest)
                done += 1
                _download_progress = {"done": done, "total": total, "current": f"{dataset}/{filename}"}
        print("[startup] Download complete. Models are ready.")
    except Exception as e:
        print(f"[startup] ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        _models_ready = True


# Minimal mirror of training.config for dataset access
_DATA_DIR = os.environ.get("XAI_DATA_DIR")
if not _DATA_DIR:
    # Default to ../datasets relative to project root, or Desktop/datasets on Windows
    _DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    if os.name == "nt" and not os.path.isdir(_DATA_DIR):
        _DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "datasets")
TRAINING_DATA_DIR = _DATA_DIR
LOAN_APPROVAL_CSV = "loan_data_set.csv"
BANKRUPTCY_CSV = "american_bankruptcy.csv"
CREDIT_RISK_CSV_CANDIDATES = ["cs-training.csv", "GiveMeSomeCredit.csv", "cs_train.csv"]

# Pre-generated explanation images (cvision-style)
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

import threading
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):
    threading.Thread(target=_download_models_worker, daemon=True).start()
    yield

app = FastAPI(
    title="XAI Financial Services API",
    description="Prediction and explainability (SHAP, LIME, DiCE) for financial models.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Request/Response schemas -----

class PredictionRequest(BaseModel):
    dataset: Literal["loan", "bankruptcy", "credit_risk"] = "loan"
    features: List[float] = Field(..., description="Feature vector in model order")
    model_type: Optional[Literal["rf", "gb", "lr", "mlp"]] = None


class PredictionResponse(BaseModel):
    prediction: int  # 0 = Deny / Not approved / Not bankrupt, 1 = Approve / etc.
    prediction_label: str  # "Approved" / "Denied" etc.
    probability: float
    dataset: str


class XAIRequest(BaseModel):
    dataset: Literal["loan", "bankruptcy", "credit_risk"] = "loan"
    features: List[float]
    expertise: Literal["expert", "non_expert"] = "non_expert"
    method: Literal["SHAP", "LIME", "DiCE"]
    model_type: Optional[Literal["rf", "gb", "lr", "mlp"]] = None


class LoanSampleResponse(BaseModel):
    loan_id: str
    loan_status: Optional[str]
    raw: dict
    features: List[float]
    feature_names: List[str]


class BankruptcySampleResponse(BaseModel):
    company_name: str
    year: Optional[int]
    status_label: Optional[str]
    raw: dict
    features: List[float]
    feature_names: List[str]


class CreditSampleResponse(BaseModel):
    index: int
    serious_dlq: Optional[int]
    raw: dict
    features: List[float]
    feature_names: List[str]


# ----- Helpers -----

def _load_background_data(dataset: str):
    """Load a small background set for SHAP/LIME (from training data)."""
    base = os.path.join(os.environ.get("XAI_TRAINING_OUTPUT", os.path.join(os.path.dirname(__file__), "..", "training", "output")), dataset)
    X_path = os.path.join(base, "X_train.npy")
    if not os.path.isfile(X_path):
        return None
    return np.load(X_path)


def _build_background_from_csv(dataset: str, scaler, max_rows: int = 200, feature_names: List[str] = None):
    """Build XAI background from CSV when X_train.npy is missing. Returns (X_scaled, None) or (None, error_msg)."""
    if feature_names is None:
        try:
            feature_names = get_feature_names(dataset)
        except Exception:
            feature_names = []
    try:
        if dataset == "loan":
            path = os.path.join(TRAINING_DATA_DIR, LOAN_APPROVAL_CSV)
            if not os.path.isfile(path):
                return None, "Loan CSV not found"
            df = pd.read_csv(path)
            X_all, _ = _preprocess_loan_for_sample(df)
            if feature_names and set(feature_names) <= set(X_all.columns):
                X_all = X_all[feature_names]
            X_all = X_all.astype(float).iloc[:max_rows]
            if len(X_all) < 2:
                return None, "Not enough loan rows"
            return scaler.transform(X_all), None
        if dataset == "bankruptcy":
            df = _load_bankruptcy_csv()
            df.columns = [c.strip() for c in df.columns]
            if "company_name" in df.columns and "year" in df.columns:
                df = df.sort_values("year", ascending=False).groupby("company_name", as_index=False).first()
            for c in ["company_name", "year"]:
                if c in df.columns:
                    df = df.drop(columns=[c])
            for target_c in ["status_label", "Bankrupt?"]:
                if target_c in df.columns:
                    df = df.drop(columns=[target_c])
            for c in df.select_dtypes(include=[np.number]).columns:
                df[c] = df[c].fillna(df[c].median())
            X_df = df.astype(float).iloc[:max_rows]
            if feature_names and set(feature_names) <= set(X_df.columns):
                X_df = X_df[feature_names]
            if len(X_df) < 2:
                return None, "Not enough bankruptcy rows"
            return scaler.transform(X_df), None
        if dataset == "credit_risk":
            df = _load_credit_csv()
            target_col = "SeriousDlqin2yrs" if "SeriousDlqin2yrs" in df.columns else df.columns[0]
            df = df.drop(columns=[target_col])
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            for c in df.select_dtypes(include=[np.number]).columns:
                df[c] = df[c].fillna(df[c].median())
            X_df = df.astype(float).iloc[:max_rows]
            if feature_names and set(feature_names) <= set(X_df.columns):
                X_df = X_df[feature_names]
            if len(X_df) < 2:
                return None, "Not enough credit rows"
            return scaler.transform(X_df), None
    except Exception as e:
        return None, str(e)
    return None, "Unknown dataset"


def _preprocess_loan_for_sample(df: pd.DataFrame):
    """
    Backend-local copy of the loan preprocessing used during training.
    Returns X (DataFrame), feature_names (list[str]).
    """
    df = df.copy()
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    target_col = "Loan_Status"
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        mode_vals = df[c].mode()
        df[c] = df[c].fillna(mode_vals.iloc[0] if len(mode_vals) else "")

    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    feature_names = df.columns.tolist()
    return df, feature_names


def _load_bankruptcy_csv() -> pd.DataFrame:
    path = os.path.join(TRAINING_DATA_DIR, BANKRUPTCY_CSV)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bankruptcy dataset not found at {path}")
    return pd.read_csv(path)


def _preprocess_bankruptcy_row(df_raw: pd.DataFrame, company_name: str):
    """
    Preprocess a single bankruptcy record.
    - If dataset has 'company_name' column: look up by name.
    - Otherwise: treat company_name as a 0-based row index string.
    Drops identifier and target columns, fills NaNs with medians.
    """
    df = df_raw.copy()
    # Strip leading/trailing spaces from column names
    df.columns = [c.strip() for c in df.columns]

    has_company = "company_name" in df.columns

    if has_company:
        subset = df[df["company_name"] == company_name]
        if subset.empty:
            raise KeyError(f"company_name '{company_name}' not found.")
        if "year" in subset.columns:
            subset = subset.sort_values("year", ascending=False).iloc[[0]]
        row = subset.iloc[0]
        year = int(row["year"]) if "year" in subset.columns else None
    else:
        try:
            idx = int(company_name)
        except ValueError:
            raise KeyError(f"company_name '{company_name}' not found and is not a valid index.")
        if idx < 0 or idx >= len(df):
            raise IndexError(f"Index {idx} out of range (dataset has {len(df)} rows).")
        row = df.iloc[idx]
        year = None

    # Detect target column
    if "status_label" in df.columns:
        target_col = "status_label"
    elif "Bankrupt?" in df.columns:
        target_col = "Bankrupt?"
    else:
        target_col = None

    # Compute medians for fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    medians = {c: df[c].median() for c in numeric_cols}

    feat_row = row.to_frame().T
    if target_col and target_col in feat_row.columns:
        feat_row = feat_row.drop(columns=[target_col])
    for drop_col in ["company_name", "year"]:
        if drop_col in feat_row.columns:
            feat_row = feat_row.drop(columns=[drop_col])
    for c in feat_row.select_dtypes(include=[np.number]).columns:
        feat_row[c] = feat_row[c].fillna(medians.get(c, feat_row[c].median()))

    # Keep only the features the trained model was trained on (from feature_names.pkl)
    try:
        trained_feature_names = get_feature_names("bankruptcy")
        if trained_feature_names:
            # Only keep columns that exist in both feat_row and trained list
            keep = [f for f in trained_feature_names if f in feat_row.columns]
            if keep:
                feat_row = feat_row[keep]
    except Exception:
        pass

    feature_names = feat_row.columns.tolist()
    features_vec = feat_row.iloc[0].astype(float).tolist()

    if "status_label" in row.index:
        status_label = str(row["status_label"])
    elif "Bankrupt?" in row.index:
        status_label = "Bankrupt" if int(row["Bankrupt?"]) == 1 else "Alive"
    else:
        status_label = None

    return year, status_label, features_vec, feature_names


def _load_credit_csv() -> pd.DataFrame:
    # Try configured CREDIT_RISK_CSV_CANDIDATES
    for name in CREDIT_RISK_CSV_CANDIDATES:
        path = os.path.join(TRAINING_DATA_DIR, name)
        if os.path.isfile(path):
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Credit-risk dataset not found. Tried: {', '.join(os.path.join(TRAINING_DATA_DIR, n) for n in CREDIT_RISK_CSV_CANDIDATES)}"
    )


def _get_loan_sample_ids(limit: int = 50) -> List[str]:
    """Return first `limit` loan IDs from the dataset for dropdown/listing."""
    path = os.path.join(TRAINING_DATA_DIR, LOAN_APPROVAL_CSV)
    if not os.path.isfile(path):
        return []
    try:
        df = pd.read_csv(path, nrows=limit * 2)
        if "Loan_ID" not in df.columns:
            return []
        ids = df["Loan_ID"].astype(str).drop_duplicates().head(limit).tolist()
        return ids
    except Exception:
        return []


def _get_bankruptcy_company_names(limit: int = 50) -> List[str]:
    """Return first `limit` company names (or row-index strings if no company_name column)."""
    try:
        df = _load_bankruptcy_csv()
        if "company_name" in df.columns:
            names = df["company_name"].astype(str).drop_duplicates().head(limit).tolist()
            return names
        # New dataset style: return row indices as strings
        return [str(i) for i in range(min(limit, len(df)))]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _get_credit_sample_indices(limit: int = 50) -> List[int]:
    """Return first `limit` valid indices for credit-risk dataset."""
    try:
        df = _load_credit_csv()
        n = len(df)
        return list(range(min(limit, n)))
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _preprocess_credit_row(df_raw: pd.DataFrame, index: int):
    """
    Approximate training.preprocess_credit for a single row by index.
    """
    df = df_raw.copy()
    target_col = "SeriousDlqin2yrs" if "SeriousDlqin2yrs" in df.columns else df.columns[0]
    serious = int(df[target_col].iloc[index]) if index < len(df) else None
    df = df.drop(columns=[target_col])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Fill numeric NaNs with column medians
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of range (0..{len(df)-1}).")
    features_vec = df.iloc[index].astype(float).tolist()
    feature_names = df.columns.tolist()
    raw = df_raw.iloc[index].to_dict()
    return serious, features_vec, feature_names, raw


# ----- Endpoints -----

@app.get("/")
def root():
    return {"service": "XAI Financial Services API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/loan/sample/{loan_id}", response_model=LoanSampleResponse)
def get_loan_sample(loan_id: str):
    """
    Load a single loan record from the original CSV and return the preprocessed feature vector
    matching the trained model, plus basic loan details.
    """
    csv_path = os.path.join(TRAINING_DATA_DIR, LOAN_APPROVAL_CSV)
    if not os.path.isfile(csv_path):
        raise HTTPException(404, f"Loan dataset not found at {csv_path}.")

    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read loan dataset: {e}")

    if "Loan_ID" not in df_raw.columns:
        raise HTTPException(500, "Loan dataset does not contain 'Loan_ID' column.")

    row_df = df_raw[df_raw["Loan_ID"] == loan_id]
    if row_df.empty:
        raise HTTPException(404, f"Loan_ID {loan_id} not found in dataset.")

    # Apply backend-local preprocessing that mirrors training.preprocess_loan
    try:
        X_all, feature_names = _preprocess_loan_for_sample(df_raw)
    except Exception as e:
        raise HTTPException(500, f"Failed to preprocess loan dataset: {e}")

    idx = row_df.index[0]
    if idx not in X_all.index:
        raise HTTPException(500, "Preprocessed data index mismatch for selected loan.")

    features_vec = X_all.loc[idx].astype(float).tolist()
    loan_status = row_df["Loan_Status"].iloc[0] if "Loan_Status" in row_df.columns else None

    return LoanSampleResponse(
        loan_id=loan_id,
        loan_status=str(loan_status) if loan_status is not None else None,
        raw=row_df.iloc[0].to_dict(),
        features=features_vec,
        feature_names=feature_names,
    )


@app.get("/bankruptcy/sample/{company_name}", response_model=BankruptcySampleResponse)
def get_bankruptcy_sample(company_name: str):
    """
    Load latest-year record for a company from american_bankruptcy.csv and return
    preprocessed features matching the trained bankruptcy model.
    """
    try:
        df_raw = _load_bankruptcy_csv()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to read bankruptcy dataset: {e}")

    try:
        year, status_label, features_vec, feature_names = _preprocess_bankruptcy_row(df_raw, company_name)
        df_raw.columns = [c.strip() for c in df_raw.columns]
        if "company_name" in df_raw.columns:
            raw = (
                df_raw[df_raw["company_name"] == company_name]
                .sort_values("year", ascending=False)
                .iloc[0]
                .to_dict()
            )
        else:
            raw = df_raw.iloc[int(company_name)].to_dict()
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to preprocess bankruptcy record: {e}")

    return BankruptcySampleResponse(
        company_name=company_name,
        year=year,
        status_label=status_label,
        raw=raw,
        features=features_vec,
        feature_names=feature_names,
    )


@app.get("/credit/sample/{index}", response_model=CreditSampleResponse)
def get_credit_sample(index: int):
    """
    Load a single credit-risk record by 0-based index from the GiveMeSomeCredit dataset.
    """
    try:
        df_raw = _load_credit_csv()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to read credit-risk dataset: {e}")

    try:
        serious, features_vec, feature_names, raw = _preprocess_credit_row(df_raw, index)
    except IndexError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to preprocess credit-risk record: {e}")

    return CreditSampleResponse(
        index=index,
        serious_dlq=serious,
        raw=raw,
        features=features_vec,
        feature_names=feature_names,
    )


@app.get("/datasets")
def list_datasets():
    """Return datasets, expertise levels, and expected feature count per dataset (when model is loaded)."""
    feature_counts = {}
    for ds in DATASETS:
        try:
            scaler = get_scaler(ds)
            feature_counts[ds] = getattr(scaler, "n_features_in_", None)
        except FileNotFoundError:
            feature_counts[ds] = None
    return {"datasets": DATASETS, "expertise_levels": EXPERTISE_LEVELS, "feature_counts": feature_counts}


@app.get("/loan/samples")
def list_loan_samples(limit: int = 50):
    """Return list of loan IDs available for 'Load from dataset' (cvision-style: data is listed)."""
    return {"loan_ids": _get_loan_sample_ids(limit)}


@app.get("/bankruptcy/samples")
def list_bankruptcy_samples(limit: int = 50):
    """Return list of company names available for 'Load from dataset'."""
    return {"company_names": _get_bankruptcy_company_names(limit)}


@app.get("/credit/samples")
def list_credit_samples(limit: int = 50):
    """Return list of valid indices for credit-risk 'Load from dataset'."""
    return {"indices": _get_credit_sample_indices(limit)}


@app.get("/ready")
def ready():
    """Returns whether model weights have finished downloading, plus progress info."""
    return {
        "ready": _models_ready,
        "done": _download_progress["done"],
        "total": _download_progress["total"],
        "current": _download_progress["current"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: PredictionRequest):
    """Return prediction (Approve/Deny) for the given feature vector."""
    if not _models_ready:
        raise HTTPException(503, "Models are still downloading, please wait a moment and try again.")
    if req.dataset not in DATASETS:
        raise HTTPException(400, f"Unknown dataset: {req.dataset}")
    model_type = req.model_type or DEFAULT_MODEL_TYPE
    try:
        model = get_model(req.dataset, model_type)
        scaler = get_scaler(req.dataset)
        feature_names = get_feature_names(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not loaded: {e}")
    X = np.array(req.features, dtype=float).reshape(1, -1)
    if X.shape[1] != scaler.n_features_in_:
        raise HTTPException(400, f"Expected {scaler.n_features_in_} features, got {len(req.features)}")
    try:
        X_df = pd.DataFrame(X, columns=feature_names)
        X_scaled = np.asarray(scaler.transform(X_df))
        pred = predict(model, X_scaled, model_type)
        proba = predict_proba(model, X_scaled, model_type)
        if proba.shape[1] >= 2:
            proba_positive = float(proba[0, 1])
        else:
            proba_positive = float(proba[0, 0])
        label = "Approved" if pred[0] == 1 else "Denied"
        if req.dataset == "bankruptcy":
            label = "Bankrupt" if pred[0] == 1 else "Alive"
        elif req.dataset == "credit_risk":
            label = "Default" if pred[0] == 1 else "No default"
        return PredictionResponse(
            prediction=int(pred[0]),
            prediction_label=label,
            probability=round(proba_positive, 4),
            dataset=req.dataset,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")


@app.post("/xai")
def xai_explain(req: XAIRequest):
    """Return SHAP, LIME, or DiCE explanation tailored to expertise level. For SHAP/LIME also returns image_url to a pre-generated PNG (cvision-style)."""
    if not _models_ready:
        raise HTTPException(503, "Models are still downloading, please wait a moment and try again.")
    if req.dataset not in DATASETS:
        raise HTTPException(400, f"Unknown dataset: {req.dataset}")
    model_type = req.model_type or DEFAULT_MODEL_TYPE
    try:
        model = get_model(req.dataset, model_type)
        scaler = get_scaler(req.dataset)
        feature_names = get_feature_names(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not loaded: {e}")
    X = np.array(req.features, dtype=float).reshape(1, -1)
    if X.shape[1] != scaler.n_features_in_:
        raise HTTPException(400, f"Expected {scaler.n_features_in_} features, got {len(req.features)}")
    X_df = pd.DataFrame(X, columns=feature_names)
    X_scaled = scaler.transform(X_df)
    X_scaled = np.asarray(X_scaled)
    X_train = _load_background_data(req.dataset)
    if X_train is not None:
        X_background = X_train[: min(200, len(X_train))]
    else:
        X_bg, _ = _build_background_from_csv(req.dataset, scaler, 200, feature_names)
        if X_bg is not None:
            X_background = np.asarray(X_bg)
        else:
            rng = np.random.RandomState(42)
            noise = rng.randn(100, X_scaled.shape[1]) * 0.15
            X_background = np.clip(X_scaled + noise, -5, 5)
    try:
        if req.method == "SHAP":
            result = get_shap_explanation(
                model, X_background, X_scaled, feature_names, req.dataset, req.expertise, model_type
            )
            if not result.get("error"):
                filename = save_shap_plot(result, OUTPUTS_DIR)
                if filename:
                    result["image_url"] = f"{BASE_URL.rstrip('/')}/static/outputs/{filename}"
        elif req.method == "LIME":
            result = get_lime_explanation(
                model, X_background, X_scaled, feature_names, req.dataset, req.expertise, model_type
            )
            if not result.get("error"):
                filename = save_lime_plot(result, OUTPUTS_DIR)
                if filename:
                    result["image_url"] = f"{BASE_URL.rstrip('/')}/static/outputs/{filename}"
        elif req.method == "DiCE":
            result = get_dice_counterfactuals(
                model, X_background, X_scaled, feature_names, req.dataset, req.expertise, model_type
            )
        else:
            raise HTTPException(400, f"Unknown method: {req.method}")
        return result
    except Exception as e:
        traceback.print_exc()
        if req.method in ("SHAP", "LIME"):
            return {
                "error": str(e),
                "method": req.method,
                "expertise": req.expertise,
                "feature_names": feature_names,
                "importance": [1.0] * len(feature_names),
                "description": f"Explanation could not be computed: {e}",
            }
        return {"error": str(e), "counterfactuals": [], "description": str(e)}


# Serve pre-generated explanation images (cvision-style)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
