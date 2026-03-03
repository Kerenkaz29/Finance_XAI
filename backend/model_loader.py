"""Load trained models and artifacts from training/output."""
import os
import joblib
import numpy as np

from config import TRAINING_OUTPUT, DATASETS, DEFAULT_MODEL_TYPE

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_models = {}
_scalers = {}
_feature_names = {}
_label_encoders = {}


def _model_path(dataset: str, model_type: str) -> str:
    base = os.path.join(TRAINING_OUTPUT, dataset)
    if model_type == "mlp":
        return os.path.join(base, "model_mlp.pt")
    return os.path.join(base, f"model_{model_type}.pkl")


def get_model(dataset: str, model_type: str = None):
    model_type = model_type or DEFAULT_MODEL_TYPE
    key = (dataset, model_type)
    if key in _models:
        return _models[key]
    path = _model_path(dataset, model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}. Run training first.")
    if model_type == "mlp" and TORCH_AVAILABLE:
        data = torch.load(path, map_location="cpu", weights_only=True)
        from models.mlp_wrapper import MLPWrapper
        wrapper = MLPWrapper(data["n_features"])
        wrapper.model.load_state_dict(data["state_dict"])
        _models[key] = wrapper
    else:
        _models[key] = joblib.load(path)
    return _models[key]


def get_scaler(dataset: str):
    if dataset in _scalers:
        return _scalers[dataset]
    path = os.path.join(TRAINING_OUTPUT, dataset, "scaler.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scaler not found: {path}")
    _scalers[dataset] = joblib.load(path)
    return _scalers[dataset]


def get_feature_names(dataset: str):
    if dataset in _feature_names:
        return _feature_names[dataset]
    path = os.path.join(TRAINING_OUTPUT, dataset, "feature_names.pkl")
    if not os.path.isfile(path):
        return []
    _feature_names[dataset] = joblib.load(path)
    return _feature_names[dataset]


def get_label_encoders(dataset: str):
    if dataset == "loan" and dataset not in _label_encoders:
        path = os.path.join(TRAINING_OUTPUT, dataset, "label_encoders.pkl")
        if os.path.isfile(path):
            _label_encoders[dataset] = joblib.load(path)
        else:
            _label_encoders[dataset] = {}
    return _label_encoders.get(dataset, {})


def predict_proba(model, X: np.ndarray, model_type: str = "rf") -> np.ndarray:
    """Return (n_samples, 2) probability array; index 1 = positive class."""
    if model_type == "mlp" and TORCH_AVAILABLE:
        return model.predict_proba(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    pred = model.predict(X)
    return np.column_stack([1 - pred, pred])


def predict(model, X: np.ndarray, model_type: str = "rf") -> np.ndarray:
    if model_type == "mlp" and hasattr(model, "predict"):
        return model.predict(X)
    return model.predict(X)
