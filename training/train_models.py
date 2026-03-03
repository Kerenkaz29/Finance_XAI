"""
Train ensemble (sklearn) and optional DL (PyTorch) models for Loan, Bankruptcy, and Credit Risk.
Saves .pkl (sklearn) and .pt (PyTorch) to training/output/<dataset>/.
Run preprocess_*.py first for each dataset.
"""
import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import OUTPUT_DIR, RANDOM_STATE

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_preprocessed(dataset: str):
    """dataset: 'loan' | 'bankruptcy' | 'credit_risk'"""
    base = os.path.join(OUTPUT_DIR, dataset)
    X_train = np.load(os.path.join(base, "X_train.npy"))
    y_train = np.load(os.path.join(base, "y_train.npy"))
    X_test = np.load(os.path.join(base, "X_test.npy"))
    y_test = np.load(os.path.join(base, "y_test.npy"))
    feature_names = joblib.load(os.path.join(base, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    return X_train, y_train, X_test, y_test, feature_names, scaler


# ----- Ensemble (sklearn) -----

def train_ensemble(X_train, y_train, X_test, y_test, dataset: str, model_type="rf"):
    base = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(base, exist_ok=True)

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    elif model_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        # Handle single-class case (e.g. bankruptcy with only one class in split)
        y_proba = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
    else:
        y_proba = y_pred.astype(float)

    print(f"[{dataset}] {model_type.upper()} — Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    try:
        if len(np.unique(y_test)) >= 2:
            print(f"  AUC: {roc_auc_score(y_test, y_proba):.4f}")
        else:
            print(f"  AUC: (skipped — only one class in test set)")
    except Exception:
        pass

    path = os.path.join(base, f"model_{model_type}.pkl")
    joblib.dump(clf, path)
    print(f"  Saved: {path}")
    return clf


# ----- PyTorch MLP -----

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, n_features, n_classes=2, hidden=(64, 32)):
            super().__init__()
            layers = []
            prev = n_features
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
                prev = h
            layers += [nn.Linear(prev, n_classes)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


def train_pytorch(X_train, y_train, X_test, y_test, dataset: str, epochs=50, lr=1e-2):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed; skipping .pt model.")
        return None
    base = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(base, exist_ok=True)

    n_features = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = criterion(out, yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            pred = out.argmax(1).numpy()
            acc = accuracy_score(y_train, pred)
            print(f"  Epoch {ep+1}/{epochs} loss={loss.item():.4f} train_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        y_pred = logits.argmax(1).numpy()
        y_proba = torch.softmax(logits, dim=1)[:, 1].numpy()

    print(f"[{dataset}] PyTorch MLP — Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    try:
        print(f"  AUC: {roc_auc_score(y_test, y_proba):.4f}")
    except Exception:
        pass

    path = os.path.join(base, "model_mlp.pt")
    torch.save({"state_dict": model.state_dict(), "n_features": n_features}, path)
    print(f"  Saved: {path}")
    return model


# ----- Entrypoint -----

DATASETS = ["loan", "bankruptcy", "credit_risk"]


def main():
    parser = argparse.ArgumentParser(description="Train models for XAI Financial Services")
    parser.add_argument("--dataset", choices=DATASETS, default=None, help="Train one dataset; default all")
    parser.add_argument("--model", choices=["rf", "gb", "lr", "mlp", "all"], default="all")
    parser.add_argument("--skip-preprocess", action="store_true", help="Assume preprocessed data exists")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    for ds in datasets:
        base = os.path.join(OUTPUT_DIR, ds)
        if not os.path.isfile(os.path.join(base, "X_train.npy")):
            print(f"Missing preprocessed data for {ds}. Run preprocess_loan.py / preprocess_bankruptcy.py / preprocess_credit_risk.py first.")
            continue
        X_train, y_train, X_test, y_test, _, _ = load_preprocessed(ds)
        n_classes_train = len(np.unique(y_train))
        if n_classes_train < 2:
            print(f"[{ds}] Skipped — training set has only one class (need at least 2 for classification). Re-run preprocess with a balanced dataset or use more data.")
            continue
        if args.model in ("rf", "all"):
            train_ensemble(X_train, y_train, X_test, y_test, ds, "rf")
        if args.model in ("gb", "all"):
            train_ensemble(X_train, y_train, X_test, y_test, ds, "gb")
        if args.model in ("lr", "all"):
            train_ensemble(X_train, y_train, X_test, y_test, ds, "lr")
        if args.model in ("mlp", "all") and TORCH_AVAILABLE:
            train_pytorch(X_train, y_train, X_test, y_test, ds)


if __name__ == "__main__":
    main()
