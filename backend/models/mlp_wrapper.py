"""PyTorch MLP wrapper for inference (must match training architecture)."""
import numpy as np
import torch
import torch.nn as nn


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


class MLPWrapper:
    def __init__(self, n_features):
        self.model = MLP(n_features)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x)
            return logits.argmax(1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x)
            proba = torch.softmax(logits, dim=1).numpy()
            return proba
