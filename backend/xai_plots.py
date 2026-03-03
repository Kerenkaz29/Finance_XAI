"""
Generate static PNG images for SHAP/LIME explanations (cvision-style).
Saves to fixed filenames so frontend can load from stable URL: shap_importance.png, lime_explanation.png.
"""
import os
from typing import Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def _safe_values(data: Dict[str, Any], key_importance: str = "importance") -> list:
    vals = data.get(key_importance) or data.get("importance_raw") or []
    return [float(x) if x is not None and (isinstance(x, (int, float)) or (hasattr(x, "__float__") and callable(getattr(x, "__float__")))) else 0.0 for x in vals]


def _truncate_label(name: str, max_len: int = 40) -> str:
    s = str(name)
    return s[: max_len - 1] + "…" if len(s) > max_len else s


def save_shap_plot(result: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Draw SHAP horizontal bar chart and save as PNG. Returns filename or None."""
    if not MPL_AVAILABLE:
        return None
    if result.get("error") or not result.get("feature_names"):
        return None
    names = result["feature_names"]
    values = _safe_values(result)
    if len(values) != len(names):
        values = _safe_values(result, "importance_raw")
    if len(values) != len(names):
        return None
    # Use absolute magnitude for bar length, keep order (already sorted by importance)
    abs_vals = [abs(v) for v in values]
    labels = [_truncate_label(n) for n in names]
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, abs_vals, color="#1d4ed8", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 11)
    ax.set_xlabel("Impact (0–10)" if result.get("expertise") == "non_expert" else "mean(|SHAP value|) (0–10 scale)", fontsize=10)
    expertise = result.get("expertise", "expert")
    dataset = result.get("dataset", "")
    if expertise == "expert":
        model_label = {"loan": "Loan Approval Model", "bankruptcy": "Bankruptcy Model", "credit_risk": "Credit Risk Model"}.get(dataset, "Model")
        ax.set_title(f"Global Feature Importance – {model_label} (Expert View)", fontsize=11)
    else:
        ax.set_title("What most affected this decision", fontsize=11)
    # Add +value labels at end of each bar
    for bar, v in zip(bars, abs_vals):
        label = f"+{v:.1f}"
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left",
            fontsize=10, fontweight="bold", color="#111827"
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = "shap_importance.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return filename


def save_lime_plot(result: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Draw LIME horizontal bar chart and save as PNG. Returns filename or None."""
    if not MPL_AVAILABLE:
        return None
    if result.get("error") or not result.get("feature_names"):
        return None
    names = result["feature_names"]
    # LIME image must preserve sign: negative left, positive right.
    values = _safe_values(result, "importance_raw")
    if len(values) != len(names):
        values = _safe_values(result)
    if len(values) != len(names):
        return None
    labels = [_truncate_label(n) for n in names]
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color="#1d4ed8", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    min_v = min(values) if values else 0.0
    max_v = max(values) if values else 0.0
    lim = max(abs(min_v), abs(max_v), 0.01)
    pad = lim * 0.12
    if min_v < 0 and max_v > 0:
        ax.set_xlim(-lim - pad, lim + pad)
        ax.axvline(0, color="#374151", linewidth=1.2)
    elif min_v < 0:
        ax.set_xlim(min_v - pad, 0)
    else:
        ax.set_xlim(0, max_v + pad)
    ax.set_xlabel("Probability effect", fontsize=10)
    ax.set_title("LIME Local Explanation", fontsize=11)
    # Signed labels, no forced plus sign.
    label_offset = max(lim * 0.01, 0.001)
    for bar, v in zip(bars, values):
        label = f"{v:.4f}"
        x = bar.get_width() + label_offset if v >= 0 else bar.get_width() - label_offset
        ha = "left" if v >= 0 else "right"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha=ha,
            fontsize=10, fontweight="bold", color="#111827"
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = "lime_explanation.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return filename
