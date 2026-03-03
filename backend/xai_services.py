"""
XAI service: SHAP, LIME, DiCE with expertise-level tailoring.
Returns JSON-friendly structures for expert vs non-expert views.
"""
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional

from config import EXPERTISE_LEVELS

try:
    from ai_terms import get_ai_feature_labels
except ImportError:
    get_ai_feature_labels = None

# Optional imports so backend can start even if XAI libs missing
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False




def get_feature_display_names(dataset: str, feature_names: List[str], for_expert: bool = False) -> Dict[str, str]:
    """Always use Gemini to generate feature display labels."""
    if get_ai_feature_labels:
        ai_labels = get_ai_feature_labels(feature_names, dataset, for_expert)
        if ai_labels:
            return ai_labels
    # Gemini unavailable or failed — raise so the caller knows
    raise RuntimeError(
        f"Gemini could not generate labels for dataset='{dataset}', for_expert={for_expert}. "
        "Check GEMINI_API_KEY in .env and that google-generativeai is installed."
    )


def _feature_name_for_expert(raw_name: str) -> str:
    """Format raw feature name for expert view: no underscores, title-style (e.g. Loan_Amount_Term -> Loan Amount Term)."""
    return raw_name.replace("_", " ").strip()


def _replace_lime_rule_feature(rule_text: str, feature_names: List[str], display_names: Dict[str, str]) -> str:
    """
    LIME returns rule strings (e.g. '-0.2 < Credit_History <= 0.7').
    Replace only the feature-name token with the Gemini label while keeping numeric bounds/operators.
    """
    if not rule_text:
        return rule_text

    # Longer names first to avoid partial matches.
    for raw_name in sorted(feature_names, key=len, reverse=True):
        replacement = display_names.get(raw_name, _feature_name_for_expert(raw_name))
        variants = [
            raw_name,
            raw_name.replace("_", " "),
            raw_name.replace(" ", "_"),
        ]
        for variant in variants:
            if not variant:
                continue
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])")
            if pattern.search(rule_text):
                return pattern.sub(replacement, rule_text, count=1)

    # Fallback path for rules where feature token formatting differs from feature_names:
    # preserve numeric bounds/operators and rewrite only the middle feature chunk.
    two_sided = re.match(r"^\s*([-+]?\d*\.?\d+)\s*<\s*(.+?)\s*<=\s*([-+]?\d*\.?\d+)\s*$", rule_text)
    if two_sided:
        lo, feature_chunk, hi = two_sided.groups()
        mapped = display_names.get(feature_chunk) or display_names.get(feature_chunk.replace(" ", "_"))
        if mapped:
            return f"{lo} < {mapped} <= {hi}"

    one_sided = re.match(r"^\s*(.+?)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+)\s*$", rule_text)
    if one_sided:
        feature_chunk, op, val = one_sided.groups()
        mapped = display_names.get(feature_chunk) or display_names.get(feature_chunk.replace(" ", "_"))
        if mapped:
            return f"{mapped} {op} {val}"

    return rule_text


def _scale_importance(values: np.ndarray, scale_max: float = 10.0) -> List[float]:
    """Scale importance values to 0–scale_max for display. If all zeros, return uniform small values so chart shows bars."""
    v = np.asarray(values, dtype=float).flatten()
    if v.size == 0:
        return []
    abs_v = np.abs(v)
    m = float(abs_v.max())
    if m <= 0:
        return [1.0] * len(v)
    scaled = (abs_v / m) * scale_max
    return (np.sign(v) * scaled).tolist() if np.any(v != 0) else scaled.tolist()


# ----- SHAP -----
def get_shap_explanation(
    model,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    feature_names: List[str],
    dataset: str,
    expertise: str,
    model_type: str = "rf",
) -> Dict[str, Any]:
    """SHAP values; expert gets raw names, non-expert gets display names and simplified wording."""
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed", "feature_names": feature_names, "importance": [1.0] * len(feature_names)}

    try:
        if model_type == "mlp":
            def pred_fn(x):
                return model.predict_proba(x)
            explainer = shap.KernelExplainer(pred_fn, X_background[:min(50, len(X_background))])
            shap_values = explainer.shap_values(X_instance)
        elif model_type in ("rf", "gb"):
            # Tree models: no background needed (like cvision). SHAP uses tree structure.
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_instance)
        else:
            # e.g. lr: use background for KernelExplainer
            def pred_fn(x):
                return model.predict_proba(x)
            explainer = shap.KernelExplainer(pred_fn, X_background[:min(50, len(X_background))])
            shap_values = explainer.shap_values(X_instance)
        if isinstance(shap_values, list):
            if len(shap_values) == 0:
                mean_abs = np.zeros(len(feature_names))
            elif len(shap_values) == 2:
                shap_values = np.asarray(shap_values[1])
            else:
                shap_values = np.asarray(shap_values[0])
        if isinstance(shap_values, list):
            mean_abs = np.zeros(len(feature_names))
        else:
            arr = np.asarray(shap_values)
            # Newer SHAP returns 3D (n_samples, n_features, n_classes): extract positive class
            if arr.ndim == 3:
                arr = arr[:, :, 1]
            if arr.ndim >= 2 and int(arr.shape[0]) > 1:
                mean_abs = np.mean(np.abs(arr), axis=0).flatten()
            else:
                mean_abs = np.abs(arr.flatten())
            if mean_abs.size == 0:
                mean_abs = np.zeros(len(feature_names))
        if mean_abs.size == 0:
            mean_abs = np.zeros(len(feature_names))
        n_f = min(mean_abs.size, len(feature_names))
        mean_abs = mean_abs[:n_f]
        feature_names_use = feature_names[:n_f]
        order = np.argsort(-mean_abs)
        names = [feature_names_use[i] for i in order]
        values = mean_abs[order].tolist()
        scaled = _scale_importance(mean_abs[order], 10.0)

        display_names = get_feature_display_names(dataset, feature_names, for_expert=(expertise == "expert"))
        names_display = [display_names.get(n, _feature_name_for_expert(n)) for n in names]
        if expertise == "non_expert":
            return {
                "method": "SHAP",
                "expertise": expertise,
                "dataset": dataset,
                "feature_names": names_display,
                "feature_names_raw": names,
                "importance": scaled,
                "importance_raw": values,
                "description": "This chart shows which factors matter most. The longer the bar, the more that factor affected the decision.",
            }
        return {
            "method": "SHAP",
            "expertise": expertise,
            "dataset": dataset,
            "feature_names": names_display,
            "importance": scaled,
            "importance_raw": values,
            "description": "SHapley Additive exPlanations: mean absolute SHAP value per feature (0–10 scaled).",
        }
    except Exception as e:
        return {"error": str(e), "feature_names": feature_names, "importance": [1.0] * len(feature_names), "description": f"SHAP could not be computed: {e}"}


# ----- LIME -----
def get_lime_explanation(
    model,
    X_train: np.ndarray,
    X_instance: np.ndarray,
    feature_names: List[str],
    dataset: str,
    expertise: str,
    model_type: str = "rf",
    num_features: int = 10,
) -> Dict[str, Any]:
    """LIME local explanation; tailored labels by expertise."""
    if not LIME_AVAILABLE:
        return {"error": "LIME not installed", "feature_names": feature_names, "importance": [1.0] * len(feature_names)}

    try:
        def pred_fn(x):
            # LIME classification expects (n_samples, n_classes) probabilities.
            return model.predict_proba(x)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode="classification",
            random_state=42,
        )
        exp = explainer.explain_instance(
            X_instance[0],
            pred_fn,
            num_features=num_features,
            top_labels=1,
        )
        # Explain the predicted class for this instance.
        pred_probs = model.predict_proba(X_instance)
        pred_label = int(np.argmax(pred_probs[0])) if pred_probs.ndim == 2 else 1
        list_exp = exp.as_list(label=pred_label)
        names = [x[0] for x in list_exp]
        weights = [float(x[1]) for x in list_exp]
        scaled = _scale_importance(weights, 10.0)

        display_names = get_feature_display_names(dataset, feature_names, for_expert=(expertise == "expert"))
        names_display = [
            _replace_lime_rule_feature(n, feature_names, display_names)
            for n in names
        ]
        if expertise == "non_expert":
            return {
                "method": "LIME",
                "expertise": expertise,
                "feature_names": names_display,
                "feature_names_raw": names,
                "importance": scaled,
                "importance_raw": weights,
                "description": "How each factor affected this decision. Positive = pushed toward approval.",
            }
        return {
            "method": "LIME",
            "expertise": expertise,
            "feature_names": names_display,
            "importance": scaled,
            "importance_raw": weights,
            "description": "LIME: local linear approximation of the model for this instance (feature weights).",
        }
    except Exception as e:
        return {"error": str(e), "feature_names": feature_names, "importance": [1.0] * len(feature_names), "description": f"LIME could not be computed: {e}"}


# ----- DiCE (counterfactuals) -----
def get_dice_counterfactuals(
    model,
    X_train: np.ndarray,
    X_instance: np.ndarray,
    feature_names: List[str],
    dataset: str,
    expertise: str,
    model_type: str = "rf",
    num_cf: int = 3,
) -> Dict[str, Any]:
    """DiCE counterfactuals: what to change to flip the outcome."""
    if not DICE_AVAILABLE:
        return {"error": "dice_ml not installed", "counterfactuals": []}

    try:
        import pandas as pd
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_instance_df = pd.DataFrame(X_instance, columns=feature_names)

        if model_type == "mlp":
            d = dice_ml.Data(dataframe=X_train_df, continuous_features=feature_names, outcome_name="outcome")
            # DiCE model wrapper for PyTorch-style predict_proba
            class SklearnWrapper:
                def __init__(self, m):
                    self.m = m
                def predict(self, x):
                    return self.m.predict(x)
                def predict_proba(self, x):
                    return self.m.predict_proba(x)
            backend = dice_ml.Model(model=SklearnWrapper(model), backend="sklearn")
        else:
            d = dice_ml.Data(dataframe=X_train_df, continuous_features=feature_names, outcome_name="outcome")
            backend = dice_ml.Model(model=model, backend="sklearn")
        exp = dice_ml.Dice(d, backend)
        # Generate counterfactuals
        dice_exp = exp.generate_counterfactuals(
            X_instance_df,
            total_CFs=num_cf,
            desired_class="opposite",
        )
        cf_list = dice_exp.cf_examples_list
        if not cf_list or len(cf_list) == 0 or cf_list[0] is None:
            return {"method": "DiCE", "expertise": expertise, "counterfactuals": [], "description": "No counterfactuals generated."}
        cfs = cf_list[0]
        if hasattr(cfs, "final_cfs_df"):
            cf_df = cfs.final_cfs_df
        else:
            cf_df = cfs
        # Build simple list of changes per CF
        instance_vals = X_instance_df.iloc[0]
        counterfactuals = []
        for i in range(min(num_cf, len(cf_df))):
            row = cf_df.iloc[i]
            changes = {}
            for c in feature_names:
                if c in row.index and c in instance_vals.index:
                    ov = float(instance_vals[c])
                    nv = float(row[c])
                    if abs(ov - nv) > 1e-6:
                        changes[c] = {"from": round(ov, 4), "to": round(nv, 4)}
            counterfactuals.append({"changes": changes})
        display_names = get_feature_display_names(dataset, feature_names, for_expert=(expertise == "expert"))
        for cf in counterfactuals:
            cf["changes_display"] = {display_names.get(k, _feature_name_for_expert(k)): v for k, v in cf["changes"].items()}
        return {
            "method": "DiCE",
            "expertise": expertise,
            "counterfactuals": counterfactuals,
            "description": "What changes would flip the decision? These are example changes." if expertise == "non_expert" else "DiCE counterfactuals: minimal feature perturbations to flip predicted class.",
        }
    except Exception as e:
        return {"error": str(e), "counterfactuals": []}
