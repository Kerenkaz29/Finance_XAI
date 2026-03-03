"""
Generate feature display labels using Google Gemini.
Always calls Gemini — no static fallbacks.
Raises RuntimeError if the key is missing or the call fails.
"""
import json
import re
from typing import List, Dict

from config import GEMINI_API_KEY

_cache: Dict[tuple, Dict[str, str]] = {}


def get_ai_feature_labels(
    feature_names: List[str],
    dataset: str,
    for_expert: bool,
) -> Dict[str, str]:
    """
    Ask Gemini to map each feature name to a short display label.
    - for_expert=True  → technical names (e.g. "Applicant Income").
    - for_expert=False → simple everyday language (e.g. "Your monthly income").
    Raises RuntimeError if GEMINI_API_KEY is not set or the call fails.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env and restart the backend.")
    if not feature_names:
        raise RuntimeError("No feature names provided to Gemini labeller.")

    key = (dataset, for_expert, tuple(sorted(feature_names)))
    if key in _cache:
        return _cache[key]

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=GEMINI_API_KEY)

    audience = "financial professionals" if for_expert else "general users (no ML background)"
    style = (
        "Use precise technical or variable-style names (e.g. 'Applicant Income', 'Credit History'). "
        "No underscores; use spaces. Short labels."
        if for_expert
        else "Use simple, everyday language (e.g. 'Your monthly income', 'Your past credit history'). "
        "Short and clear for non-experts."
    )

    prompt = f"""You are a financial ML expert. For a {dataset} prediction model, map each feature name below to a short display label for {audience}.

Style: {style}

Feature names (use these exact strings as JSON keys):
{chr(10).join(feature_names)}

Reply with ONLY a JSON object: each key is one of the feature names above, each value is the display label. No markdown, no explanation. Example: {{"ApplicantIncome": "Applicant Income", "Credit_History": "Credit History"}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = (response.text or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    mapping = json.loads(text)
    if not isinstance(mapping, dict):
        raise RuntimeError(f"Gemini returned unexpected format: {text[:200]}")

    result = {}
    for f in feature_names:
        if f not in mapping or not mapping[f]:
            raise RuntimeError(f"Gemini did not return a label for feature '{f}'.")
        result[f] = str(mapping[f]).strip()

    _cache[key] = result
    return result
