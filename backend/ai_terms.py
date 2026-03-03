"""
Generate feature display labels using Google Gemini.
Always calls Gemini — no static fallbacks.
Raises RuntimeError if the key is missing or the call fails.
"""
import json
import re
from typing import List, Dict, Any

from config import GEMINI_API_KEY

PROMPT_VERSION = "v7_stronger_technical_split"


def _force_nonexpert_surface(label: str) -> str:
    """
    Enforce a visibly non-expert style surface form so Expert/Non-Expert
    labels are clearly different in the UI.
    """
    text = str(label or "").strip()
    if not text:
        return text
    # Non-expert labels should be plain, without forced "Your ..." prefix.
    return text


def _normalize_nonexpert_label(label: str) -> str:
    """Force plain-language surface and remove technical jargon."""
    text = _force_nonexpert_surface(label)
    text = text.replace("_", " ")
    replacements = [
        (r"\bratio\b", "level"),
        (r"\bcoefficient\b", "score"),
        (r"\bvolatility\b", "changes over time"),
        (r"\bleverage\b", "debt level"),
        (r"\bsolvency\b", "ability to pay debts"),
        (r"\bretained earnings\b", "saved profit"),
        (r"\bcoverage\b", "ability to pay"),
        (r"\bcapital\b", "money"),
        (r"\bequity\b", "owned value"),
        (r"\bliabilities?\b", "debts"),
        (r"\basset turnover\b", "business activity"),
        (r"\bliquidity\b", "cash availability"),
        (r"\bdefault probability\b", "chance of default"),
        (r"\bdebt[- ]to[- ]income\b", "debt level"),
        (r"\bdelinquency\b", "late payment history"),
        (r"\butilization\b", "usage"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_expert_label(label: str) -> str:
    """Force technical analyst-style wording."""
    text = str(label or "").strip()
    text = re.sub(r"^\s*your\s+", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    replacements = [
        (r"\bmonthly income\b", "Applicant Income"),
        (r"\bincome\b", "Income"),
        (r"\bdebt level\b", "Debt Ratio"),
        (r"\bdebt\b", "Debt Exposure"),
        (r"\bpayment history\b", "Credit History"),
        (r"\bcredit score\b", "Credit Score"),
        (r"\bability to pay interest\b", "Interest Coverage Ratio"),
        (r"\bsaved profit\b", "Retained Earnings"),
        (r"\bcash availability\b", "Liquidity"),
        (r"\bbusiness activity\b", "Asset Turnover"),
        (r"\bchance of default\b", "Default Risk"),
        (r"\blate payment history\b", "Delinquency History"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


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

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=GEMINI_API_KEY)

    audience = "financial professionals" if for_expert else "general users (no ML background)"
    mode_name = "EXPERT" if for_expert else "NON_EXPERT"
    if for_expert:
        style_rules = """
MODE = EXPERT
Goal: technical analyst-style labels.
Rules:
1) Use finance/credit terminology where relevant.
2) Prefer technical terms such as: ratio, exposure, leverage, coverage, liquidity, delinquency, utilization, turnover, solvency, risk.
3) Keep labels short: 2-5 words.
4) Use noun-phrase labels (no full sentences).
5) Use Title Case, no underscores.
6) Do NOT use second-person wording ("you", "your").
7) Do NOT simplify technical terms.
8) Avoid generic words like "money", "things", "status".
Good examples:
- ApplicantIncome -> Applicant Income
- Debt_ratio_percent -> Debt-to-Asset Ratio
- Interest_Coverage_Ratio -> Interest Coverage Ratio
- Borrowing_dependency -> Borrowing Dependency
- Credit_History -> Delinquency History
"""
    else:
        style_rules = """
MODE = NON_EXPERT
Goal: plain language labels for everyday users.
Rules:
1) Use simple daily words only.
3) Keep labels short: 2-6 words.
4) STRICTLY avoid technical words:
   ratio, leverage, coverage, delinquency, utilization, solvency,
   turnover, exposure, liquidity, retained earnings, equity, liabilities.
5) Prefer plain phrases users instantly understand:
   - "Monthly income"
   - "Debt level"
   - "History of late payments"
   - "Available cash"
6) No abbreviations and no jargon.
Good examples:
- ApplicantIncome -> Monthly income
- Debt_ratio_percent -> Debt level
- Interest_Coverage_Ratio -> Ability to pay interest
- Credit_History -> Payment history
- Borrowing_dependency -> Need to borrow
"""

    prompt = f"""You are generating UI labels for a {dataset} risk model.
Audience: {audience}
Mode: {mode_name}

Follow these rules strictly:
{style_rules}
Critical contrast rule:
- EXPERT labels must read like analyst terminology.
- NON_EXPERT labels must read like everyday language.
- The two modes must be noticeably different in wording style.

Feature names (use these exact strings as JSON keys):
{chr(10).join(feature_names)}

Reply with ONLY a JSON object: each key is one of the feature names above, each value is the display label.
No markdown. No explanation. No extra keys.
If any label violates rules, correct it before final answer."""

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
        label = str(mapping[f]).strip()
        if for_expert:
            label = _normalize_expert_label(label)
        else:
            label = _normalize_nonexpert_label(label)
        result[f] = label

    return result


def get_ai_dice_scenario_explanations(
    *,
    dataset: str,
    expertise: str,
    scenarios: List[Dict[str, Any]],
) -> List[str]:
    """
    Use Gemini to explain each DiCE scenario in natural language.
    Returns one explanation per scenario (same order as input).
    Raises RuntimeError if GEMINI_API_KEY is missing or output is invalid.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env and restart the backend.")
    if not scenarios:
        return []

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    is_expert = (expertise == "expert")
    tone = (
        "technical and analytical for financial analysts"
        if is_expert
        else "plain language for non-experts"
    )

    payload = json.dumps(scenarios, ensure_ascii=False)
    prompt = f"""You explain counterfactual scenarios for a {dataset} model.
Audience tone: {tone}

Input scenarios JSON:
{payload}

Task:
For each scenario, write a detailed explanation with:
1) What key feature changes are suggested.
2) Why those changes move the prediction toward the target class.
3) What trade-off or practical implication the user should note.

Rules:
- Return ONLY a JSON array of strings.
- Same number/order as input scenarios.
- No markdown, no extra text.
- Each explanation should be 2-3 sentences.
- Expert mode: include technical wording (risk drivers, model signal, feature sensitivity).
- Non-expert mode: keep wording simple and practical.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = (response.text or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    arr = json.loads(text)
    if not isinstance(arr, list):
        raise RuntimeError(f"Gemini returned invalid scenario explanations format: {text[:200]}")
    if len(arr) != len(scenarios):
        raise RuntimeError(
            f"Gemini returned {len(arr)} scenario explanations, expected {len(scenarios)}."
        )
    out = [str(x).strip() for x in arr]
    out = [re.sub(r"\s+", " ", s.replace("_", " ")).strip() for s in out]
    if any(not s for s in out):
        raise RuntimeError("Gemini returned an empty DiCE scenario explanation.")
    return out
