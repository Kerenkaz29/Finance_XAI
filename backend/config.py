"""Backend configuration: model paths, CORS."""
import os
from pathlib import Path

# Load .env from project root (parent of backend/) so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Path to training output (models + scalers + feature names)
# Default: sibling folder ../training/output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_OUTPUT = os.environ.get(
    "XAI_TRAINING_OUTPUT",
    os.path.join(BASE_DIR, "..", "training", "output"),
)

# Model subfolders: loan, bankruptcy, credit_risk
DATASETS = ["loan", "bankruptcy", "credit_risk"]
DEFAULT_MODEL_TYPE = "rf"  # rf | gb | lr | mlp

# CORS
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

# Public base URL for pre-generated assets (images). Set when behind proxy so image_url points to backend.
BASE_URL = os.environ.get("XAI_BASE_URL", "http://localhost:8000")

# Expertise levels for XAI tailoring
EXPERTISE_LEVELS = ["expert", "non_expert"]

# Optional: Gemini API key for AI-generated feature labels (set to enable dynamic wording)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")).strip()

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "https://finance-xai.vercel.app",
]
