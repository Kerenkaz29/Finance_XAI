# Backend — API & XAI Engine

This folder provides the **FastAPI** backend for the XAI Financial Services project: prediction endpoints, sample loading from CSVs, and explainability (SHAP, LIME, DiCE) tailored by expertise level (Expert / Non-Expert).

## Tech Stack

- **FastAPI** — REST API
- **SHAP, LIME, dice-ml** — explainability
- **NumPy, Scikit-learn, PyTorch** — model inference (must match training outputs)
- **Google Gemini** (optional) — AI-generated feature labels and DiCE scenario narratives
- **Matplotlib** — pre-generated SHAP/LIME PNG images served as static assets

## Setup

1. **Python 3.10+** and a virtualenv recommended.

2. **Install dependencies:**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Trained models:** Either run the `training` pipeline, or let the backend auto-download weights from Google Drive on first start (background thread). To point at a custom output folder:

   ```bash
   # Windows
   set XAI_TRAINING_OUTPUT=C:\Users\97254\Desktop\Finance_XAI\training\output
   ```

4. **Optional — Gemini labels:** Add `GEMINI_API_KEY` to a `.env` file at the project root (or set the env var). Without it, the backend falls back to deterministic local feature labels.

## Run

```bash
python -m uvicorn main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- Model download status: http://localhost:8000/ready  

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/ready` | Model download progress (`ready`, `done`, `total`, `current`) |
| GET | `/datasets` | List datasets, expertise levels, feature counts |
| GET | `/loan/samples` | List loan IDs for sample picker |
| GET | `/loan/sample/{loan_id}` | Load preprocessed features for one loan |
| GET | `/bankruptcy/samples` | List company names (or row indices) |
| GET | `/bankruptcy/sample/{company_name}` | Load preprocessed bankruptcy features |
| GET | `/credit/samples` | List valid credit-risk row indices |
| GET | `/credit/sample/{index}` | Load preprocessed credit features by index |
| POST | `/predict` | Predict from a feature vector |
| POST | `/xai` | SHAP, LIME, or DiCE explanation |

Static explanation PNGs are served at `/static/outputs/` (SHAP and LIME).

### POST /predict

Body (JSON):

```json
{
  "dataset": "loan",
  "features": [1.0, 0.0, 0.0, 5849, 0, 128, 360, 1, 2, ...],
  "model_type": "rf"
}
```

Response:

```json
{
  "prediction": 1,
  "prediction_label": "Approved",
  "probability": 0.87,
  "dataset": "loan"
}
```

### POST /xai

Body (JSON):

```json
{
  "dataset": "loan",
  "features": [1.0, 0.0, ...],
  "expertise": "non_expert",
  "method": "SHAP",
  "model_type": "rf"
}
```

Response structure depends on `method` (SHAP/LIME/DiCE) and `expertise`; includes `feature_names`, `importance`, optional `image_url` (SHAP/LIME), and optional `counterfactuals` (DiCE).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XAI_TRAINING_OUTPUT` | `../training/output` | Model artifacts directory |
| `XAI_DATA_DIR` | `../datasets` | Raw CSV datasets (for sample endpoints) |
| `XAI_BASE_URL` | `http://localhost:8000` | Base URL embedded in `image_url` responses |
| `GEMINI_API_KEY` | — | Enables Gemini feature labels and DiCE narratives |
| `XAI_DOWNLOAD_WORKERS` | `6` | Parallel Google Drive download threads |
| `XAI_DICE_FAST_MODE` | `1` | Use faster random search for DiCE |
| `XAI_DICE_NUM_CF` | `3` | Number of counterfactual scenarios |

## Project layout

```
backend/
├── main.py              # FastAPI app, endpoints, model auto-download
├── config.py            # Paths, CORS, expertise levels, Gemini key
├── model_loader.py      # Load .pkl/.pt and scalers from training/output
├── xai_services.py      # SHAP, LIME, DiCE + expertise tailoring
├── xai_plots.py         # Pre-generated SHAP/LIME PNG images
├── ai_terms.py          # Gemini prompts for labels and DiCE narratives
├── models/
│   ├── __init__.py
│   └── mlp_wrapper.py   # PyTorch MLP inference (matches training arch)
├── static/outputs/      # Generated explanation PNGs (created at runtime)
├── requirements.txt
└── README.md
```
