# Backend — API & XAI Engine

This folder provides the **FastAPI** backend for the XAI Financial Services project: prediction endpoints and explainability (SHAP, LIME, DiCE) tailored by expertise level (Expert / Non-Expert).

## Tech Stack

- **FastAPI** — REST API
- **SHAP, LIME, dice-ml** — explainability
- **NumPy, Scikit-learn, PyTorch** — model inference (must match training outputs)

## Setup

1. **Python 3.10+** and a virtualenv recommended.

2. **Install dependencies:**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Trained models:** Ensure the `training` pipeline has been run and outputs are available. Set the path to the training output folder:

   ```bash
   # Windows
   set XAI_TRAINING_OUTPUT=C:\Users\97254\Desktop\Finance_XAI\training\output

   # Or copy/symlink training/output so it sits at ../training/output relative to backend.
   ```

## Run

```bash
python -m uvicorn main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/datasets` | List datasets and expertise levels |
| POST | `/predict` | Predict (Approve/Deny) from a feature vector |
| POST | `/xai` | Get SHAP, LIME, or DiCE explanation (expert or non_expert) |

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

Response structure depends on `method` (SHAP/LIME/DiCE) and `expertise`; includes `feature_names`, `importance`, and optional `counterfactuals` (DiCE), with descriptions tailored to expert vs non-expert.

## Project layout

```
backend/
├── main.py              # FastAPI app, /predict, /xai
├── config.py            # Paths, CORS, expertise levels
├── model_loader.py      # Load .pkl/.pt and scalers from training/output
├── xai_services.py      # SHAP, LIME, DiCE + expertise tailoring
├── xai_plots.py         # Pre-generated SHAP/LIME PNG images
├── models/
│   ├── __init__.py
│   └── mlp_wrapper.py  # PyTorch MLP inference
├── requirements.txt
└── README.md
```
