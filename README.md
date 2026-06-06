# XAI Financial Services

Research-based **Explainable AI (XAI)** system for financial decision-making with SHAP, LIME, and DiCE explanations across three datasets: loan approval, corporate bankruptcy, and credit risk.

## Structure

| Folder | Purpose | Tech |
|--------|---------|------|
| **training** | Train predictive models on 3 datasets | Python, PyTorch, Scikit-learn |
| **backend** | REST API + XAI engine (SHAP, LIME, DiCE) | FastAPI, SHAP, LIME, dice-ml, Gemini |
| **frontend** | Interactive analysis dashboard | React, Vite, Tailwind, Recharts |

## Datasets

- **Loan Approval:** `loan_data_set.csv` (target: `Loan_Status` Y/N)
- **Corporate Bankruptcy:** `american_bankruptcy.csv` (target: `status_label` or `Bankrupt?`)
- **Credit Risk:** Give Me Some Credit — extract `GiveMeSomeCredit.zip` and use `cs-training.csv`

Place CSVs in a folder and set:

- **Training:** `XAI_DATA_DIR` (default: `../datasets` relative to project root), or edit `training/config.py`.
- **Backend:** `XAI_TRAINING_OUTPUT` to the path of `training/output` (default: `../training/output`).

On first backend start, model weights are **auto-downloaded from Google Drive** if missing. You can also run the training pipeline locally (see below).

## Quick Start (Windows / PowerShell)

### 1. Training (optional if models already exist or auto-download is used)

```powershell
cd "C:\Users\97254\Desktop\Finance_XAI\training"
py -m pip install -r requirements.txt
# Set XAI_DATA_DIR if needed, e.g. C:\Users\97254\Desktop\datasets
py preprocess_loan.py
py preprocess_bankruptcy.py
py preprocess_credit_risk.py
py train_models.py
```

### 2. Backend

```powershell
cd "C:\Users\97254\Desktop\Finance_XAI\backend"
py -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
# Optional: AI-generated feature labels and DiCE narratives
$env:GEMINI_API_KEY = "your-api-key"
py -m uvicorn main:app --reload --port 8000
```

### 3. Frontend

```powershell
cd "C:\Users\97254\Desktop\Finance_XAI\frontend"
npm install
npm run dev
```

Open `http://localhost:5173` and run predictions + XAI (SHAP, LIME, DiCE).

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `XAI_DATA_DIR` | training, backend | Path to CSV datasets |
| `XAI_TRAINING_OUTPUT` | backend | Path to `training/output` (models, scalers) |
| `GEMINI_API_KEY` | backend | Optional — expert/non-expert feature labels and DiCE narratives |
| `XAI_BASE_URL` | backend | Public URL for static explanation PNGs (default: `http://localhost:8000`) |
| `VITE_API_URL` | frontend | Backend base URL (default: `/api` via Vite proxy) |

## Notes

- The frontend currently runs in **expert mode** by default (no expert/non-expert tab in the UI; `ExpertiseContext` is wired for future use).
- Backend health: `http://127.0.0.1:8000/health` — model download status: `http://127.0.0.1:8000/ready`
- See [HOW_TO_OPERATE.md](HOW_TO_OPERATE.md) for step-by-step operation and [INSTALL.md](INSTALL.md) for full setup.

## Reference

Structure inspired by [Understanding-of-AI-Based-Recruitment-Outcomes (cvision)](https://github.com/YoavKatz99/Understanding-of-AI-Based-Recruitment-Outcomes/tree/main/cvision) (backend/frontend layout, Vite + Tailwind).

## License

Use as needed for your research.
