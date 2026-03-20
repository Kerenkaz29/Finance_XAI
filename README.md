# XAI Financial Services

Research-based **Explainable AI (XAI)** system for financial decision-making with SHAP, LIME, and DiCE explanations.

## Structure

| Folder | Purpose | Tech |
|--------|---------|------|
| **training** | Train predictive models on 3 datasets | Python, PyTorch, Scikit-learn |
| **backend** | API + XAI engine (SHAP, LIME, DiCE) | FastAPI, Firebase Admin SDK |
| **frontend** | Dashboard + survey | React, Tailwind, Plotly |

## Datasets

- **Loan Approval:** `loan_data_set.csv` (e.g. `Loan_Status` Y/N)
- **Corporate Bankruptcy:** `american_bankruptcy.csv` (e.g. `status_label` alive/bankrupt)
- **Credit Risk:** Give Me Some Credit — extract `GiveMeSomeCredit.zip` and use `cs-training.csv` (or the main CSV)

Place CSVs in a folder and set:

- **Training:** `XAI_DATA_DIR` (default: `../datasets` relative to project root), or edit `training/config.py`.
- **Backend:** `XAI_TRAINING_OUTPUT` to the path of `training/output` (or leave default `../training/output`).

## Quick Start (Windows / PowerShell)

### 1. Training (optional if models already exist)

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
py -m uvicorn main:app --reload --port 8000
```

### 3. Frontend

```powershell
cd "C:\Users\97254\Desktop\Finance_XAI\frontend"
npm install
npm run dev
```

Open `http://localhost:5173` and run predictions + XAI (SHAP, LIME, DiCE).

Notes:
- The frontend currently runs in **expert-style explanation mode** by default (no expert/non-expert tab toggle in the UI).
- Backend health endpoint: `http://127.0.0.1:8000/health`

## Reference

Structure inspired by [Understanding-of-AI-Based-Recruitment-Outcomes (cvision)](https://github.com/YoavKatz99/Understanding-of-AI-Based-Recruitment-Outcomes/tree/main/cvision) (backend/frontend layout, Vite + Tailwind).

## License

Use as needed for your research.
