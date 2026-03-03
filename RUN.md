# How to Run XAI Financial Services

Run **training** and **backend** in a **Python environment**; run **frontend** with **Node.js (npm)**. Use the order below.

---

## Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/downloads/) or Windows Store.
- **Node.js 18+** — [nodejs.org](https://nodejs.org/) (includes npm).
- **Datasets** in one folder, e.g. `C:\Users\97254\Desktop\datasets`:
  - `loan_data_set.csv`
  - `american_bankruptcy.csv`
  - From **GiveMeSomeCredit.zip**: extract and place `cs-training.csv` (or the main CSV) in the same folder.

---

## 1. Python environment (Training + Backend)

Use a **virtual environment** so dependencies don’t conflict with other projects.

### Option A: venv (recommended)

Open **PowerShell** in the project root and run these **in order**:

**Step 1 — Create the virtual environment** (run once; wait until it finishes):

```powershell
cd C:\Users\97254\Desktop\Finance_XAI
python -m venv venv
```

**Step 2 — Activate it** (you must do this every time you open a new terminal for backend/training):

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt. If you get *"cannot be loaded because running scripts is disabled"*, run this **once** (then try activating again):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

If you get *"Activate.ps1 is not recognized"*, the `venv` folder was not created: make sure Step 1 finished without errors and that you are in `C:\Users\97254\Desktop\Finance_XAI` when you run Step 2.

```powershell
# Install training deps
pip install -r training\requirements.txt

# Optional: set datasets folder (if not using Desktop\datasets)
$env:XAI_DATA_DIR = "C:\Users\97254\Desktop\datasets"

# Preprocess & train (run from project root so imports work)
cd training
python preprocess_loan.py
python preprocess_bankruptcy.py
python preprocess_credit_risk.py
python train_models.py
cd ..
```

### Option B: conda

```powershell
cd C:\Users\97254\Desktop\Finance_XAI
conda create -n finance-xai python=3.11 -y
conda activate finance-xai
pip install -r training\requirements.txt
$env:XAI_DATA_DIR = "C:\Users\97254\Desktop\datasets"
cd training
python preprocess_loan.py
python preprocess_bankruptcy.py
python preprocess_credit_risk.py
python train_models.py
cd ..
```

---

## 2. Backend (same Python env)

Keep the same terminal with the venv/conda activated:

```powershell
# From project root
pip install -r backend\requirements.txt

# Optional: point to training output (default is ..\training\output)
$env:XAI_TRAINING_OUTPUT = "C:\Users\97254\Desktop\Finance_XAI\training\output"

# Run API (from project root, then backend)
cd backend
python -m uvicorn main:app --reload --port 8000
```
If `uvicorn` is not found, ensure the venv is activated and install: `pip install uvicorn`

Leave this running. API: http://localhost:8000 — docs: http://localhost:8000/docs.

SHAP and LIME explanations are generated as PNG images (cvision-style) and served under `/static/outputs/`. The frontend displays these images when you run SHAP or LIME analysis. If the backend is not on `http://localhost:8000`, set `XAI_BASE_URL` so image URLs point to the correct origin.

---

## 3. Frontend (Node.js / npm)

Open a **second terminal** (no need for Python venv here):

```powershell
cd C:\Users\97254\Desktop\Finance_XAI\frontend
npm install
npm run dev
```

Browser: http://localhost:5173. The app proxies `/api` to the backend on port 8000.

---

## Summary: what runs where

| Part      | Environment        | Command                          | URL                  |
|-----------|--------------------|----------------------------------|----------------------|
| Training  | Python (venv/conda)| `cd training && python train_models.py` | —                    |
| Backend   | Python (same env)  | `cd backend && python -m uvicorn main:app --reload --port 8000` | http://localhost:8000 |
| Frontend  | Node.js            | `cd frontend && npm run dev`     | http://localhost:5173 |

---

## One-time setup checklist

1. Create and activate Python venv (or conda env).
2. `pip install -r training\requirements.txt` and run preprocessing + training.
3. `pip install -r backend\requirements.txt`.
4. In the frontend folder: `npm install`.

## Daily run (after setup)

1. **Terminal 1:** Activate venv → `cd backend` → `python -m uvicorn main:app --reload --port 8000`.
2. **Terminal 2:** `cd frontend` → `npm run dev`.
3. Open http://localhost:5173 and use the dashboard (toggle Expert/Non-Expert, run prediction and SHAP/LIME/DiCE, then Survey).

---

## Troubleshooting

- **“Model not found”** — Run training first and ensure `training\output\loan` (and bankruptcy, credit_risk) contain `.pkl`/`.pt` and `scaler.pkl`. Set `XAI_TRAINING_OUTPUT` if you moved the output folder.
- **Dataset not found** — Set `XAI_DATA_DIR` to the folder that contains `loan_data_set.csv`, `american_bankruptcy.csv`, and the Credit Risk CSV.
- **CORS / API errors in browser** — Backend must be running on port 8000; frontend dev server proxies `/api` to it.
- **PowerShell can’t run Activate.ps1** — Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.
