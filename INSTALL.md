# XAI Financial Services — Full install

Run these in order. Use **PowerShell** (or Command Prompt for the non-PowerShell parts).

---

## 1. Create and activate Python venv

```powershell
cd C:\Users\97254\Desktop\Finance_XAI
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get "running scripts is disabled":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then run `.\venv\Scripts\Activate.ps1` again. You should see `(venv)` in the prompt.

---

## 2. Install all Python dependencies (training + backend)

With the venv activated:

```powershell
pip install -r training\requirements.txt
pip install -r backend\requirements.txt
```

Or install from the project root in one go (both requirement files):

```powershell
pip install -r training\requirements.txt
pip install -r backend\requirements.txt
```

---

## 3. Install frontend dependencies (Node.js)

Open a **new** terminal (no need to activate venv). From the project root:

```powershell
cd C:\Users\97254\Desktop\Finance_XAI\frontend
npm install
```

---

## Summary — copy/paste block (after venv exists and is activated)

**Python (run once per machine / after creating venv):**
```powershell
cd C:\Users\97254\Desktop\Finance_XAI
.\venv\Scripts\Activate.ps1
pip install -r training\requirements.txt
pip install -r backend\requirements.txt
```

**Frontend (run once per machine):**
```powershell
cd C:\Users\97254\Desktop\Finance_XAI\frontend
npm install
```

---

## What gets installed

| File | Installs |
|------|----------|
| `training\requirements.txt` | pandas, numpy, scikit-learn, torch, joblib, tqdm |
| `backend\requirements.txt` | fastapi, uvicorn, firebase-admin, numpy, scikit-learn, pandas, joblib, torch, shap, lime, dice-ml, pydantic |
| `frontend\npm install` | react, react-dom, react-router-dom, plotly.js, react-plotly.js, vite, tailwindcss, etc. |

After this, you can run training, then start backend and frontend (see RUN.md).
