# XAI Financial Services

Research-based **Explainable AI (XAI)** system for financial decision-making, comparing how **experts** and **non-experts** accept AI explanations.

## Structure

| Folder | Purpose | Tech |
|--------|---------|------|
| **training** | Train predictive models on 3 datasets | Python, PyTorch, Scikit-learn |
| **backend** | API + XAI engine (SHAP, LIME, DiCE) | FastAPI, Firebase Admin SDK |
| **frontend** | Dashboard (Expert / Non-Expert) + survey | React, Tailwind, Plotly |

## Datasets

- **Loan Approval:** `loan_data_set.csv` (e.g. `Loan_Status` Y/N)
- **Corporate Bankruptcy:** `american_bankruptcy.csv` (e.g. `status_label` alive/bankrupt)
- **Credit Risk:** Give Me Some Credit — extract `GiveMeSomeCredit.zip` and use `cs-training.csv` (or the main CSV)

Place CSVs in a folder and set:

- **Training:** `XAI_DATA_DIR` (default: `../datasets` relative to project root), or edit `training/config.py`.
- **Backend:** `XAI_TRAINING_OUTPUT` to the path of `training/output` (or leave default `../training/output`).

## Quick start

### 1. Training

```bash
cd training
pip install -r requirements.txt
# Set XAI_DATA_DIR if needed, e.g. C:\Users\97254\Desktop\datasets
python preprocess_loan.py
python preprocess_bankruptcy.py
python preprocess_credit_risk.py   # after extracting GiveMeSomeCredit.zip
python train_models.py
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Use the **Expert / Non-Expert** toggle and run predictions and XAI (SHAP, LIME, DiCE). Complete the **Survey** (S-TIAS, SCS) from the nav.

## Reference

Structure inspired by [Understanding-of-AI-Based-Recruitment-Outcomes (cvision)](https://github.com/YoavKatz99/Understanding-of-AI-Based-Recruitment-Outcomes/tree/main/cvision) (backend/frontend layout, Vite + Tailwind).

## License

Use as needed for your research.
