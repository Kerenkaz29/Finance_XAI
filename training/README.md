# Training — Model Development

This folder trains predictive models for **Loan Approval**, **Corporate Bankruptcy**, and **Credit Risk** for the XAI Financial Services project. Outputs are consumed by the backend for inference and explainability (SHAP, LIME, DiCE).

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** — preprocessing, ensemble models (Random Forest, Gradient Boosting, Logistic Regression)
- **PyTorch** — optional deep learning (MLP) models

## Dataset Setup

Place (or symlink) your datasets so the scripts can find them:

1. **Loan Approval:** `loan_data_set.csv`  
   - Columns: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, **Loan_Status** (Y/N).

2. **Corporate Bankruptcy:** `american_bankruptcy.csv`  
   - Columns: company_name, status_label (alive/bankrupt), year, X1–X18.

3. **Credit Risk (Give Me Some Credit):** Extract `GiveMeSomeCredit.zip` and place `cs-training.csv` (or the main CSV) in the same directory.  
   - Target: `SeriousDlqin2yrs` (default in 2 years).

Set the data directory (default: `../datasets` relative to project root):

```bash
# Windows PowerShell
$env:XAI_DATA_DIR = "C:\Users\97254\Desktop\datasets"

# Or edit training/config.py and set DATA_DIR.
```

## Quick Start

```bash
cd training
pip install -r requirements.txt
```

### 1. Preprocess each dataset

```bash
python preprocess_loan.py
python preprocess_bankruptcy.py
python preprocess_credit_risk.py
```

Preprocessed arrays and scalers are written to `output/loan/`, `output/bankruptcy/`, `output/credit_risk/`.

### 2. Train models

Train all datasets and all model types (RF, GB, LR, PyTorch MLP):

```bash
python train_models.py
```

Train a single dataset and/or model:

```bash
python train_models.py --dataset loan --model rf
python train_models.py --dataset credit_risk --model gb
python train_models.py --dataset bankruptcy --model mlp
```

### 3. Outputs

- **Sklearn:** `output/<dataset>/model_rf.pkl`, `model_gb.pkl`, `model_lr.pkl`
- **PyTorch:** `output/<dataset>/model_mlp.pt`
- **Artifacts:** `scaler.pkl`, `feature_names.pkl`, `label_encoders.pkl` (loan only)

Copy or symlink the `output/` folder (or selected subfolders) into the backend so the API can load these models and artifacts.

## Project layout

```
training/
├── config.py              # DATA_DIR, output paths, constants
├── preprocess_loan.py      # Loan approval preprocessing
├── preprocess_bankruptcy.py
├── preprocess_credit_risk.py
├── train_models.py        # Ensemble + PyTorch training
├── requirements.txt
├── README.md
└── output/                # Created after preprocessing + training
    ├── loan/
    ├── bankruptcy/
    └── credit_risk/
```
