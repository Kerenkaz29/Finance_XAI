# Run Training — Create models for backend

Run these commands **in order** from the project root. Use a terminal where the venv is **activated** (`(venv)` in the prompt).

```powershell
cd C:\Users\97254\Desktop\Finance_XAI
.\venv\Scripts\Activate.ps1
```

Set your datasets folder (where `loan_data_set.csv`, `american_bankruptcy.csv`, and the Credit Risk CSV are):

```powershell
$env:XAI_DATA_DIR = "C:\Users\97254\Desktop\datasets"
```

Preprocess and train **all three** datasets:

```powershell
cd training
python preprocess_loan.py
python preprocess_bankruptcy.py
python preprocess_credit_risk.py
python train_models.py
cd ..
```

**Result:** The folder `training\output\` will contain:

- `loan\` — model_rf.pkl, scaler.pkl, feature_names.pkl, etc.
- `bankruptcy\` — model_rf.pkl, scaler.pkl, feature_names.pkl, etc.
- `credit_risk\` — model_rf.pkl, scaler.pkl, feature_names.pkl, etc.

Then start the **backend** again; it will find the models.

---

**If a preprocess script fails:**

- **Loan / Bankruptcy:** Check that `loan_data_set.csv` and `american_bankruptcy.csv` exist in the folder you set in `XAI_DATA_DIR`.
- **Credit risk:** Extract **GiveMeSomeCredit.zip** and put the main CSV (e.g. `cs-training.csv`) in the same folder. If the filename is different, you may need to copy it to `cs-training.csv` or edit `training\config.py` and set `CREDIT_RISK_CSV` to the actual filename.

**If you only have the Loan dataset:** Run at least:

```powershell
python preprocess_loan.py
python train_models.py --dataset loan
```

Then use only **Loan Approval** in the dashboard; Bankruptcy and Credit Risk will still show "Model not loaded" until you add those datasets and run their preprocess + train.
