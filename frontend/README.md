# Frontend — User Interface

Interactive dashboard for the XAI Financial Services project: **Expert** and **Non-Expert** modes, SHAP/LIME/DiCE visualizations, and a feedback survey (S-TIAS, SCS).

## Tech Stack

- **React 18** + **Vite**
- **Tailwind CSS** for styling
- **React Plotly.js** for charts (SHAP feature importance, LIME local explanation)
- **React Router** for navigation

## Setup

```bash
cd frontend
npm install
```

## Run

```bash
npm run dev
```

Open http://localhost:5173. The app proxies `/api` to `http://localhost:8000`, so run the backend with:

```bash
cd ../backend && uvicorn main:app --reload --port 8000
```

## Features

1. **Expertise toggle (Expert / Non-Expert)**  
   Stored in `localStorage`; sent to the backend as `expertise` for tailored XAI responses.

2. **Dashboard**  
   - **Prediction model:** Loan Approval, Corporate Bankruptcy, Credit Risk  
   - **Explainability method:** SHAP, LIME, DiCE  
   - **Feature vector:** Comma- or space-separated values (must match the selected dataset’s feature count from training).  
   - **Get prediction** → shows Approve/Deny and probability.  
   - **Run X analysis** → shows:
     - **SHAP:** Horizontal bar chart of feature importance (expert: technical names; non-expert: plain-language labels).
     - **LIME:** Local explanation bar chart.
     - **DiCE:** What-if scenarios (changes that would flip the decision).

3. **Survey**  
   - **S-TIAS** (Trust in Automation): 5 items, 1–7 Likert.  
   - **SCS** (Subjective Clarity/Satisfaction): 5 items, 1–7 Likert.  
   Submit logs payload to console (you can later send to your backend).

## Build

```bash
npm run build
npm run preview
```

## Project layout

```
frontend/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── README.md
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── index.css
    ├── api/
    │   └── client.js       # predict, getXAI, health, getDatasets
    ├── context/
    │   └── ExpertiseContext.jsx
    ├── components/
    │   ├── ExpertiseToggle.jsx
    │   ├── SHAPChart.jsx
    │   ├── LIMEChart.jsx
    │   └── DiCEPanel.jsx
    └── pages/
        ├── Dashboard.jsx
        └── Survey.jsx
```
