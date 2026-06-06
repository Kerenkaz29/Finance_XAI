# Frontend — User Interface

Interactive dashboard for the XAI Financial Services project: prediction + SHAP/LIME/DiCE visualizations for loan approval, bankruptcy, and credit risk models.

## Tech Stack

- **React 18** + **Vite**
- **Tailwind CSS** for styling
- **Recharts** for fallback bar charts (when backend PNGs are unavailable)
- **React Router** for navigation

## Setup

```bash
cd frontend
npm install
```

Copy `.env.example` if you need a custom API URL:

```bash
# Optional — defaults to /api (proxied to localhost:8000 in dev)
VITE_API_URL=http://localhost:8000
```

## Run

```bash
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` to `http://localhost:8000`, so run the backend with:

```bash
cd ../backend && uvicorn main:app --reload --port 8000
```

On first visit, a loading overlay may appear while the backend downloads model weights from Google Drive.

## Features

1. **Dashboard**
   - **Prediction model:** Loan Approval, Corporate Bankruptcy, Credit Risk
   - **Explainability method:** SHAP, LIME, DiCE
   - **Auto sample loading:** On dataset change, loads real records from the backend and fills the feature vector
   - **Run Analysis:** Runs prediction + XAI in one step
   - **SHAP/LIME:** Shows backend-generated PNG when available; falls back to Recharts bar charts
   - **DiCE:** Counterfactual scenarios with optional Gemini-generated explanations

2. **Expertise mode**
   - The UI currently runs in **expert mode** (`expertise: "expert"` sent to the API).
   - `ExpertiseContext` and `ExpertiseToggle` are available for a future expert/non-expert toggle.

3. **Survey** (`src/pages/Survey.jsx`)
   - Embedded Google Form for research feedback. Not currently linked in the router; add a route in `App.jsx` to expose it.

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
├── vite.config.js       # Dev server + /api proxy to backend
├── tailwind.config.js
├── postcss.config.js
├── .env.example
├── README.md
└── src/
    ├── main.jsx         # React entry point
    ├── App.jsx          # Routing, model-download gate
    ├── index.css
    ├── api/
    │   └── client.js    # predict, getXAI, samples, getReady
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
