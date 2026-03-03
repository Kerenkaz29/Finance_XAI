import React, { useState, useCallback, useEffect } from 'react'
import { ExpertiseToggle } from '../components/ExpertiseToggle'
import { SHAPChart } from '../components/SHAPChart'
import { LIMEChart } from '../components/LIMEChart'
import { DiCEPanel } from '../components/DiCEPanel'
import { useExpertise } from '../context/ExpertiseContext'
import { predict, getXAI, getDatasets, getLoanSample, getBankruptcySample, getCreditSample, getLoanSamples, getBankruptcySamples, getCreditSamples } from '../api/client'

const DATASET_LABELS = {
  loan: 'Loan Approval Prediction',
  bankruptcy: 'Corporate Bankruptcy Prediction',
  credit_risk: 'Credit Risk Prediction',
}
const METHOD_OPTIONS = ['SHAP', 'LIME', 'DiCE']

// Demo feature vectors per dataset (length must match preprocessed features)
// Bankruptcy: 10 Altman-Z-based features =
//   Working Capital/TA, Retained Earnings/TA, ROA(C), Net worth/Assets,
//   Total Asset Turnover, Debt ratio %, Cash Flow/TA,
//   Interest Coverage Ratio, Current Ratio, Borrowing dependency
const DEMO_FEATURES = {
  loan: [1, 0, 0, 5849, 0, 128, 360, 1, 2, 0, 1, 1],
  bankruptcy: [0.15, 0.12, 0.04, 0.55, 0.65, 0.45, 0.06, 3.5, 1.8, 0.30],
  credit_risk: [0.5, 35, 0, 0.2, 5000, 5, 0, 0, 2],
}

function makeDemoVector(n) {
  return Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 1 : 0.5))
}

export default function Dashboard() {
  const { mode } = useExpertise()
  const [dataset, setDataset] = useState('loan')
  const [method, setMethod] = useState('SHAP')
  const [features, setFeatures] = useState(DEMO_FEATURES.loan.join(', '))
  const [featureCounts, setFeatureCounts] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [xaiData, setXaiData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [loanId, setLoanId] = useState('LP001002')
  const [loanDetails, setLoanDetails] = useState(null)
  const [companyName, setCompanyName] = useState('C_1')
  const [creditIndex, setCreditIndex] = useState(0)
  const [apiReady, setApiReady] = useState(false)

  // Fetch datasets and feature counts on load; detect if backend is reachable
  useEffect(() => {
    getDatasets()
      .then((data) => {
        setFeatureCounts(data.feature_counts || {})
        setApiReady(true)
        setError(null)
      })
      .catch(() => {
        setApiReady(false)
        setError('Cannot reach backend. Is it running on port 8000?')
      })
  }, [])

  // When dataset changes: load list of samples and auto-load first sample (cvision-style so data is visible)
  useEffect(() => {
    setFeatures('')
    setPrediction(null)
    setXaiData(null)
    setError(null)
    if (!apiReady) return
    if (dataset === 'loan') {
      getLoanSamples(30)
        .then((data) => {
          const ids = data.loan_ids || []
          if (ids.length) {
            setLoanId(ids[0])
            return getLoanSample(ids[0])
          }
        })
        .then((sample) => {
          if (sample?.features?.length) {
            setFeatures(sample.features.join(', '))
            setLoanDetails({ loan_id: sample.loan_id, loan_status: sample.loan_status })
          }
        })
        .catch(() => {})
    } else if (dataset === 'bankruptcy') {
      getBankruptcySamples(30)
        .then((data) => {
          const names = data.company_names || []
          if (names.length) {
            setCompanyName(names[0])
            return getBankruptcySample(names[0])
          }
        })
        .then((sample) => {
          if (sample?.features?.length) {
            setFeatures(sample.features.join(', '))
          } else {
            setFeatures(DEMO_FEATURES.bankruptcy.join(', '))
          }
        })
        .catch(() => {
          setFeatures(DEMO_FEATURES.bankruptcy.join(', '))
        })
    } else if (dataset === 'credit_risk') {
      getCreditSamples(30)
        .then((data) => {
          const indices = data.indices || []
          if (indices.length) {
            setCreditIndex(indices[0])
            return getCreditSample(indices[0])
          }
        })
        .then((sample) => {
          if (sample?.features?.length) setFeatures(sample.features.join(', '))
        })
        .catch(() => {})
    }
  }, [dataset, apiReady])

  // When switching Expert ↔ Non-Expert, clear results so the screen resets
  useEffect(() => {
    setPrediction(null)
    setXaiData(null)
    setError(null)
  }, [mode])

  const runAnalyze = useCallback(async () => {
    setError(null)
    setLoading(true)
    setPrediction(null)
    setXaiData(null)
    try {
      const f = features.split(/[\s,]+/).map((s) => parseFloat(s.trim())).filter((n) => !Number.isNaN(n))
      if (!f.length) {
        throw new Error('Load a sample from the dataset first.')
      }
      const [predRes, xaiRes] = await Promise.all([
        predict({ dataset, features: f, model_type: 'rf' }),
        getXAI({ dataset, features: f, expertise: mode, method, model_type: 'rf' }),
      ])
      setPrediction(predRes)
      setXaiData({ ...xaiRes, _ts: Date.now() })
      if (dataset === 'loan') {
        setLoanDetails((prev) => ({ ...(prev || {}), loan_id: loanId, loan_status: predRes.prediction_label }))
      }
    } catch (e) {
      let msg = e.message
      const match = msg.match(/Expected (\d+) features, got (\d+)/)
      if (match) msg = `Wrong feature count: this model expects ${match[1]} features but you sent ${match[2]}. Load a sample first.`
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [dataset, features, mode, method, loanId])

  const runXAI = useCallback(async () => {
    setError(null)
    setLoading(true)
    setXaiData(null)
    try {
      const f = features.split(/[\s,]+/).map((s) => parseFloat(s.trim())).filter((n) => !Number.isNaN(n))
      if (!f.length) {
        throw new Error('Load a sample from the dataset first.')
      }
      const res = await getXAI({
        dataset,
        features: f,
        expertise: mode,
        method,
        model_type: 'rf',
      })
      setXaiData(res)
    } catch (e) {
      let msg = e.message
      const match = msg.match(/Expected (\d+) features, got (\d+)/)
      if (match) msg = `Wrong feature count: this model expects ${match[1]} features but you sent ${match[2]}. Enter ${match[1]} numbers.`
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [dataset, features, mode, method])

  const isExpert = mode === 'expert'
  const title = isExpert ? 'Expert Analysis Dashboard' : 'Simplified Analysis Interface'
  const subtitle = isExpert
    ? 'Advanced model interpretation tools for financial professionals. Select your prediction model and explainability method to generate detailed technical insights.'
    : 'User-friendly explanations of AI predictions. Choose your prediction type and explanation style to understand how decisions are made.'

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white">
        <div className="mx-auto max-w-6xl px-4 py-4">
          <h1 className="text-2xl font-bold text-slate-900">
            Financial Model Explainability Platform
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Advanced AI Interpretability Tools for Economic Finance
          </p>
          <ExpertiseToggle />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        <section className="mb-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="mb-1 text-lg font-semibold text-slate-900">
            {isExpert ? 'Expert Analysis Dashboard' : 'Simplified Analysis Interface'}
          </h2>
          <p className="mb-4 text-sm text-slate-600">
            {isExpert
              ? 'Advanced model interpretation tools for financial professionals. Select your prediction model and explainability method to generate detailed technical insights.'
              : 'User-friendly explanations of AI predictions. Choose your prediction type and explanation style to understand how decisions are made.'}
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium text-gray-700">Prediction Model</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              >
                {Object.entries(DATASET_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-gray-700">Explainability Method</label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
              >
                {METHOD_OPTIONS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="mt-4 flex gap-2">
            <button
              type="button"
              onClick={runAnalyze}
              disabled={loading}
              className="rounded-lg bg-primary-700 px-4 py-2 text-sm font-medium text-white hover:bg-primary-800 disabled:opacity-50"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        </section>

        {prediction && (
          <section className="mb-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <h2 className="mb-4 text-lg font-bold text-slate-900">
              {dataset === 'loan' ? 'Loan Details' : dataset === 'credit_risk' ? 'Credit Details' : 'Company Details'}
            </h2>
            <div className="flex flex-wrap items-baseline gap-x-8 gap-y-1">
              {dataset === 'loan' && loanDetails && (
                <p className="text-sm text-gray-700">
                  Loan ID: <span className="font-bold text-slate-900">{loanDetails.loan_id}</span>
                </p>
              )}
              {dataset === 'bankruptcy' && (
                <p className="text-sm text-gray-700">
                  Company: <span className="font-bold text-slate-900">{companyName}</span>
                </p>
              )}
              {dataset === 'credit_risk' && (
                <p className="text-sm text-gray-700">
                  Record: <span className="font-bold text-slate-900">#{creditIndex}</span>
                </p>
              )}
              <p className="text-sm text-gray-700">
                Status: <span className={`font-bold ${prediction.prediction === 1 ? 'text-emerald-600' : 'text-red-600'}`}>{prediction.prediction_label}</span>
              </p>
            </div>
          </section>
        )}

        <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="mb-1 text-lg font-bold text-slate-900">
            {method === 'SHAP' ? 'SHAP Analysis' : method === 'LIME' ? 'LIME Analysis' : method === 'DiCE' ? 'DiCE Counterfactuals' : 'Explanation'}
          </h2>
          <p className="mb-4 text-sm text-gray-600">
            {method === 'SHAP'
              ? (dataset === 'loan'
                ? 'SHapley Additive eXplanations showing feature contribution to the loan approval decision:'
                : xaiData?.description || 'SHapley Additive eXplanations showing feature contribution to the decision.')
              : xaiData?.description
              ? xaiData.description
              : method === 'LIME'
              ? 'Local explanation of how each factor affected this decision.'
              : 'What could change this decision.'}
          </p>
          {method === 'SHAP' && (
            <>
              {xaiData?.image_url ? (
                <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                  <h3 className="mb-3 text-sm font-semibold text-gray-700">
                    {isExpert ? 'Global Feature Importance - Loan Approval Model (Expert View)' : 'What most affected the loan decision (Non-Expert View)'}
                  </h3>
                  <img
                    src={xaiData.image_url + (xaiData._ts ? `?t=${xaiData._ts}` : '')}
                    alt="SHAP feature importance"
                    className="max-w-full rounded border border-gray-100"
                  />
                </div>
              ) : (
                <SHAPChart data={xaiData} title={isExpert ? 'Global Feature Importance - Loan Approval Model (Expert View)' : 'What most affected the loan decision (Non-Expert View)'} />
              )}
            </>
          )}
          {method === 'LIME' && (
            <>
              {xaiData?.image_url ? (
                <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                  <h3 className="mb-3 text-sm font-semibold text-gray-700">
                    {isExpert ? 'LIME local explanation (Expert View)' : 'Local explanation (Non-Expert View)'}
                  </h3>
                  <img
                    src={xaiData.image_url + (xaiData._ts ? `?t=${xaiData._ts}` : '')}
                    alt="LIME local explanation"
                    className="max-w-full rounded border border-gray-100"
                  />
                </div>
              ) : (
                <LIMEChart data={xaiData} title={isExpert ? 'LIME local explanation (Expert View)' : 'Local explanation (Non-Expert View)'} />
              )}
            </>
          )}
          {method === 'DiCE' && <DiCEPanel data={xaiData} />}
          {!xaiData && !loading && (
            <p className="rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4 text-center text-sm text-gray-500">
              Choose options above and run analysis to see the chart.
            </p>
          )}
        </section>
      </main>
    </div>
  )
}
