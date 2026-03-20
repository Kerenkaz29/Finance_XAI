import React, { useState, useCallback, useEffect } from 'react'
import { SHAPChart } from '../components/SHAPChart'
import { LIMEChart } from '../components/LIMEChart'
import { DiCEPanel } from '../components/DiCEPanel'
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
  const [bankruptcyDetails, setBankruptcyDetails] = useState(null)
  const [creditIndex, setCreditIndex] = useState(0)
  const [creditDetails, setCreditDetails] = useState(null)
  const [apiReady, setApiReady] = useState(false)
  const hasValue = (v) => v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && Number.isNaN(v))
  const isPositiveNumber = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0
  const hasValidLoanStoryData = (sample) => {
    const raw = sample?.raw || {}
    return Boolean(sample?.loan_id)
      && isPositiveNumber(Number(raw.ApplicantIncome))
      && isPositiveNumber(Number(raw.CoapplicantIncome))
      && hasValue(raw.Loan_Amount_Term)
      && hasValue(raw.Credit_History)
  }
  const hasValidBankruptcyStoryData = (sample) => {
    const raw = sample?.raw || {}
    const numericVals = Object.entries(raw)
      .filter(([k, v]) => !['company_name', 'year', 'status_label', 'Bankrupt?'].includes(k) && typeof v === 'number' && Number.isFinite(v))
      .map(([, v]) => Number(v))
    const nonZeroCount = numericVals.filter((v) => v !== 0).length
    return Boolean(sample?.company_name) && nonZeroCount >= 3
  }
  const hasValidCreditStoryData = (sample) => {
    const raw = sample?.raw || {}
    return Boolean(sample?.index !== undefined)
      && isPositiveNumber(Number(raw.MonthlyIncome))
      && isPositiveNumber(Number(raw.age))
      && isPositiveNumber(Number(raw.NumberOfOpenCreditLinesAndLoans))
  }

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
    let active = true
    const targetDataset = dataset

    setFeatures(DEMO_FEATURES[dataset].join(', '))
    setPrediction(null)
    setXaiData(null)
    setError(null)
    setLoanDetails(null)
    setBankruptcyDetails(null)
    setCreditDetails(null)
    if (!apiReady) return () => { active = false }
    if (dataset === 'loan') {
      getLoanSamples(30)
        .then(async (data) => {
          if (!active || targetDataset !== 'loan') return null
          const ids = data.loan_ids || []
          if (!ids.length) return null
          let fallbackSample = null
          for (const id of ids) {
            try {
              const sample = await getLoanSample(id)
              if (!fallbackSample) fallbackSample = sample
              if (hasValidLoanStoryData(sample)) {
                return sample
              }
            } catch {
              // Try next sample ID.
            }
          }
          return fallbackSample
        })
        .then((sample) => {
          if (!active || targetDataset !== 'loan') return
          if (sample?.features?.length) {
            setLoanId(sample.loan_id || '')
            setFeatures(sample.features.join(', '))
            setLoanDetails({
              loan_id: sample.loan_id,
              loan_status: sample.loan_status,
              raw: sample.raw || {},
            })
          }
        })
        .catch(() => {
          if (!active || targetDataset !== 'loan') return
          setFeatures(DEMO_FEATURES.loan.join(', '))
        })
    } else if (dataset === 'bankruptcy') {
      getBankruptcySamples(30)
        .then(async (data) => {
          if (!active || targetDataset !== 'bankruptcy') return null
          const names = data.company_names || []
          if (!names.length) return null
          let fallbackSample = null
          for (const name of names) {
            try {
              const sample = await getBankruptcySample(name)
              if (!fallbackSample) fallbackSample = sample
              if (hasValidBankruptcyStoryData(sample)) {
                return sample
              }
            } catch {
              // Try next company.
            }
          }
          return fallbackSample
        })
        .then((sample) => {
          if (!active || targetDataset !== 'bankruptcy') return
          if (sample?.features?.length) {
            setCompanyName(sample.company_name || '')
            setFeatures(sample.features.join(', '))
            setBankruptcyDetails({
              company_name: sample.company_name,
              year: sample.year,
              status_label: sample.status_label,
              raw: sample.raw || {},
            })
          } else {
            setFeatures(DEMO_FEATURES.bankruptcy.join(', '))
          }
        })
        .catch(() => {
          if (!active || targetDataset !== 'bankruptcy') return
          setFeatures(DEMO_FEATURES.bankruptcy.join(', '))
          setBankruptcyDetails(null)
        })
    } else if (dataset === 'credit_risk') {
      getCreditSamples(30)
        .then(async (data) => {
          if (!active || targetDataset !== 'credit_risk') return null
          const indices = data.indices || []
          if (!indices.length) return null
          let fallbackSample = null
          for (const idx of indices) {
            try {
              const sample = await getCreditSample(idx)
              if (!fallbackSample) fallbackSample = sample
              if (hasValidCreditStoryData(sample)) {
                return sample
              }
            } catch {
              // Try next credit row.
            }
          }
          return fallbackSample
        })
        .then((sample) => {
          if (!active || targetDataset !== 'credit_risk') return
          if (sample?.features?.length) {
            setCreditIndex(Number.isFinite(sample.index) ? sample.index : 0)
            setFeatures(sample.features.join(', '))
          }
          if (sample?.raw) {
            setCreditDetails({
              index: sample.index,
              seriousDlq: sample.serious_dlq,
              raw: sample.raw,
            })
          } else {
            setCreditDetails(null)
          }
        })
        .catch(() => {
          if (!active || targetDataset !== 'credit_risk') return
          setFeatures(DEMO_FEATURES.credit_risk.join(', '))
          setCreditDetails(null)
        })
    }
    return () => {
      active = false
    }
  }, [dataset, apiReady])

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
      const expected = featureCounts?.[dataset]
      if (Number.isFinite(expected) && f.length !== expected) {
        setFeatures(DEMO_FEATURES[dataset].join(', '))
        throw new Error(`Wrong feature count: this model expects ${expected} features but you sent ${f.length}.`)
      }
      const [predRes, xaiRes] = await Promise.all([
        predict({ dataset, features: f, model_type: 'rf' }),
        getXAI({ dataset, features: f, expertise: 'expert', method, model_type: 'rf' }),
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
  }, [dataset, features, method, loanId, featureCounts])

  const runXAI = useCallback(async () => {
    setError(null)
    setLoading(true)
    setXaiData(null)
    try {
      const f = features.split(/[\s,]+/).map((s) => parseFloat(s.trim())).filter((n) => !Number.isNaN(n))
      if (!f.length) {
        throw new Error('Load a sample from the dataset first.')
      }
      const expected = featureCounts?.[dataset]
      if (Number.isFinite(expected) && f.length !== expected) {
        setFeatures(DEMO_FEATURES[dataset].join(', '))
        throw new Error(`Wrong feature count: this model expects ${expected} features but you sent ${f.length}.`)
      }
      const res = await getXAI({
        dataset,
        features: f,
        expertise: 'expert',
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
  }, [dataset, features, method, featureCounts])

  const isExpert = true
  const probabilityLabelsByDataset = {
    loan: ['Denied', 'Approved'],
    bankruptcy: ['Alive', 'Bankrupt'],
    credit_risk: ['No default', 'Default'],
  }
  const [negativeLabel, positiveLabel] = probabilityLabelsByDataset[dataset] || ['Class 0', 'Class 1']
  const positiveProbability = Number.isFinite(prediction?.probability)
    ? Math.min(1, Math.max(0, Number(prediction.probability)))
    : null
  const negativeProbability = positiveProbability == null ? null : 1 - positiveProbability
  const title = isExpert ? 'Expert Analysis Dashboard' : 'Simplified Analysis Interface'
  const subtitle = isExpert
    ? 'Advanced model interpretation tools for financial professionals. Select your prediction model and explainability method to generate detailed technical insights.'
    : 'User-friendly explanations of AI predictions. Choose your prediction type and explanation style to understand how decisions are made.'
  const formatValue = (value) => {
    if (value == null || value === '') return 'N/A'
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) return 'N/A'
      if (Math.abs(value) >= 1000) return value.toLocaleString()
      if (Math.abs(value) < 1) return value.toFixed(3)
      return value.toFixed(2)
    }
    return String(value)
  }
  const predictionPct = positiveProbability == null ? null : (positiveProbability * 100).toFixed(1)
  const loanRaw = loanDetails?.raw || {}
  const bankruptcyRaw = bankruptcyDetails?.raw || {}
  const creditRaw = creditDetails?.raw || {}
  const decisionStory = (() => {
    if (!prediction || predictionPct == null) return null
    if (dataset === 'loan') {
      const applicantIncome = loanRaw.ApplicantIncome != null ? formatValue(loanRaw.ApplicantIncome) : 'N/A'
      const coapplicantIncome = loanRaw.CoapplicantIncome != null ? formatValue(loanRaw.CoapplicantIncome) : 'N/A'
      const loanTerm = loanRaw.Loan_Amount_Term != null ? formatValue(loanRaw.Loan_Amount_Term) : 'N/A'
      const creditHistory = loanRaw.Credit_History != null ? formatValue(loanRaw.Credit_History) : 'N/A'
      return `Loan ID ${loanDetails?.loan_id || loanId} shows applicant income ${applicantIncome}, coapplicant income ${coapplicantIncome}, loan term ${loanTerm}, and credit history ${creditHistory}. The model concluded "${prediction.prediction_label}" with confidence ${predictionPct}%.`
    }
    if (dataset === 'bankruptcy') {
      const company = bankruptcyDetails?.company_name || companyName
      const yearPart = bankruptcyDetails?.year != null ? ` in year ${bankruptcyDetails.year}` : ''
      const metricPreview = Object.entries(bankruptcyRaw)
        .filter(([k, v]) => !['company_name', 'year', 'status_label', 'Bankrupt?'].includes(k) && typeof v === 'number' && Number.isFinite(v))
        .slice(0, 2)
        .map(([k, v]) => `${k.replace(/_/g, ' ')} ${formatValue(v)}`)
        .join(', ')
      return `Company ${company}${yearPart} was assessed using financial indicators${metricPreview ? ` including ${metricPreview}` : ''}. The model decision is "${prediction.prediction_label}" with confidence ${predictionPct}%.`
    }
    const age = hasValue(creditRaw.age) ? formatValue(creditRaw.age) : 'N/A'
    const income = hasValue(creditRaw.MonthlyIncome) ? formatValue(creditRaw.MonthlyIncome) : 'N/A'
    const openLines = hasValue(creditRaw.NumberOfOpenCreditLinesAndLoans) ? formatValue(creditRaw.NumberOfOpenCreditLinesAndLoans) : 'N/A'
    return `Credit record #${creditDetails?.index ?? creditIndex} shows age ${age}, monthly income ${income}, and open credit lines ${openLines}. The model predicts "${prediction.prediction_label}" with confidence ${predictionPct}%.`
  })()

  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-50/60 via-cyan-50/20 to-white">
      <header className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 shadow-sm">
        <div className="mx-auto max-w-6xl px-4 py-5">
          <h1 className="font-display text-3xl font-extrabold tracking-tight text-white drop-shadow-sm md:text-4xl">
            Financial Model Explainability Platform
          </h1>
          <p className="mt-1 text-sm text-emerald-50/95">
            Advanced AI Interpretability Tools for Economic Finance
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">
        <section className="mb-6 rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <h2 className="font-display mb-1 text-lg font-semibold text-slate-900 md:text-xl">
            {title}
          </h2>
          <p className="mb-4 text-sm text-slate-600">
            {subtitle}
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-600">Prediction Model</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-200"
              >
                {Object.entries(DATASET_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-slate-600">Explainability Method</label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-200"
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
              className="rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:from-emerald-700 hover:to-teal-700 disabled:opacity-50"
            >
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </div>
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        </section>

        {prediction && (
          <section className="mb-6 rounded-2xl border-2 border-emerald-200/80 bg-gradient-to-r from-emerald-50/40 to-cyan-50/30 p-6 shadow-sm ring-1 ring-emerald-100/80">
            <h2 className="font-display mb-4 text-lg font-bold text-slate-900 md:text-xl">
              {dataset === 'loan' ? 'Loan Details' : dataset === 'credit_risk' ? 'Credit Details' : 'Company Details'}
            </h2>
            <div className="flex flex-wrap items-baseline gap-x-8 gap-y-1">
              {dataset === 'loan' && loanDetails && (
                <div className="w-full">
                  <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr] md:items-start">
                    <div className="rounded-xl border border-slate-200 bg-white/70 p-4">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Details</p>
                      {decisionStory && (
                        <p className="mt-1 text-sm leading-relaxed text-slate-700">
                          {decisionStory}
                        </p>
                      )}
                    </div>
                    <div className="hidden self-stretch text-center text-2xl font-semibold text-slate-300 md:flex md:items-center md:justify-center">|</div>
                    <div className="rounded-xl border border-slate-200 bg-white/70 p-4">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Decision</p>
                      <p className="mt-1 text-sm text-slate-700">
                        Status{' '}
                        <span className={`font-bold ${prediction.prediction === 1 ? 'text-emerald-600' : 'text-red-600'}`}>
                          {prediction.prediction_label}
                        </span>
                      </p>
                      <p className="mt-1 text-sm text-slate-700">
                        Probability{' '}
                        <span className="font-semibold text-slate-900">{predictionPct}%</span>
                      </p>
                    </div>
                  </div>
                </div>
              )}
              {dataset === 'bankruptcy' && (
                <div className="w-full">
                  <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr] md:items-start">
                    <div className="rounded-xl border border-slate-200 bg-white/70 p-4">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Details</p>
                      {decisionStory && (
                        <p className="mt-1 text-sm leading-relaxed text-slate-700">
                          {decisionStory}
                        </p>
                      )}
                    </div>
                    <div className="hidden self-stretch text-center text-2xl font-semibold text-slate-300 md:flex md:items-center md:justify-center">|</div>
                    <div className="rounded-xl border border-slate-200 bg-white/70 p-4">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Decision</p>
                      <p className="mt-1 text-sm text-slate-700">
                        Status{' '}
                        <span className={`font-bold ${prediction.prediction === 1 ? 'text-emerald-600' : 'text-red-600'}`}>
                          {prediction.prediction_label}
                        </span>
                      </p>
                      <p className="mt-1 text-sm text-slate-700">
                        Probability{' '}
                        <span className="font-semibold text-slate-900">{predictionPct}%</span>
                      </p>
                    </div>
                  </div>
                </div>
              )}
              {dataset === 'credit_risk' && (
                <div className="w-full">
                  <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr] md:items-start">
                    <div className="rounded-xl border border-slate-200 bg-white/70 p-4">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Details</p>
                      {decisionStory && (
                        <p className="mt-1 text-sm leading-relaxed text-slate-700">
                          {decisionStory}
                        </p>
                      )}
                    </div>
                    <div className="hidden self-stretch text-center text-2xl font-semibold text-slate-300 md:flex md:items-center md:justify-center">|</div>
                    <div className="rounded-xl border border-slate-200 bg-white/70 p-4">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Decision</p>
                      <p className="mt-1 text-sm text-slate-700">
                        Status{' '}
                        <span className={`font-bold ${prediction.prediction === 1 ? 'text-emerald-600' : 'text-red-600'}`}>
                          {prediction.prediction_label}
                        </span>
                      </p>
                      <p className="mt-1 text-sm text-slate-700">
                        Probability{' '}
                        <span className="font-semibold text-slate-900">
                          {predictionPct}%
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        <section className="rounded-2xl border-2 border-slate-200 bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <h2 className="font-display mb-1 text-lg font-bold text-slate-900 md:text-xl">
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
                <div className="rounded-lg border-2 border-slate-200 bg-white p-4 shadow-sm ring-1 ring-slate-100">
                  <h3 className="font-display mb-3 text-sm font-semibold text-gray-700">
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
                <div className="rounded-lg border-2 border-slate-200 bg-white p-4 shadow-sm ring-1 ring-slate-100">
                  <h3 className="font-display mb-3 text-sm font-semibold text-gray-700">
                    {isExpert ? 'LIME local explanation (Expert View)' : 'Local explanation (Non-Expert View)'}
                  </h3>
                  <img
                    src={xaiData.image_url + (xaiData._ts ? `?t=${xaiData._ts}` : '')}
                    alt="LIME local explanation"
                    className="max-w-full rounded border border-gray-100"
                  />
                  {prediction && positiveProbability != null && (
                    <div className="mt-5 border-t border-gray-200 pt-4">
                      <div className="mx-auto w-full max-w-xl rounded-2xl border-2 border-slate-200 bg-gradient-to-b from-white via-slate-50 to-white p-6 shadow-sm ring-1 ring-slate-100">
                        <div>
                          <div>
                            <h3 className="text-base font-semibold tracking-wide text-slate-800">Class probabilities</h3>
                            <p className="mt-0.5 text-xs text-slate-500">Model confidence by class</p>
                          </div>
                        </div>
                        <div className="mt-4 space-y-4">
                          <div>
                            <div className="mb-1.5 grid grid-cols-[1fr_auto] items-center text-sm text-gray-700">
                              <span className="font-medium">{negativeLabel}</span>
                              <span className="border-l border-slate-200 pl-3 font-semibold tabular-nums text-slate-700">
                                {(negativeProbability * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="h-3.5 overflow-hidden rounded-full border border-blue-100 bg-blue-50">
                              <div
                                className="h-full rounded-full bg-blue-500 transition-all"
                                style={{ width: `${(negativeProbability * 100).toFixed(2)}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="mb-1.5 grid grid-cols-[1fr_auto] items-center text-sm text-gray-700">
                              <span className="font-medium">{positiveLabel}</span>
                              <span className="border-l border-slate-200 pl-3 font-semibold tabular-nums text-slate-700">
                                {(positiveProbability * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="h-3.5 overflow-hidden rounded-full border border-orange-100 bg-orange-50">
                              <div
                                className="h-full rounded-full bg-orange-500 transition-all"
                                style={{ width: `${(positiveProbability * 100).toFixed(2)}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <LIMEChart data={xaiData} title={isExpert ? 'LIME local explanation (Expert View)' : 'Local explanation (Non-Expert View)'} />
              )}
            </>
          )}
          {method === 'DiCE' && <DiCEPanel data={xaiData} />}
          {!xaiData && !loading && (
            <p className="rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 p-4 text-center text-sm text-gray-500">
              Choose options above and run analysis to see the chart.
            </p>
          )}
        </section>
      </main>
    </div>
  )
}
