/**
 * Main dashboard: dataset/method selection, sample loading, prediction + XAI visualization.
 * Currently runs in expert mode (expertise sent as "expert" to the API).
 */
import React, { useState, useCallback, useEffect, useRef } from 'react'
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

// Fallback feature vectors when CSV samples are unavailable (lengths must match trained models).
const DEMO_FEATURES = {
  loan: [1, 0, 0, 5849, 0, 128, 360, 1, 2, 0, 1],
  bankruptcy: [0.15, 0.12, 0.04, 0.55, 0.65, 0.45, 0.06, 3.5, 1.8, 0.30],
  credit_risk: [0.5, 35, 0, 0.2, 5000, 5, 0, 0, 2, 1],
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
  const [detailComparisons, setDetailComparisons] = useState({ loan: [], bankruptcy: [], credit_risk: [] })
  const [apiReady, setApiReady] = useState(false)
  const [sampleLoading, setSampleLoading] = useState(false)
  const selectedSampleRef = useRef({ loan: null, bankruptcy: null, credit_risk: null })

  // --- Sample validation (pick records suitable for the comparison table) ---

  const hasValue = (v) => v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && Number.isNaN(v))
  const isPositiveNumber = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0
  const hasValidLoanStoryData = (sample) => {
    const raw = sample?.raw || {}
    return Boolean(sample?.loan_id)
      && isPositiveNumber(Number(raw.ApplicantIncome))
      && isPositiveNumber(Number(raw.CoapplicantIncome))
      && isPositiveNumber(Number(raw.LoanAmount))
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
  const getDatasetStatusLabel = (datasetName, sample) => {
    if (!sample) return null
    if (datasetName === 'loan') {
      const rawStatus = String(sample?.loan_status ?? sample?.raw?.Loan_Status ?? '').trim().toUpperCase()
      if (rawStatus === 'Y' || rawStatus.includes('APPROV')) return 'Approved'
      if (rawStatus === 'N' || rawStatus.includes('DENIED') || rawStatus.includes('REJECT')) return 'Not Approved'
      return null
    }
    if (datasetName === 'bankruptcy') {
      const rawStatus = String(sample?.status_label ?? sample?.raw?.status_label ?? sample?.raw?.['Bankrupt?'] ?? '').trim().toLowerCase()
      if (rawStatus.includes('alive') || rawStatus === '0') return 'Alive'
      if (rawStatus.includes('bankrupt') || rawStatus === '1') return 'Bankrupt'
      return null
    }
    const serious = Number(sample?.serious_dlq ?? sample?.raw?.SeriousDlqin2yrs)
    if (!Number.isFinite(serious)) return null
    return serious >= 1 ? 'Higher Payment Risk' : 'Lower Payment Risk'
  }
  const pickComparisonSamples = (datasetName, samples) => {
    const valid = (samples || []).filter(Boolean)
    if (!valid.length) return []
    const desiredPair = {
      loan: ['Approved', 'Not Approved'],
      bankruptcy: ['Alive', 'Bankrupt'],
      credit_risk: ['Lower Payment Risk', 'Higher Payment Risk'],
    }[datasetName] || [null, null]
    const favorable = valid.find((s) => getDatasetStatusLabel(datasetName, s) === desiredPair[0]) || null
    const adverse = valid.find((s) => getDatasetStatusLabel(datasetName, s) === desiredPair[1] && s !== favorable) || null
    if (favorable && adverse) return [favorable, adverse]
    return valid.slice(0, 2)
  }

  // --- Effects: backend connectivity + auto-load samples on dataset change ---

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
    setDetailComparisons((prev) => ({ ...prev, [dataset]: [] }))
    if (!apiReady) {
      setSampleLoading(false)
      return () => { active = false }
    }
    setSampleLoading(true)
    if (dataset === 'loan') {
      getLoanSamples(30)
        .then(async (data) => {
          if (!active || targetDataset !== 'loan') return null
          const ids = data.loan_ids || []
          if (!ids.length) return null
          let fallbackSample = null
          const validSamples = []
          const fetchedSamples = []
          for (const id of ids) {
            try {
              const sample = await getLoanSample(id)
              fetchedSamples.push(sample)
              if (!fallbackSample) fallbackSample = sample
              if (hasValidLoanStoryData(sample)) {
                validSamples.push(sample)
              }
            } catch {
              // Try next sample ID.
            }
          }
          if (validSamples.length) {
            const savedId = selectedSampleRef.current.loan
            const savedSample = savedId
              ? validSamples.find((s) => s?.loan_id === savedId)
              : null
            const chosen = savedSample || validSamples[0]
            selectedSampleRef.current.loan = chosen?.loan_id || null
            return { chosen, comparisons: pickComparisonSamples('loan', fetchedSamples) }
          }
          if (fallbackSample?.loan_id) {
            selectedSampleRef.current.loan = fallbackSample.loan_id
          }
          return { chosen: fallbackSample, comparisons: pickComparisonSamples('loan', fetchedSamples) }
        })
        .then((result) => {
          if (!active || targetDataset !== 'loan') return
          const sample = result?.chosen
          setDetailComparisons((prev) => ({ ...prev, loan: result?.comparisons || [] }))
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
          setDetailComparisons((prev) => ({ ...prev, loan: [] }))
        })
        .finally(() => {
          if (!active || targetDataset !== 'loan') return
          setSampleLoading(false)
        })
    } else if (dataset === 'bankruptcy') {
      getBankruptcySamples(30)
        .then(async (data) => {
          if (!active || targetDataset !== 'bankruptcy') return null
          const names = data.company_names || []
          if (!names.length) return null
          let fallbackSample = null
          const validSamples = []
          const fetchedSamples = []
          for (const name of names) {
            try {
              const sample = await getBankruptcySample(name)
              fetchedSamples.push(sample)
              if (!fallbackSample) fallbackSample = sample
              if (hasValidBankruptcyStoryData(sample)) {
                validSamples.push(sample)
              }
            } catch {
              // Try next company.
            }
          }
          if (validSamples.length) {
            const savedName = selectedSampleRef.current.bankruptcy
            const savedSample = savedName
              ? validSamples.find((s) => s?.company_name === savedName)
              : null
            const chosen = savedSample || validSamples[0]
            selectedSampleRef.current.bankruptcy = chosen?.company_name || null
            return { chosen, comparisons: pickComparisonSamples('bankruptcy', fetchedSamples) }
          }
          if (fallbackSample?.company_name) {
            selectedSampleRef.current.bankruptcy = fallbackSample.company_name
          }
          return { chosen: fallbackSample, comparisons: pickComparisonSamples('bankruptcy', fetchedSamples) }
        })
        .then((result) => {
          if (!active || targetDataset !== 'bankruptcy') return
          const sample = result?.chosen
          setDetailComparisons((prev) => ({ ...prev, bankruptcy: result?.comparisons || [] }))
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
          setDetailComparisons((prev) => ({ ...prev, bankruptcy: [] }))
        })
        .finally(() => {
          if (!active || targetDataset !== 'bankruptcy') return
          setSampleLoading(false)
        })
    } else if (dataset === 'credit_risk') {
      getCreditSamples(30)
        .then(async (data) => {
          if (!active || targetDataset !== 'credit_risk') return null
          const indices = data.indices || []
          if (!indices.length) return null
          let fallbackSample = null
          const validSamples = []
          const fetchedSamples = []
          for (const idx of indices) {
            try {
              const sample = await getCreditSample(idx)
              fetchedSamples.push(sample)
              if (!fallbackSample) fallbackSample = sample
              if (hasValidCreditStoryData(sample)) {
                validSamples.push(sample)
              }
            } catch {
              // Try next credit row.
            }
          }
          if (validSamples.length) {
            const savedIndex = selectedSampleRef.current.credit_risk
            const savedSample = Number.isInteger(savedIndex)
              ? validSamples.find((s) => Number(s?.index) === savedIndex)
              : null
            const chosen = savedSample || validSamples[0]
            selectedSampleRef.current.credit_risk = Number.isFinite(Number(chosen?.index))
              ? Number(chosen.index)
              : null
            return { chosen, comparisons: pickComparisonSamples('credit_risk', fetchedSamples) }
          }
          if (Number.isFinite(Number(fallbackSample?.index))) {
            selectedSampleRef.current.credit_risk = Number(fallbackSample.index)
          }
          return { chosen: fallbackSample, comparisons: pickComparisonSamples('credit_risk', fetchedSamples) }
        })
        .then((result) => {
          if (!active || targetDataset !== 'credit_risk') return
          const sample = result?.chosen
          setDetailComparisons((prev) => ({ ...prev, credit_risk: result?.comparisons || [] }))
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
          setDetailComparisons((prev) => ({ ...prev, credit_risk: [] }))
        })
        .finally(() => {
          if (!active || targetDataset !== 'credit_risk') return
          setSampleLoading(false)
        })
    }
    return () => {
      active = false
    }
  }, [dataset, apiReady])

  // --- Analysis handlers ---

  const runAnalyze = useCallback(async () => {
    setError(null)
    if (sampleLoading) {
      setError('Please wait for dataset sample loading to finish, then run analysis.')
      return
    }
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
  }, [dataset, features, method, loanId, featureCounts, sampleLoading])

  const runXAI = useCallback(async () => {
    setError(null)
    if (sampleLoading) {
      setError('Please wait for dataset sample loading to finish, then run analysis.')
      return
    }
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
  }, [dataset, features, method, featureCounts, sampleLoading])

  // --- Derived UI state & formatting helpers ---

  // Hardcoded expert view; ExpertiseContext/ExpertiseToggle exist for future non-expert UI.
  const isExpert = true
  const probabilityLabelsByDataset = {
    loan: ['Denied', 'Approved'],
    bankruptcy: ['Alive', 'Bankrupt'],
    credit_risk: ['Lower Payment Risk', 'Higher Payment Risk'],
  }
  const [negativeLabel, positiveLabel] = probabilityLabelsByDataset[dataset] || ['Class 0', 'Class 1']
  const parsedFeatureCount = features
    .split(/[\s,]+/)
    .map((s) => parseFloat(s.trim()))
    .filter((n) => !Number.isNaN(n))
    .length
  const expectedFeatureCount = featureCounts?.[dataset]
  const isFeatureCountValid = !Number.isFinite(expectedFeatureCount) || parsedFeatureCount === expectedFeatureCount
  const canRunAnalyze = apiReady && !loading && !sampleLoading && parsedFeatureCount > 0 && isFeatureCountValid
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
  const formatUSD = (value) => {
    const n = Number(value)
    if (!Number.isFinite(n)) return 'N/A'
    return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
  }
  const formatLoanAmountUSD = (value) => {
    const n = Number(value)
    if (!Number.isFinite(n) || n <= 0) return 'N/A'
    return `$${Math.round(n * 1000).toLocaleString()}`
  }
  const formatCount = (value) => {
    const n = Number(value)
    if (!Number.isFinite(n)) return 'N/A'
    return Math.round(n).toLocaleString()
  }
  const formatRatioAsPercent = (value) => {
    const n = Number(value)
    if (!Number.isFinite(n)) return 'N/A'
    return `${(n * 100).toFixed(1)}%`
  }
  const formatBankruptcyMetricLabel = (rawName) => {
    const key = String(rawName || '').trim()
    const map = {
      'ROA(C) before interest and depreciation before interest': 'Operating return on assets',
      'ROA(A) before interest and % after tax': 'After-tax return on assets',
      'Debt ratio %': 'Debt ratio',
      'Net worth/Assets': 'Net worth to assets',
      'Working Capital/Total Assets': 'Working capital to total assets',
      'Total Asset Turnover': 'Asset turnover',
      'Current Ratio': 'Current ratio',
      'Cash Flow/Total Assets': 'Cash flow to total assets',
      'Interest Coverage Ratio': 'Interest coverage ratio',
      'Borrowing dependency': 'Borrowing dependency',
      'Retained Earnings/Total Assets': 'Retained earnings to total assets',
    }
    return map[key] || key.replace(/_/g, ' ')
  }
  const formatBankruptcyMetricValue = (rawName, value) => {
    const key = String(rawName || '').trim()
    const percentLike = new Set([
      'ROA(C) before interest and depreciation before interest',
      'ROA(A) before interest and % after tax',
      'Debt ratio %',
      'Net worth/Assets',
      'Working Capital/Total Assets',
      'Cash Flow/Total Assets',
      'Retained Earnings/Total Assets',
      'Borrowing dependency',
    ])
    if (percentLike.has(key)) return formatRatioAsPercent(value)
    if (key === 'Total Asset Turnover' || key === 'Current Ratio' || key === 'Interest Coverage Ratio') {
      const n = Number(value)
      return Number.isFinite(n) ? `${n.toFixed(2)}x` : 'N/A'
    }
    return formatValue(value)
  }
  const predictionPct = positiveProbability == null ? null : (positiveProbability * 100).toFixed(1)
  const loanRaw = loanDetails?.raw || {}
  const bankruptcyRaw = bankruptcyDetails?.raw || {}
  const creditRaw = creditDetails?.raw || {}
  const rawPredictionLabel = String(prediction?.prediction_label || '')
  const predictionLabelLower = String(prediction?.prediction_label || '').toLowerCase()
  const displayPredictionLabel = (() => {
    if (dataset !== 'credit_risk') return rawPredictionLabel
    if (predictionLabelLower.includes('no default')) return 'Lower Payment Risk'
    if (predictionLabelLower.includes('default')) return 'Higher Payment Risk'
    return rawPredictionLabel
  })()
  const decisionStatusClass = (() => {
    if (dataset === 'loan') {
      return predictionLabelLower.includes('approved') ? 'text-emerald-600' : 'text-red-600'
    }
    if (dataset === 'bankruptcy') {
      return predictionLabelLower.includes('bankrupt') ? 'text-red-600' : 'text-emerald-600'
    }
    // credit_risk
    return predictionLabelLower.includes('default') && !predictionLabelLower.includes('no default')
      ? 'text-red-600'
      : 'text-emerald-600'
  })()
  const decisionStory = (() => {
    if (!prediction || predictionPct == null) return null
    if (dataset === 'loan') {
      const applicantIncome = loanRaw.ApplicantIncome != null ? formatUSD(loanRaw.ApplicantIncome) : 'N/A'
      const coapplicantIncome = loanRaw.CoapplicantIncome != null ? formatUSD(loanRaw.CoapplicantIncome) : 'N/A'
      const householdIncomeNum = Number(loanRaw.ApplicantIncome || 0) + Number(loanRaw.CoapplicantIncome || 0)
      const householdIncome = householdIncomeNum > 0 ? formatUSD(householdIncomeNum) : 'N/A'
      const loanAmount = loanRaw.LoanAmount != null ? formatLoanAmountUSD(loanRaw.LoanAmount) : 'N/A'
      const loanTerm = loanRaw.Loan_Amount_Term != null ? `${formatCount(loanRaw.Loan_Amount_Term)} months` : 'N/A'
      const creditHistoryNum = Number(loanRaw.Credit_History)
      const creditHistory = Number.isFinite(creditHistoryNum)
        ? (creditHistoryNum >= 1 ? 'positive (1)' : 'limited (0)')
        : 'N/A'
      return `Loan ID ${loanDetails?.loan_id || loanId} has applicant income ${applicantIncome}, coapplicant income ${coapplicantIncome} (total ${householdIncome}), requested loan amount ${loanAmount}, and a ${loanTerm} term.`
    }
    if (dataset === 'bankruptcy') {
      const company = bankruptcyDetails?.company_name || companyName
      const yearPart = bankruptcyDetails?.year != null ? ` in year ${bankruptcyDetails.year}` : ''
      const preferredOrder = [
        'ROA(C) before interest and depreciation before interest',
        'ROA(A) before interest and % after tax',
        'Debt ratio %',
        'Current Ratio',
        'Interest Coverage Ratio',
        'Cash Flow/Total Assets',
        'Net worth/Assets',
        'Working Capital/Total Assets',
        'Total Asset Turnover',
        'Borrowing dependency',
        'Retained Earnings/Total Assets',
      ]
      const metricOrderIndex = new Map(preferredOrder.map((k, i) => [k, i]))
      const metricPreview = Object.entries(bankruptcyRaw)
        .filter(([k, v]) => !['company_name', 'year', 'status_label', 'Bankrupt?'].includes(k) && typeof v === 'number' && Number.isFinite(v))
        .sort(([a], [b]) => {
          const ai = metricOrderIndex.has(a) ? metricOrderIndex.get(a) : Number.MAX_SAFE_INTEGER
          const bi = metricOrderIndex.has(b) ? metricOrderIndex.get(b) : Number.MAX_SAFE_INTEGER
          if (ai !== bi) return ai - bi
          return a.localeCompare(b)
        })
        .slice(0, 4)
        .map(([k, v]) => `${formatBankruptcyMetricLabel(k)} ${formatBankruptcyMetricValue(k, v)}`)
        .join(', ')
      return `Company ${company}${yearPart} was assessed using key bankruptcy indicators${metricPreview ? `, including ${metricPreview}` : ''}.`
    }
    const importantCreditFields = [
      ['age', (v) => `${formatCount(v)} years`],
      ['MonthlyIncome', (v) => `${formatUSD(v)} per month`],
      ['DebtRatio', (v) => formatRatioAsPercent(v)],
      ['RevolvingUtilizationOfUnsecuredLines', (v) => formatRatioAsPercent(v)],
      ['NumberOfTime30-59DaysPastDueNotWorse', (v) => `${formatCount(v)} times`],
      ['NumberOfOpenCreditLinesAndLoans', (v) => `${formatCount(v)} lines`],
    ]
    const creditPreview = importantCreditFields
      .filter(([k]) => hasValue(creditRaw[k]))
      .slice(0, 4)
      .map(([k, formatter]) => {
        const label = k
          .replace('MonthlyIncome', 'monthly income')
          .replace('DebtRatio', 'debt ratio')
          .replace('RevolvingUtilizationOfUnsecuredLines', 'revolving utilization')
          .replace('NumberOfTime30-59DaysPastDueNotWorse', '30-59 days past due')
          .replace('NumberOfOpenCreditLinesAndLoans', 'open credit lines')
          .replace('age', 'age')
        return `${label} ${formatter(creditRaw[k])}`
      })
      .join(', ')
    return `Credit record #${creditDetails?.index ?? creditIndex} reflects the following profile: ${creditPreview || 'key credit-risk details available'}.`
  })()
  const buildSampleStory = (datasetName, sample) => {
    const raw = sample?.raw || {}
    if (datasetName === 'loan') {
      const applicantIncome = raw.ApplicantIncome != null ? formatUSD(raw.ApplicantIncome) : '$0'
      const coapplicantIncome = raw.CoapplicantIncome != null ? formatUSD(raw.CoapplicantIncome) : '$0'
      const householdIncomeNum = Number(raw.ApplicantIncome || 0) + Number(raw.CoapplicantIncome || 0)
      const householdIncome = householdIncomeNum > 0 ? formatUSD(householdIncomeNum) : '$0'
      const loanAmountNumeric = Number(raw.LoanAmount)
      const loanAmount = Number.isFinite(loanAmountNumeric) && loanAmountNumeric > 0
        ? formatLoanAmountUSD(loanAmountNumeric)
        : `${formatLoanAmountUSD(128)} (imputed)`
      const loanTerm = raw.Loan_Amount_Term != null ? `${formatCount(raw.Loan_Amount_Term)} months` : '360 months'
      return `Loan ID ${sample?.loan_id || loanId} has applicant income ${applicantIncome}, coapplicant income ${coapplicantIncome} (total ${householdIncome}), requested loan amount ${loanAmount}, and a ${loanTerm} term.`
    }
    if (datasetName === 'bankruptcy') {
      const company = sample?.company_name || companyName
      const yearPart = sample?.year != null ? ` in year ${sample.year}` : ''
      const preferredOrder = [
        'ROA(C) before interest and depreciation before interest',
        'ROA(A) before interest and % after tax',
        'Debt ratio %',
        'Current Ratio',
        'Interest Coverage Ratio',
        'Cash Flow/Total Assets',
        'Net worth/Assets',
        'Working Capital/Total Assets',
        'Total Asset Turnover',
        'Borrowing dependency',
        'Retained Earnings/Total Assets',
      ]
      const metricOrderIndex = new Map(preferredOrder.map((k, i) => [k, i]))
      const metricPreview = Object.entries(raw)
        .filter(([k, v]) => !['company_name', 'year', 'status_label', 'Bankrupt?'].includes(k) && typeof v === 'number' && Number.isFinite(v))
        .sort(([a], [b]) => {
          const ai = metricOrderIndex.has(a) ? metricOrderIndex.get(a) : Number.MAX_SAFE_INTEGER
          const bi = metricOrderIndex.has(b) ? metricOrderIndex.get(b) : Number.MAX_SAFE_INTEGER
          if (ai !== bi) return ai - bi
          return a.localeCompare(b)
        })
        .slice(0, 4)
        .map(([k, v]) => `${formatBankruptcyMetricLabel(k)} ${formatBankruptcyMetricValue(k, v)}`)
        .join(', ')
      return `Company ${company}${yearPart} was assessed using key bankruptcy indicators${metricPreview ? `, including ${metricPreview}` : ''}.`
    }
    const importantCreditFields = [
      ['age', (v) => `${formatCount(v)} years`],
      ['MonthlyIncome', (v) => `${formatUSD(v)} per month`],
      ['DebtRatio', (v) => formatRatioAsPercent(v)],
      ['RevolvingUtilizationOfUnsecuredLines', (v) => formatRatioAsPercent(v)],
      ['NumberOfTime30-59DaysPastDueNotWorse', (v) => `${formatCount(v)} times`],
      ['NumberOfOpenCreditLinesAndLoans', (v) => `${formatCount(v)} lines`],
    ]
    const creditPreview = importantCreditFields
      .filter(([k]) => hasValue(raw[k]))
      .slice(0, 4)
      .map(([k, formatter]) => {
        const label = k
          .replace('MonthlyIncome', 'monthly income')
          .replace('DebtRatio', 'debt ratio')
          .replace('RevolvingUtilizationOfUnsecuredLines', 'revolving utilization')
          .replace('NumberOfTime30-59DaysPastDueNotWorse', '30-59 days past due')
          .replace('NumberOfOpenCreditLinesAndLoans', 'open credit lines')
          .replace('age', 'age')
        return `${label} ${formatter(raw[k])}`
      })
      .join(', ')
    return `Credit record #${sample?.index ?? creditIndex} reflects the following profile: ${creditPreview || 'key credit-risk details available'}.`
  }
  const comparisonProfiles = (detailComparisons?.[dataset] || []).map((sample, idx) => {
    const statusLabel = getDatasetStatusLabel(dataset, sample) || 'Unknown'
    const isNegative = /not approved|bankrupt|higher/i.test(statusLabel)
    const isPositive = !isNegative && /approved|alive|lower/i.test(statusLabel)
    const statusToneClass = isNegative ? 'text-red-600' : isPositive ? 'text-emerald-600' : 'text-slate-700'
    const statusBadgeClass = isNegative ? 'border-red-200 bg-red-50' : isPositive ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200 bg-slate-50'
    const subjectLabel = dataset === 'bankruptcy' ? `Company ${String.fromCharCode(65 + idx)}` : `Applicant ${String.fromCharCode(65 + idx)}`
    return {
      key: `${dataset}-${idx}-${statusLabel}-${subjectLabel}`,
      subjectLabel,
      statusLabel,
      statusToneClass,
      statusBadgeClass,
      story: buildSampleStory(dataset, sample),
    }
  })

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
                className="w-full rounded-lg border border-slate-400 bg-white px-3 py-2 text-sm shadow-sm focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-300"
                style={{ fontFamily: 'Inter, system-ui, sans-serif', letterSpacing: '0', wordSpacing: '0' }}
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
                className="w-full rounded-lg border border-slate-400 bg-white px-3 py-2 text-sm shadow-sm focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-300"
                style={{ fontFamily: 'Inter, system-ui, sans-serif', letterSpacing: '0', wordSpacing: '0' }}
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
              disabled={!canRunAnalyze}
              className="rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:from-emerald-700 hover:to-teal-700 disabled:opacity-50"
            >
              {loading ? 'Analyzing...' : sampleLoading ? 'Loading sample...' : 'Run Analysis'}
            </button>
          </div>
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        </section>

        {comparisonProfiles.length > 0 && (
          <section className="mb-6 rounded-2xl border-2 border-slate-300 bg-white p-6 shadow-sm ring-1 ring-slate-200">
            <h2 className="font-display mb-4 text-lg font-bold text-slate-900 md:text-xl">
              {dataset === 'loan' ? 'Loan Details' : dataset === 'credit_risk' ? 'Credit Details' : 'Company Details'}
            </h2>
            <div className="overflow-x-auto rounded-xl border-2 border-slate-400">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="border-b border-r border-slate-400 px-3 py-2 text-left font-bold uppercase tracking-wide text-slate-700">Applicant</th>
                    <th className="border-b border-r border-slate-400 px-3 py-2 text-left font-bold uppercase tracking-wide text-slate-700">Details</th>
                    <th className="border-b border-slate-400 px-3 py-2 text-left font-bold uppercase tracking-wide text-slate-700">Status</th>
                  </tr>
                </thead>
                <tbody className="bg-white">
                  {comparisonProfiles.map((profile, idx) => (
                    <tr key={profile.key} className={idx % 2 === 1 ? 'bg-slate-50/40' : ''}>
                      <td className="border-b border-r border-slate-400 px-3 py-2 font-semibold text-slate-800">{profile.subjectLabel}</td>
                      <td className="border-b border-r border-slate-400 px-3 py-2 leading-relaxed text-slate-700">{profile.story}</td>
                      <td className="border-b border-slate-400 px-3 py-2">
                        <span className={`inline-flex rounded-full border px-2 py-0.5 text-xs font-bold ${profile.statusToneClass} ${profile.statusBadgeClass}`}>
                          {profile.statusLabel}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="rounded-2xl border-2 border-slate-300 bg-white p-6 shadow-sm ring-1 ring-slate-200">
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
                <div className="rounded-lg border-2 border-slate-300 bg-white p-4 shadow-sm ring-1 ring-slate-200">
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
                <div className="rounded-lg border-2 border-slate-300 bg-white p-4 shadow-sm ring-1 ring-slate-200">
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
                      <div className="mx-auto w-full max-w-xl rounded-2xl border-2 border-slate-300 bg-gradient-to-b from-white via-slate-50 to-white p-6 shadow-sm ring-1 ring-slate-200">
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
                              <span className="border-l border-slate-300 pl-3 font-semibold tabular-nums text-slate-700">
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
                              <span className="border-l border-slate-300 pl-3 font-semibold tabular-nums text-slate-700">
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
            <p className="rounded-lg border-2 border-dashed border-slate-400 bg-slate-50 p-4 text-center text-sm text-gray-500">
              Choose options above and run analysis to see the chart.
            </p>
          )}
        </section>
      </main>
    </div>
  )
}
