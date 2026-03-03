const API_BASE = import.meta.env.VITE_API_URL || '/api'

export async function health() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}

export async function getReady() {
  try {
    const res = await fetch(`${API_BASE}/ready`)
    if (!res.ok) return { ready: false, done: 0, total: 0, current: '' }
    return await res.json()
  } catch {
    return { ready: false, done: 0, total: 0, current: '' }
  }
}

export async function getDatasets() {
  const res = await fetch(`${API_BASE}/datasets`)
  return res.json()
}

export async function getLoanSamples(limit = 50) {
  const res = await fetch(`${API_BASE}/loan/samples?limit=${limit}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getBankruptcySamples(limit = 50) {
  const res = await fetch(`${API_BASE}/bankruptcy/samples?limit=${limit}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getCreditSamples(limit = 50) {
  const res = await fetch(`${API_BASE}/credit/samples?limit=${limit}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getLoanSample(loanId) {
  const res = await fetch(`${API_BASE}/loan/sample/${encodeURIComponent(loanId)}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getBankruptcySample(companyName) {
  const res = await fetch(`${API_BASE}/bankruptcy/sample/${encodeURIComponent(companyName)}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getCreditSample(index) {
  const res = await fetch(`${API_BASE}/credit/sample/${index}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function predict({ dataset = 'loan', features, model_type = 'rf' }) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, features, model_type }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getXAI({ dataset = 'loan', features, expertise, method, model_type = 'rf' }) {
  const res = await fetch(`${API_BASE}/xai`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, features, expertise, method, model_type }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
