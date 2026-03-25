const API_BASE = import.meta.env.VITE_API_URL || '/api'

async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController()
  const useTimeout = Number.isFinite(timeoutMs) && timeoutMs > 0
  const timer = useTimeout ? setTimeout(() => controller.abort(), timeoutMs) : null
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (err) {
    if (useTimeout && err?.name === 'AbortError') {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`)
    }
    throw err
  } finally {
    if (timer) clearTimeout(timer)
  }
}

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
  const res = await fetchWithTimeout(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, features, model_type }),
  }, null)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getXAI({ dataset = 'loan', features, expertise, method, model_type = 'rf' }) {
  const timeoutMs = null
  const startedAt = Date.now()
  if (method === 'DiCE') {
    console.info('[DiCE] Request start', {
      dataset,
      expertise,
      model_type,
      feature_count: Array.isArray(features) ? features.length : null,
      timeout_ms: timeoutMs,
    })
  }
  const res = await fetchWithTimeout(`${API_BASE}/xai`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, features, expertise, method, model_type }),
  }, timeoutMs)
  if (method === 'DiCE') {
    console.info('[DiCE] Response received', {
      status: res.status,
      elapsed_ms: Date.now() - startedAt,
    })
  }
  if (!res.ok) throw new Error(await res.text())
  const payload = await res.json()
  if (method === 'DiCE') {
    const cfs = Array.isArray(payload?.counterfactuals) ? payload.counterfactuals.length : null
    if (payload?.error) {
      console.error('[DiCE] Response error payload', {
        elapsed_ms: Date.now() - startedAt,
        error: payload.error,
        payload,
      })
    } else {
      console.info('[DiCE] Response success payload', {
        elapsed_ms: Date.now() - startedAt,
        counterfactual_count: cfs,
      })
    }
  }
  return payload
}
