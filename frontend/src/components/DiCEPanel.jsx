import React from 'react'

export function DiCEPanel({ data }) {
  if (!data) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6 text-gray-500">
        Choose DiCE and run analysis to see what-if scenarios.
      </div>
    )
  }
  if (data.error) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-800">
        {data.error}
      </div>
    )
  }
  const cfs = data.counterfactuals || []
  const useDisplay = data.expertise === 'non_expert' && cfs[0]?.changes_display
  const currentProb = Number.isFinite(data.current_probability) ? data.current_probability : null
  const bestProb = Number.isFinite(data.best_counterfactual_probability) ? data.best_counterfactual_probability : null
  const potentialGain = Number.isFinite(data.potential_gain) ? data.potential_gain : null

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <h3 className="mb-1 text-sm font-bold tracking-wide text-slate-800">COUNTERFACTUAL EXPLANATIONS</h3>
      <p className="mb-4 text-xs text-gray-500">==============================</p>
      {(currentProb != null || bestProb != null) && (
        <div className="mb-4 space-y-2 text-sm text-slate-700">
          {currentProb != null && (
            <p>Current target probability: {(currentProb * 100).toFixed(1)}%</p>
          )}
          {bestProb != null && (
            <p>Best counterfactual target probability: {(bestProb * 100).toFixed(1)}%</p>
          )}
          {potentialGain != null && (
            <p className="font-medium text-emerald-700">Potential gain: {(potentialGain >= 0 ? '+' : '') + (potentialGain * 100).toFixed(1)}%</p>
          )}
        </div>
      )}
      <p className="mb-4 text-xs text-gray-500">{data.description}</p>
      {cfs.length === 0 ? (
        <p className="text-gray-500">No counterfactuals generated for this instance.</p>
      ) : (
        <ul className="space-y-4">
          {cfs.map((cf, i) => (
            <li key={i} className="rounded border border-gray-100 bg-gray-50 p-3">
              <span className="text-xs font-semibold text-slate-700">
                Scenario {i + 1}
                {Number.isFinite(cf.target_probability) ? `: ${(cf.target_probability * 100).toFixed(1)}%` : ''}
              </span>
              <ul className="mt-2 space-y-1 text-sm">
                {(useDisplay ? Object.entries(cf.changes_display || {}) : Object.entries(cf.changes || {})).map(([name, v]) => (
                  <li key={name}>
                    <strong>{name}</strong>: {v.from} → {v.to}
                  </li>
                ))}
              </ul>
              {cf.ai_explanation && (
                <p className="mt-3 rounded border border-blue-100 bg-blue-50 p-2 text-sm text-slate-700">
                  {cf.ai_explanation}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
