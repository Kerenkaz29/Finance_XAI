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

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <h3 className="mb-2 text-sm font-semibold text-gray-700">What-if scenarios</h3>
      <p className="mb-4 text-xs text-gray-500">{data.description}</p>
      {cfs.length === 0 ? (
        <p className="text-gray-500">No counterfactuals generated for this instance.</p>
      ) : (
        <ul className="space-y-4">
          {cfs.map((cf, i) => (
            <li key={i} className="rounded border border-gray-100 bg-gray-50 p-3">
              <span className="text-xs font-medium text-gray-500">Scenario {i + 1}</span>
              <ul className="mt-2 space-y-1 text-sm">
                {(useDisplay ? Object.entries(cf.changes_display || {}) : Object.entries(cf.changes || {})).map(([name, v]) => (
                  <li key={name}>
                    <strong>{name}</strong>: {v.from} → {v.to}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
