import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

function ValueLabel(props) {
  const { x, y, value, payload } = props || {}
  if (x == null || y == null) return null
  const label =
    payload && typeof payload.label === 'string'
      ? payload.label
      : (Number.isFinite(value) ? Number(value).toFixed(1) : '')
  if (!label) return null
  return (
    <text
      x={x + 4}
      y={y + 4}
      fontSize={11}
      textAnchor="start"
      fill="#111827"
    >
      {label}
    </text>
  )
}

export function LIMEChart({ data, title = 'Local explanation', height = 420 }) {
  if (!data?.feature_names?.length) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6 text-gray-500">
        {data?.error || 'No LIME data available. Run analysis above.'}
      </div>
    )
  }
  const names = data.feature_names
  const rawValues = data.importance ?? data.importance_raw ?? []

  const chartData = names.map((name, i) => {
    const v = Number(rawValues[i])
    const finite = Number.isFinite(v)
    const base = finite ? v : 0
    const abs = Math.abs(base)
    const label = `${base >= 0 ? '+' : '-'}${abs.toFixed(1)}`
    return {
      name: String(name).length > 35 ? String(name).slice(0, 32) + '…' : name,
      fullName: name,
      value: abs,
      label,
    }
  }).reverse()

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      {data.error && (
        <p className="mb-3 rounded bg-amber-50 p-2 text-sm text-amber-800">
          {data.error}
        </p>
      )}
      <h3 className="mb-3 text-sm font-semibold text-gray-700">{title}</h3>
      <p className="mb-2 text-xs text-gray-500">How each factor pushed this specific decision.</p>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ left: 20, right: 70, top: 5, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5edf5" />
            <XAxis
              type="number"
              tick={{ fontSize: 11 }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={210}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(_value, _name, ctx) =>
                ctx?.payload?.label ? [ctx.payload.label, 'Contribution'] : ['', 'Contribution']
              }
              contentStyle={{ fontSize: 12 }}
              labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName ?? ''}
            />
            <Bar
              dataKey="value"
              maxBarSize={26}
              radius={3}
              fill="#1d4ed8"
              label={<ValueLabel />}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
