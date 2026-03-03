import React, { useState } from 'react'
import { useExpertise } from '../context/ExpertiseContext'

// S-TIAS (Trust in Automation Scale) — short form; 1–7 Likert
const STIAS_ITEMS = [
  { id: 'stias_1', text: 'The AI system is reliable.' },
  { id: 'stias_2', text: 'I trust the AI system to make good decisions.' },
  { id: 'stias_3', text: 'The AI system behaves in an unpredictable manner.' },
  { id: 'stias_4', text: 'I am confident in the AI system\'s recommendations.' },
  { id: 'stias_5', text: 'The AI system makes errors.' },
]

// SCS (e.g. subjective clarity / satisfaction) — short form; 1–7
const SCS_ITEMS = [
  { id: 'scs_1', text: 'The explanation was clear and easy to understand.' },
  { id: 'scs_2', text: 'I felt I understood why the system made this decision.' },
  { id: 'scs_3', text: 'The level of detail was appropriate for me.' },
  { id: 'scs_4', text: 'I would be comfortable explaining this decision to someone else.' },
  { id: 'scs_5', text: 'The explanation matched my expectations.' },
]

const LABELS = ['Strongly disagree', 'Disagree', 'Somewhat disagree', 'Neutral', 'Somewhat agree', 'Agree', 'Strongly agree']

export default function Survey() {
  const { mode } = useExpertise()
  const [stias, setStias] = useState({})
  const [scs, setScs] = useState({})
  const [submitted, setSubmitted] = useState(false)

  const handleStias = (id, value) => setStias((s) => ({ ...s, [id]: Number(value) }))
  const handleScs = (id, value) => setScs((s) => ({ ...s, [id]: Number(value) }))

  const handleSubmit = (e) => {
    e.preventDefault()
    const payload = {
      expertise: mode,
      stias: stias,
      scs: scs,
      stias_mean: mean(Object.values(stias)),
      scs_mean: mean(Object.values(scs)),
    }
    console.log('Survey submitted:', payload)
    setSubmitted(true)
  }

  const allStias = STIAS_ITEMS.every((i) => stias[i.id] != null)
  const allScs = SCS_ITEMS.every((i) => scs[i.id] != null)
  const canSubmit = allStias && allScs

  return (
    <div className="min-h-screen bg-surface-50">
      <header className="border-b border-gray-200 bg-white shadow-sm">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <h1 className="text-xl font-bold text-gray-900">Feedback survey</h1>
          <p className="mt-1 text-sm text-gray-600">
            Your responses help us compare how experts and non-experts accept AI explanations (S-TIAS and SCS scales).
          </p>
          <p className="mt-1 text-xs text-gray-500">Current mode: {mode === 'expert' ? 'Expert' : 'Non-Expert'}</p>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-8">
        {submitted ? (
          <div className="rounded-xl border border-green-200 bg-green-50 p-6 text-center text-green-800">
            <p className="font-medium">Thank you. Your responses have been recorded.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-10">
            <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="mb-1 text-lg font-semibold text-gray-900">S-TIAS (Trust in Automation)</h2>
              <p className="mb-6 text-sm text-gray-500">Please rate your agreement with each statement (1 = Strongly disagree, 7 = Strongly agree).</p>
              <ul className="space-y-6">
                {STIAS_ITEMS.map((item) => (
                  <li key={item.id}>
                    <p className="mb-2 text-sm font-medium text-gray-700">{item.text}</p>
                    <div className="flex flex-wrap gap-2">
                      {[1, 2, 3, 4, 5, 6, 7].map((v) => (
                        <label key={v} className="flex cursor-pointer items-center gap-1">
                          <input
                            type="radio"
                            name={item.id}
                            value={v}
                            checked={stias[item.id] === v}
                            onChange={() => handleStias(item.id, v)}
                            className="h-4 w-4 border-gray-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="text-sm text-gray-600">{v}</span>
                        </label>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            </section>

            <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="mb-1 text-lg font-semibold text-gray-900">SCS (Subjective Clarity / Satisfaction)</h2>
              <p className="mb-6 text-sm text-gray-500">Please rate your agreement (1 = Strongly disagree, 7 = Strongly agree).</p>
              <ul className="space-y-6">
                {SCS_ITEMS.map((item) => (
                  <li key={item.id}>
                    <p className="mb-2 text-sm font-medium text-gray-700">{item.text}</p>
                    <div className="flex flex-wrap gap-2">
                      {[1, 2, 3, 4, 5, 6, 7].map((v) => (
                        <label key={v} className="flex cursor-pointer items-center gap-1">
                          <input
                            type="radio"
                            name={item.id}
                            value={v}
                            checked={scs[item.id] === v}
                            onChange={() => handleScs(item.id, v)}
                            className="h-4 w-4 border-gray-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="text-sm text-gray-600">{v}</span>
                        </label>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            </section>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={!canSubmit}
                className="rounded-lg bg-primary-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
              >
                Submit feedback
              </button>
            </div>
          </form>
        )}
      </main>
    </div>
  )
}

function mean(arr) {
  const n = arr.filter((v) => typeof v === 'number' && !Number.isNaN(v)).length
  if (n === 0) return null
  return arr.reduce((a, b) => a + (typeof b === 'number' && !Number.isNaN(b) ? b : 0), 0) / n
}
