import React from 'react'
import { useExpertise } from '../context/ExpertiseContext'

export function ExpertiseToggle() {
  const { mode, setMode } = useExpertise()
  const isExpert = mode === 'expert'

  return (
    <div className="mt-6 flex border-b border-gray-200 bg-white text-sm font-medium">
      <button
        type="button"
        onClick={() => setMode('expert')}
        className={`flex-1 px-4 py-2.5 text-center transition-colors ${
          isExpert
            ? 'border-b-2 border-emerald-500 text-slate-900'
            : 'text-gray-500 hover:bg-gray-50'
        }`}
      >
        Expert Mode
      </button>
      <button
        type="button"
        onClick={() => setMode('non_expert')}
        className={`flex-1 px-4 py-2.5 text-center transition-colors ${
          !isExpert
            ? 'border-b-2 border-emerald-500 text-slate-900'
            : 'text-gray-500 hover:bg-gray-50'
        }`}
      >
        Non-Expert Mode
      </button>
    </div>
  )
}
