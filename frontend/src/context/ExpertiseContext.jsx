/**
 * Global expert / non-expert mode (persisted in localStorage).
 * Wired for future UI toggle; Dashboard currently sends "expert" directly.
 */
import { createContext, useContext, useState, useCallback } from 'react'

const ExpertiseContext = createContext({
  mode: 'expert',
  setMode: () => {},
})

export function ExpertiseProvider({ children }) {
  const [mode, setModeState] = useState(() => {
    // Restore last selected mode across page reloads.
    try {
      return localStorage.getItem('xai_expertise_mode') || 'expert'
    } catch {
      return 'expert'
    }
  })
  const setMode = useCallback((value) => {
    setModeState(value)
    try {
      localStorage.setItem('xai_expertise_mode', value)
    } catch {}
  }, [])
  return (
    <ExpertiseContext.Provider value={{ mode, setMode }}>
      {children}
    </ExpertiseContext.Provider>
  )
}

export function useExpertise() {
  const ctx = useContext(ExpertiseContext)
  if (!ctx) throw new Error('useExpertise must be used within ExpertiseProvider')
  return ctx
}
