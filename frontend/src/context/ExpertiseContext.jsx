import { createContext, useContext, useState, useCallback } from 'react'

const ExpertiseContext = createContext({
  mode: 'non_expert',
  setMode: () => {},
})

export function ExpertiseProvider({ children }) {
  const [mode, setModeState] = useState(() => {
    try {
      return localStorage.getItem('xai_expertise_mode') || 'non_expert'
    } catch {
      return 'non_expert'
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
