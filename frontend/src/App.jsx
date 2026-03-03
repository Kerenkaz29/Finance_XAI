import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { ExpertiseProvider } from './context/ExpertiseContext'
import Dashboard from './pages/Dashboard'
import Survey from './pages/Survey'
import { getReady } from './api/client'

function LoadingOverlay({ done, total, current }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white">
      <div className="flex flex-col items-center gap-6 w-80 text-center px-6">
        <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
        <h1 className="text-2xl font-bold text-slate-800">Loading Models</h1>
        <p className="text-gray-500 text-sm leading-relaxed">
          Downloading model weights from Google Drive.<br />
          This only happens once — please wait.
        </p>
        <div className="w-full">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>{current || 'Starting…'}</span>
            <span>{done}/{total}</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="text-xs text-gray-400 mt-1 text-right">{pct}%</div>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [modelsReady, setModelsReady] = useState(
    () => sessionStorage.getItem('models_ready_once') === '1',
  )
  const [showLoadingOverlay, setShowLoadingOverlay] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0, current: '' })

  useEffect(() => {
    let interval
    const check = async () => {
      const data = await getReady()
      setProgress({ done: data.done || 0, total: data.total || 0, current: data.current || '' })
      if (data.ready) {
        setModelsReady(true)
        setShowLoadingOverlay(false)
        sessionStorage.setItem('models_ready_once', '1')
        clearInterval(interval)
        return
      }
      // Show blocking overlay only for genuine first-time download.
      // Avoid flashing it on every page refresh.
      const readySeenOnce = sessionStorage.getItem('models_ready_once') === '1'
      const isDownloading = (data.total || 0) > 0 && (data.done || 0) < (data.total || 0)
      if (!readySeenOnce && isDownloading) {
        setShowLoadingOverlay(true)
      }
    }
    check()
    interval = setInterval(check, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <ExpertiseProvider>
      {!modelsReady && showLoadingOverlay && (
        <LoadingOverlay done={progress.done} total={progress.total} current={progress.current} />
      )}
      <BrowserRouter>
        <nav className="border-b border-gray-200 bg-white px-4 py-3 shadow-sm">
          <div className="mx-auto flex max-w-6xl gap-3">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `rounded-lg px-4 py-2 text-lg font-semibold transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 border border-blue-200'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 border border-transparent'
                }`
              }
            >
              Dashboard
            </NavLink>
            <NavLink
              to="/survey"
              className={({ isActive }) =>
                `rounded-lg px-4 py-2 text-lg font-semibold transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 border border-blue-200'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 border border-transparent'
                }`
              }
            >
              Survey
            </NavLink>
          </div>
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/survey" element={<Survey />} />
        </Routes>
      </BrowserRouter>
    </ExpertiseProvider>
  )
}

export default App
