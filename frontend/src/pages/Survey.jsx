import React from 'react'

export default function Survey() {
  return (
    <div className="min-h-screen bg-surface-50 p-4">
      <div className="mx-auto max-w-4xl rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <iframe
          src="https://docs.google.com/forms/d/e/1FAIpQLSdVkr-RKqMAAn9JnIaFI5izUvUfUTPxBzjse9lAsPuYUc8AMw/viewform?embedded=true"
          width="100%"
          height="1226"
          frameBorder="0"
          marginHeight="0"
          marginWidth="0"
          title="Survey Form"
        >
          Loading...
        </iframe>
      </div>
    </div>
  )
}
