import React from 'react'
import { NavLink } from 'react-router-dom'

export default function App({ children }) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950">
      <header className="border-b bg-white/70 dark:bg-gray-800/70 backdrop-blur sticky top-0 z-10">
        <div className="container py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded bg-blue-600"></div>
            <h1 className="text-xl font-semibold">Customer Segmentation</h1>
          </div>
          <nav className="flex gap-1 text-sm">
            <NavLink to="/" className={({isActive}) => `px-3 py-1.5 rounded ${isActive ? 'bg-blue-600 text-white' : 'hover:bg-blue-50 dark:hover:bg-gray-700'}`}>Dashboard</NavLink>
            <NavLink to="/upload" className={({isActive}) => `px-3 py-1.5 rounded ${isActive ? 'bg-blue-600 text-white' : 'hover:bg-blue-50 dark:hover:bg-gray-700'}`}>Upload</NavLink>
            <NavLink to="/retrain" className={({isActive}) => `px-3 py-1.5 rounded ${isActive ? 'bg-blue-600 text-white' : 'hover:bg-blue-50 dark:hover:bg-gray-700'}`}>Retrain</NavLink>
          </nav>
        </div>
      </header>
      <main className="container py-6">
        {children}
      </main>
      <footer className="border-t py-6 text-center text-xs text-gray-500">AI-Based Customer Segmentation Dashboard</footer>
    </div>
  )
}
