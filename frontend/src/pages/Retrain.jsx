import React, { useState } from 'react'
import { trainModel } from '../api/client'

export default function Retrain() {
  const [k, setK] = useState(5)
  const [status, setStatus] = useState('')

  const onSubmit = async (e) => {
    e.preventDefault()
    setStatus('Training...')
    try {
      const resp = await trainModel(Number(k))
      setStatus(`Trained ${resp.n_clusters} clusters on ${resp.num_records} records`)
    } catch (e) {
      setStatus(`Error: ${e?.response?.data?.error || e.message}`)
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Retrain Model</h2>
      <form onSubmit={onSubmit} className="flex items-end gap-3">
        <label className="text-sm">
          <span className="block mb-1">Number of clusters (2–10)</span>
          <input type="number" min={2} max={10} value={k} onChange={e => setK(e.target.value)} className="px-2 py-1 border rounded w-28" />
        </label>
        <button type="submit" className="px-3 py-1.5 border rounded text-sm">Retrain</button>
      </form>
      {status && <div className="text-sm">{status}</div>}
      <p className="text-xs text-gray-500">After training, visit the Dashboard to see updated clusters and plot.</p>
    </div>
  )
}
