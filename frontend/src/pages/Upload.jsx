import React, { useState } from 'react'
import { uploadDataset, listDatasets } from '../api/client'

export default function Upload() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('')
  const [datasets, setDatasets] = useState([])
  const [progress, setProgress] = useState(0)

  const onSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    setStatus('Uploading...')
    setProgress(0)
    try {
      const resp = await uploadDataset(file, (p) => setProgress(p))
      setStatus(`Uploaded: ${resp.filename} (${resp.row_count} rows)`) 
      const ds = await listDatasets()
      setDatasets(ds)
    } catch (e) {
      const errMsg = e?.response?.data?.error || e.message
      if (e?.response?.status === 413) {
        setStatus(`File too large. Backend limit: ${e?.response?.data?.max_mb || 'unknown'} MB`)
      } else {
        setStatus(`Error: ${errMsg}`)
      }
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Upload Dataset</h2>
      <form onSubmit={onSubmit} className="space-y-3">
        <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)} className="block" />
        <button type="submit" className="px-3 py-1.5 border rounded text-sm" disabled={!file}>Upload</button>
      </form>
      {status && <div className="text-sm">{status}</div>}
      {progress > 0 && progress < 100 && (
        <div className="w-full max-w-md">
          <div className="h-2 bg-gray-200 rounded">
            <div className="h-2 bg-blue-600 rounded" style={{ width: `${progress}%` }} />
          </div>
          <div className="text-xs text-gray-500 mt-1">{progress}%</div>
        </div>
      )}
      {datasets.length > 0 && (
        <div>
          <h3 className="font-medium mb-2">Uploaded Datasets</h3>
          <ul className="list-disc pl-5 text-sm">
            {datasets.map(d => (
              <li key={d.id}>{d.filename} — {d.row_count} rows — {new Date(d.uploaded_at).toLocaleString()}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
