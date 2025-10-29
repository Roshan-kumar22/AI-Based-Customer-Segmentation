import React, { useEffect, useState } from 'react'
import { fetchClusters, fetchPlotData } from '../api/client'
import ClusterSummaryCards from '../components/ClusterSummaryCards'
import PlotScatter from '../components/PlotScatter'
import DataTable from '../components/DataTable'

export default function Dashboard() {
  const [clusters, setClusters] = useState([])
  const [points, setPoints] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    try {
      setLoading(true)
      const [c, p] = await Promise.all([fetchClusters(), fetchPlotData()])
      setClusters(c)
      setPoints(p)
    } catch (e) {
      setError(e?.message || 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="rounded-lg border bg-white dark:bg-gray-800 p-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Dashboard</h2>
          <p className="text-xs text-gray-500">Visualize clusters, review summaries, and inspect customers.</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={load} className="px-3 py-1.5 rounded bg-blue-600 text-white text-sm hover:bg-blue-700">Refresh Data</button>
        </div>
      </div>

      {error && <div className="text-sm text-red-600">{error}</div>}

      {/* Grid layout: plot + summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-lg border bg-white dark:bg-gray-800 p-2">
          <PlotScatter points={points} />
        </div>
        <div className="lg:col-span-1">
          <ClusterSummaryCards clusters={clusters} />
        </div>
      </div>

      {/* Data table */}
      <div className="rounded-lg border bg-white dark:bg-gray-800 p-2">
        <h3 className="font-medium mb-2">Customers</h3>
        <DataTable points={points} />
      </div>
    </div>
  )
}
