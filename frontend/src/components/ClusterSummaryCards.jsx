import React from 'react'

export default function ClusterSummaryCards({ clusters }) {
  if (!clusters?.length) return <p className="text-sm text-gray-500">No clusters yet. Upload and train a model.</p>
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {clusters.map(c => (
        <div key={c.cluster} className="rounded-xl border bg-white dark:bg-gray-800 p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Cluster {c.cluster}</h3>
            <span className="text-xs px-2 py-0.5 rounded bg-blue-600/10 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">{c.count} customers</span>
          </div>
          <dl className="mt-3 grid grid-cols-3 gap-2 text-sm">
            <div>
              <dt className="text-gray-500">Avg Income</dt>
              <dd className="font-medium">${c.avg_income}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Avg Spending</dt>
              <dd className="font-medium">{c.avg_spending}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Avg Age</dt>
              <dd className="font-medium">{c.avg_age}</dd>
            </div>
          </dl>
        </div>
      ))}
    </div>
  )
}
