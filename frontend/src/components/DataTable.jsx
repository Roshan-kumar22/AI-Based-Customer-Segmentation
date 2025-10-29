import React, { useMemo, useState } from 'react'

export default function DataTable({ points, pageSize = 10 }) {
  const [page, setPage] = useState(1)
  const totalPages = Math.max(1, Math.ceil((points?.length || 0) / pageSize))

  const rows = useMemo(() => {
    const start = (page - 1) * pageSize
    return (points || []).slice(start, start + pageSize)
  }, [points, page, pageSize])

  if (!points?.length) return null

  return (
    <div className="mt-2">
      <div className="overflow-x-auto rounded-lg border bg-white dark:bg-gray-800">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-700/50">
            <tr>
              <th className="px-3 py-2 text-left font-medium text-gray-600">CustomerID</th>
              <th className="px-3 py-2 text-left font-medium text-gray-600">Age</th>
              <th className="px-3 py-2 text-left font-medium text-gray-600">Income</th>
              <th className="px-3 py-2 text-left font-medium text-gray-600">Spending</th>
              <th className="px-3 py-2 text-left font-medium text-gray-600">Cluster</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              <tr key={idx} className="border-t border-gray-100 dark:border-gray-700 hover:bg-gray-50/60 dark:hover:bg-gray-700/30">
                <td className="px-3 py-2">{r.CustomerID}</td>
                <td className="px-3 py-2">{r.Age}</td>
                <td className="px-3 py-2">{r.AnnualIncome}</td>
                <td className="px-3 py-2">{r.SpendingScore}</td>
                <td className="px-3 py-2">{r.cluster}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between mt-2 text-sm">
        <span className="text-gray-600">Page {page} of {totalPages}</span>
        <div className="flex gap-2">
          <button
            className="px-2 py-1 border rounded disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-gray-700"
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
          >Prev</button>
          <button
            className="px-2 py-1 border rounded disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-gray-700"
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
          >Next</button>
        </div>
      </div>
    </div>
  )
}
