import React from 'react'
import Plot from 'react-plotly.js'

export default function PlotScatter({ points }) {
  const byCluster = points.reduce((acc, p) => {
    const key = p.cluster
    if (!acc[key]) acc[key] = { x: [], y: [], text: [], customdata: [] }
    acc[key].x.push(p.pca1)
    acc[key].y.push(p.pca2)
    acc[key].text.push(`ID: ${p.CustomerID}`)
    acc[key].customdata.push(p)
    return acc
  }, {})

  const traces = Object.entries(byCluster).map(([cluster, v]) => ({
    x: v.x,
    y: v.y,
    mode: 'markers',
    type: 'scattergl',
    name: `Cluster ${cluster}`,
    text: v.text,
    hovertemplate: 'PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<br>%{text}<br>Age: %{customdata.Age}<br>Income: %{customdata.AnnualIncome}<br>Spending: %{customdata.SpendingScore}<extra></extra>',
    marker: { size: 8, opacity: 0.85 },
    customdata: v.customdata,
  }))

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 520,
        margin: { l: 40, r: 20, t: 40, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'PCA1', zeroline: false },
        yaxis: { title: 'PCA2', zeroline: false },
        legend: { orientation: 'h' }
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true }}
    />
  )
}
