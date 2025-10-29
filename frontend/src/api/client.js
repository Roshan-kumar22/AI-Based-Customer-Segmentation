import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000',
  timeout: 600000, // 10 minutes for big uploads
})

export const uploadDataset = async (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: evt => {
      if (onProgress && evt.total) {
        const percent = Math.round((evt.loaded * 100) / evt.total)
        onProgress(percent)
      }
    },
  })
  return data
}

export const trainModel = async (n_clusters = 5) => {
  const { data } = await api.post('/train', { n_clusters })
  return data
}

export const fetchClusters = async () => {
  const { data } = await api.get('/clusters')
  return data.clusters
}

export const fetchPlotData = async () => {
  const { data } = await api.get('/plot-data')
  return data.points
}

export const listDatasets = async () => {
  const { data } = await api.get('/datasets')
  return data.datasets
}

export const predictCluster = async (payload) => {
  const { data } = await api.post('/predict', payload)
  return data
}

export default api
