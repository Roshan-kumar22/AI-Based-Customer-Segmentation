# AI-Based Customer Segmentation Dashboard (React + Flask)

An end-to-end app that clusters customers using K-Means, projects to 2D with PCA, and visualizes results on a modern React dashboard.

## Features
- Upload CSV dataset (CustomerID, Gender, Age, Annual Income, Spending Score)
- Train K-Means (k = 2–10), StandardScaler normalization, PCA (2D)
- Interactive Plotly scatter (PCA1 vs PCA2), hover details
- Cluster summary cards (count, avg income, avg spending, avg age)
- Paginated data table
- Predict endpoint for new customers
- SQLite persistence and joblib model artifacts
- Large-file upload support with progress bar and configurable server limit
- Optional one-click Kaggle sample dataset fetch

## Project Structure
```
customer-segmentation/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── database.db                # created at runtime
│   ├── Mall_Customers.csv         # sample (or downloaded via /sample-dataset)
│   └── models/
│       ├── kmeans_model.joblib    # created after /train
│       ├── scaler.joblib
│       └── pca.joblib
└── frontend/
    ├── index.html
    ├── package.json
    ├── tailwind.config.js
    └── src/
        ├── api/client.js
        ├── components/
        ├── pages/
        ├── App.jsx
        └── main.jsx
```

## Prerequisites
- Python 3.10+
- Node.js 18+

## Backend Setup (Flask)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional: increase max upload size (MB). Default=1024 MB.
export MAX_CONTENT_LENGTH_MB=2048
python app.py
```
Backend runs at http://localhost:5000

### Endpoints
- POST `/upload` (multipart/form-data, field: file) → registers dataset in SQLite
- POST `/train` JSON: `{ "n_clusters": 5 }` → trains K-Means, saves model and PCA, stores results
- GET `/clusters` → `[{ cluster, count, avg_income, avg_spending, avg_age }]`
- GET `/plot-data` → `[ { CustomerID, Age, AnnualIncome, SpendingScore, cluster, pca1, pca2 } ]`
- POST `/predict` JSON: `{ Gender, Age, AnnualIncome, SpendingScore }` → `{ cluster }`
- GET `/datasets` → list uploaded datasets
- GET `/sample-dataset` → downloads Kaggle sample and registers it (see below)

### Large File Uploads
- Server limit is controlled by `MAX_CONTENT_LENGTH_MB` (default 1024 MB).
- Reverse proxies (e.g., Nginx) may also need limits adjusted (e.g., `client_max_body_size`).
- The app validates headers and counts rows in chunks to reduce memory usage during `/upload`.

### Kaggle Sample Dataset (Optional)
This uses `kagglehub` to fetch `shwetabh123/mall-customers` and normalize headers.
```bash
# With server running
curl http://localhost:5000/sample-dataset
# Response includes dataset_id and row_count
```
If `kagglehub` requires authentication, follow its instructions before calling the endpoint.

## Frontend Setup (React + Vite + Tailwind)
```bash
cd frontend
cp .env.example .env.local  # ensure VITE_API_URL matches your backend URL
npm install
npm run dev
```
Frontend runs at http://localhost:5173

### Frontend Notes
- Upload page shows a progress bar for large files and clearer error messages (e.g., 413 when too large).
- Dashboard: toolbar + responsive layout (plot + summaries + table).
- You can retrain from the Retrain page and then Refresh on the Dashboard.

## Typical Workflow
1) Upload your CSV on the Upload page, or call `GET /sample-dataset` to register a sample.
2) Open the Retrain page → choose `k` (2–10) → Retrain.
3) Go to Dashboard → Refresh Data → view scatter and summaries.
4) Use `/predict` to score a single customer via POST JSON.

## CSV Schema
Required columns: `CustomerID`, `Gender`, `Age`, `Annual Income`, `Spending Score`




## License
MIT

