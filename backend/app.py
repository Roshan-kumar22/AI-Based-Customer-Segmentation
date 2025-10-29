import os
import glob
import shutil
import sqlite3
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
try:
    import kagglehub  # optional, used by /sample-dataset
except Exception:
    kagglehub = None
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
DATA_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
PCA_PATH = os.path.join(MODELS_DIR, 'pca.joblib')

REQUIRED_COLUMNS = [
    'CustomerID', 'Gender', 'Age', 'Annual Income', 'Spending Score'
]

app = Flask(__name__)
# Relaxed CORS with explicit methods/headers for large multipart uploads
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.config['UPLOAD_FOLDER'] = DATA_DIR
# Configurable max upload size (default 1024 MB); set MAX_CONTENT_LENGTH_MB env var to override
_max_mb = int(os.environ.get('MAX_CONTENT_LENGTH_MB', '1024'))
app.config['MAX_CONTENT_LENGTH'] = _max_mb * 1024 * 1024


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            customer_id TEXT,
            gender TEXT,
            age REAL,
            annual_income REAL,
            spending_score REAL,
            cluster INTEGER,
            pca1 REAL,
            pca2 REAL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
        """
    )
    conn.commit()
    conn.close()


init_db()


def load_latest_dataset_path(conn: sqlite3.Connection) -> Tuple[int, str]:
    cur = conn.cursor()
    cur.execute("SELECT id, filename FROM datasets ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        raise ValueError("No dataset uploaded yet.")
    dataset_id, filename = row[0], row[1]
    return dataset_id, os.path.join(DATA_DIR, filename)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply known column renames from common Kaggle variants to our expected schema."""
    rename_map = {
        'Annual Income (k$)': 'Annual Income',
        'Spending Score (1-100)': 'Spending Score',
        'CustomerID': 'CustomerID',
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df


def validate_and_read_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def validate_headers_and_count(file_path: str) -> int:
    """Validate required headers without loading entire file and return row count using chunked reading."""
    # Read only headers
    header_df = pd.read_csv(file_path, nrows=0)
    header_df = _normalize_columns(header_df)
    missing = [c for c in REQUIRED_COLUMNS if c not in header_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Count rows efficiently with chunking
    row_count = 0
    for chunk in pd.read_csv(file_path, usecols=[c for c in header_df.columns if c in REQUIRED_COLUMNS], chunksize=100_000):
        row_count += len(chunk)
    return row_count


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    df_clean = df.dropna(subset=['Gender', 'Age', 'Annual Income', 'Spending Score']).copy()

    gender_map = {"Male": 0, "Female": 1}
    df_clean['GenderEncoded'] = df_clean['Gender'].map(gender_map)
    if df_clean['GenderEncoded'].isna().any():
        # Encode any other categories as 2
        df_clean['GenderEncoded'] = df_clean['Gender'].apply(lambda g: 0 if str(g).lower()=="male" else (1 if str(g).lower()=="female" else 2))

    features = df_clean[['GenderEncoded', 'Age', 'Annual Income', 'Spending Score']].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    return X_scaled, {
        'scaler': scaler,
        'clean_df': df_clean
    }


def train_model(df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
    X_scaled, meta = preprocess(df)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_scaled)

    # persist artifacts
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(meta['scaler'], SCALER_PATH)
    joblib.dump(pca, PCA_PATH)

    return {
        'model': kmeans,
        'pca_data': pca_2d,
        'clusters': clusters,
        'clean_df': meta['clean_df']
    }


def ensure_model_loaded():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH)):
        raise ValueError("Model not trained. Please call /train first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    return model, scaler, pca


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    save_path = os.path.join(DATA_DIR, file.filename)
    file.save(save_path)

    try:
        row_count = validate_headers_and_count(save_path)
    except Exception as e:
        return jsonify({'error': f'Invalid CSV: {str(e)}'}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (filename, row_count, uploaded_at) VALUES (?, ?, ?)",
        (file.filename, row_count, datetime.utcnow().isoformat())
    )
    conn.commit()
    dataset_id = cur.lastrowid
    conn.close()

    return jsonify({'message': 'File uploaded', 'dataset_id': dataset_id, 'filename': file.filename, 'row_count': row_count}), 200


@app.route('/train', methods=['POST'])
def train():
    payload = request.get_json(silent=True) or {}
    n_clusters = int(payload.get('n_clusters', 5))
    if n_clusters < 2 or n_clusters > 10:
        return jsonify({'error': 'n_clusters must be between 2 and 10'}), 400

    conn = get_db_connection()
    try:
        dataset_id, path = load_latest_dataset_path(conn)
        df = validate_and_read_csv(path)
        result = train_model(df, n_clusters=n_clusters)

        # Clear old customers for latest dataset to avoid duplicates
        cur = conn.cursor()
        cur.execute("DELETE FROM customers WHERE dataset_id = ?", (dataset_id,))

        clean_df = result['clean_df']
        clusters = result['clusters']
        pca_2d = result['pca_data']

        rows = []
        for i in range(len(clean_df)):
            rows.append((
                dataset_id,
                str(clean_df.iloc[i]['CustomerID']),
                str(clean_df.iloc[i]['Gender']),
                float(clean_df.iloc[i]['Age']),
                float(clean_df.iloc[i]['Annual Income']),
                float(clean_df.iloc[i]['Spending Score']),
                int(clusters[i]),
                float(pca_2d[i, 0]),
                float(pca_2d[i, 1])
            ))

        cur.executemany(
            """
            INSERT INTO customers (
                dataset_id, customer_id, gender, age, annual_income, spending_score, cluster, pca1, pca2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({'message': 'Model trained', 'n_clusters': n_clusters, 'num_records': len(rows)}), 200


@app.route('/clusters', methods=['GET'])
def clusters_summary():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM datasets")
    has_dataset = cur.fetchone()[0] > 0
    if not has_dataset:
        conn.close()
        return jsonify({'clusters': []})

    cur.execute("""
        SELECT cluster,
               COUNT(*) as count,
               AVG(annual_income) as avg_income,
               AVG(spending_score) as avg_spending,
               AVG(age) as avg_age
        FROM customers
        GROUP BY cluster
        ORDER BY cluster
    """)
    rows = cur.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            'cluster': r[0],
            'count': r[1],
            'avg_income': round(float(r[2] or 0), 2),
            'avg_spending': round(float(r[3] or 0), 2),
            'avg_age': round(float(r[4] or 0), 2)
        })
    return jsonify({'clusters': data})


@app.route('/plot-data', methods=['GET'])
def plot_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT dataset_id FROM customers ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({'points': []})
    latest_dataset_id = row[0]

    cur.execute(
        """
        SELECT customer_id, age, annual_income, spending_score, cluster, pca1, pca2
        FROM customers WHERE dataset_id = ?
        """,
        (latest_dataset_id,)
    )
    rows = cur.fetchall()
    conn.close()

    points = []
    for r in rows:
        points.append({
            'CustomerID': r[0],
            'Age': r[1],
            'AnnualIncome': r[2],
            'SpendingScore': r[3],
            'cluster': r[4],
            'pca1': r[5],
            'pca2': r[6]
        })
    return jsonify({'points': points})


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            payload = request.get_json(force=True)
        else:
            payload = request.args.to_dict()

        gender = payload.get('Gender') or payload.get('gender')
        age = float(payload.get('Age') or payload.get('age'))
        income = float(payload.get('AnnualIncome') or payload.get('annual_income') or payload.get('Annual Income'))
        spending = float(payload.get('SpendingScore') or payload.get('spending_score') or payload.get('Spending Score'))

        model, scaler, _ = ensure_model_loaded()

        gender_val = 0 if str(gender).lower()=="male" else (1 if str(gender).lower()=="female" else 2)
        X = np.array([[gender_val, age, income, spending]])
        Xs = scaler.transform(X)
        pred = int(model.predict(Xs)[0])
        return jsonify({'cluster': pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/datasets', methods=['GET'])
def list_datasets():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, row_count, uploaded_at FROM datasets ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            'id': r[0],
            'filename': r[1],
            'row_count': r[2],
            'uploaded_at': r[3],
        })
    return jsonify({'datasets': data})


@app.route('/sample-dataset', methods=['GET'])
def sample_dataset():
    if kagglehub is None:
        return jsonify({'error': 'kagglehub not installed. pip install kagglehub'}), 400

    try:
        path = kagglehub.dataset_download("shwetabh123/mall-customers")
        # find a csv file in the downloaded path
        csv_candidates = glob.glob(os.path.join(path, '*.csv'))
        if not csv_candidates:
            return jsonify({'error': 'No CSV found in downloaded dataset'}), 400
        src_csv = csv_candidates[0]

        # normalize columns and save into backend directory
        df = pd.read_csv(src_csv)
        # reuse normalization from validate function
        rename_map = {
            'Annual Income (k$)': 'Annual Income',
            'Spending Score (1-100)': 'Spending Score',
        }
        df = df.rename(columns=rename_map)
        # if CustomerID is missing, synthesize one
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = [f"{i+1:04d}" for i in range(len(df))]

        target_filename = 'Mall_Customers.csv'
        target_path = os.path.join(DATA_DIR, target_filename)
        df.to_csv(target_path, index=False)

        # validate and log dataset
        clean = validate_and_read_csv(target_path)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO datasets (filename, row_count, uploaded_at) VALUES (?, ?, ?)",
            (target_filename, len(clean), datetime.utcnow().isoformat())
        )
        conn.commit()
        dataset_id = cur.lastrowid
        conn.close()

        return jsonify({'message': 'Sample dataset downloaded', 'dataset_id': dataset_id, 'filename': target_filename, 'row_count': len(clean)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'error': 'File too large',
        'max_mb': int(app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024))
    }), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
