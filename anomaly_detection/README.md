# Unsupervised Detection of Anomalous Wellness Profiles

This small project demonstrates an unsupervised pipeline to detect anomalous wellness profiles using an Isolation Forest.

What is included
- `src/` package with preprocessing, modeling, visualization, and a CLI runner (`main.py`).
- `tests/` with a tiny pytest that exercises the pipeline on synthetic data.
- `requirements.txt` listing Python packages to install.

Quick start (Windows PowerShell):

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the pipeline against a CSV file:

```powershell
python -m src.main --csv "path\to\your\mental_health_dataset.csv" --out_dir output --contamination 0.05
```

Outputs written to `output/`:
- `anomaly_scores.csv` — original rows with `anomaly_score` and `is_anomaly` columns
- `pca_scatter.png` — PCA 2D scatter with anomalies highlighted
- `isolation_forest.joblib` — trained model

Notes
- The script will try to use the common column names from your proposal. If they are missing, it will use numeric columns in the CSV.
- For very large CSVs, consider sampling for visualization.

If you want, I can: run tests here, adapt the feature list to the specific Kaggle CSV you have, or add a notebook with exploratory analysis.
