import argparse
import os
import numpy as np
import pandas as pd

from .preprocessing import load_csv, simple_impute, encode_and_select
from .model import train_isolation_forest, save_model
from .visualize import plot_pca_scatter


def run_pipeline(csv_path, out_dir, features=None, contamination=0.05, n_estimators=100):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading data from {csv_path}...")
    df = load_csv(csv_path)
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")

    print("Imputing missing values...")
    df2 = simple_impute(df)

    print("Encoding and scaling features...")
    X, feature_names, scaler = encode_and_select(df2, requested_features=features)
    print(f"Using {len(feature_names)} features")

    print("Training IsolationForest...")
    model, scores = train_isolation_forest(X, n_estimators=n_estimators, contamination=contamination)

    # decision_function: higher == more normal, lower == more abnormal
    # We'll define anomaly as being in the bottom `contamination` fraction via model.predict
    preds = model.predict(X)  # -1 for anomaly, 1 for normal
    is_anom = preds == -1

    # Save anomaly scores with original dataframe (don't modify index)
    out_df = df.copy()
    out_df['anomaly_score'] = scores
    out_df['is_anomaly'] = is_anom

    scores_csv = os.path.join(out_dir, 'anomaly_scores.csv')
    out_df.to_csv(scores_csv, index=False)
    print(f"Wrote anomaly scores to {scores_csv}")

    model_path = os.path.join(out_dir, 'isolation_forest.joblib')
    save_model(model, model_path)
    print(f"Saved model to {model_path}")

    pca_path = os.path.join(out_dir, 'pca_scatter.png')
    print("Creating PCA scatter plot (may sample large datasets)...")
    plot_pca_scatter(X, scores, is_anom, pca_path, title='PCA scatter (anomalies in red)')
    print(f"Wrote PCA plot to {pca_path}")

    # Write top anomalies
    topk = max(5, int(len(out_df) * contamination))
    top = out_df.sort_values('anomaly_score').head(topk)
    top_path = os.path.join(out_dir, 'top_anomalies.csv')
    top.to_csv(top_path, index=False)
    print(f"Wrote top {len(top)} anomalies to {top_path}")

    return {
        'scores_csv': scores_csv,
        'model_path': model_path,
        'pca_path': pca_path,
        'top_path': top_path,
    }


def _parse_args():
    p = argparse.ArgumentParser(description='Unsupervised anomaly detection pipeline')
    p.add_argument('--csv', dest='csv', required=True, help='Path to input CSV')
    p.add_argument('--out_dir', dest='out_dir', default='output', help='Directory to write outputs')
    p.add_argument('--features', dest='features', nargs='*', help='Optional list of feature column names to use')
    p.add_argument('--contamination', dest='contamination', type=float, default=0.05, help='Anomaly contamination fraction')
    p.add_argument('--n_estimators', dest='n_estimators', type=int, default=100, help='IsolationForest n_estimators')
    return p.parse_args()


def main():
    args = _parse_args()
    run_pipeline(args.csv, args.out_dir, features=args.features, contamination=args.contamination, n_estimators=args.n_estimators)


if __name__ == '__main__':
    main()
