import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .preprocessing import simple_impute, encode_and_select


def cluster_anomalies(scores_csv, out_dir, n_clusters=3, sample_for_plot=5000):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(scores_csv)
    # Ensure boolean
    if 'is_anomaly' in df.columns and df['is_anomaly'].dtype != bool:
        df['is_anomaly'] = df['is_anomaly'].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
    an = df[df['is_anomaly'] == True].copy()
    if an.empty:
        raise ValueError('No anomalies to cluster')

    # Drop score columns and timestamp for clustering features
    drop_cols = ['Timestamp', 'anomaly_score', 'is_anomaly']
    features_df = an.drop(columns=[c for c in drop_cols if c in an.columns])
    features_df = simple_impute(features_df)

    X, feature_names, scaler = encode_and_select(features_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    an['cluster'] = labels

    clusters_csv = os.path.join(out_dir, 'anomaly_clusters.csv')
    an.to_csv(clusters_csv, index=False)
    print(f'Wrote clustered anomalies to {clusters_csv}')

    # Summarize clusters
    summary_lines = []
    summary_lines.append(f'# Cluster summary (k={n_clusters})\n\n')
    for k in range(n_clusters):
        sub = an[an['cluster'] == k]
        summary_lines.append(f'## Cluster {k} — {len(sub)} rows ({len(sub)/len(an):.2%})\n\n')
        # For categorical columns, show top values
        cat_cols = sub.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ['Timestamp']]
        for c in cat_cols:
            summary_lines.append(f'### {c}\n\n')
            vc = sub[c].value_counts(normalize=True).head(10)
            summary_lines.append('| value | pct |\n')
            summary_lines.append('|---|---:|\n')
            for v, p in vc.items():
                summary_lines.append(f'| {v} | {p:.3%} |\n')
            summary_lines.append('\n')
        # Numeric columns
        num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            summary_lines.append(f'- **{c}** mean: {sub[c].mean():.4f}\n')
        summary_lines.append('\n')

    summary_md = os.path.join(out_dir, 'cluster_summary.md')
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
    print(f'Wrote cluster summary to {summary_md}')

    # PCA scatter plot
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        # sample for plot
        n = proj.shape[0]
        if n > sample_for_plot:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=sample_for_plot, replace=False)
            proj_s = proj[idx]
            labels_s = labels[idx]
        else:
            proj_s = proj
            labels_s = labels
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for k in range(n_clusters):
            mask = labels_s == k
            plt.scatter(proj_s[mask, 0], proj_s[mask, 1], s=10, label=f'cluster {k}', alpha=0.6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.title(f'PCA of anomalies (k={n_clusters})')
        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'anomaly_clusters_pca.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Wrote cluster PCA plot to {plot_path}')
    except Exception as e:
        print('Failed to create PCA scatter:', e)

    return {
        'clusters_csv': clusters_csv,
        'summary_md': summary_md,
        'plot': plot_path if 'plot_path' in locals() else None,
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scores_csv', required=True)
    p.add_argument('--out_dir', default='output')
    p.add_argument('--n_clusters', type=int, default=3)
    args = p.parse_args()

    cluster_anomalies(args.scores_csv, args.out_dir, n_clusters=args.n_clusters)
