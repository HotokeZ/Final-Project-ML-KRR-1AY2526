import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def plot_pca_scatter(X, scores, is_anomaly, out_path, labels=None, title=None, sample=5000):
    """Create 2D PCA scatter plot and save to out_path.
    X: scaled feature matrix
    scores: decision function scores (higher=normal)
    is_anomaly: boolean mask for anomalies (True = anomaly)
    labels: optional array with labels (not required)
    sample: max points to plot for speed
    """
    n = X.shape[0]
    if n > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=sample, replace=False)
        Xs = X[idx]
        scores_s = scores[idx]
        is_anomaly_s = is_anomaly[idx]
        if labels is not None:
            labels_s = labels[idx]
        else:
            labels_s = None
    else:
        Xs = X
        scores_s = scores
        is_anomaly_s = is_anomaly
        labels_s = labels

    pca = PCA(n_components=2)
    proj = pca.fit_transform(Xs)

    df = pd.DataFrame({'x': proj[:, 0], 'y': proj[:, 1], 'score': scores_s, 'is_anomaly': is_anomaly_s})

    plt.figure(figsize=(8, 6))
    # plot normal points
    normals = df[~df['is_anomaly']]
    plt.scatter(normals['x'], normals['y'], c='lightgray', s=8, label='normal', alpha=0.6)
    # plot anomalies
    anoms = df[df['is_anomaly']]
    if not anoms.empty:
        plt.scatter(anoms['x'], anoms['y'], c='red', s=20, label='anomaly')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
