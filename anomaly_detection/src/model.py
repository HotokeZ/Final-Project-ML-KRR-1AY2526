import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


def train_isolation_forest(X, n_estimators=100, contamination=0.05, random_state=42):
    """Train IsolationForest and return fitted model and scores (the lower, the more abnormal)."""
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    iso.fit(X)
    # sklearn's decision_function: higher -> more normal, lower -> more abnormal
    scores = iso.decision_function(X)
    return iso, scores


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
