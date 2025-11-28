import tempfile
import os
import pandas as pd
import numpy as np

from src.preprocessing import simple_impute, encode_and_select
from src.model import train_isolation_forest


def make_synthetic():
    # Create simple data with one small anomalous cluster
    rng = np.random.RandomState(0)
    normal = rng.normal(loc=0, scale=1.0, size=(100, 3))
    anom = rng.uniform(low=8, high=10, size=(3, 3))
    data = np.vstack([normal, anom])
    df = pd.DataFrame(data, columns=['a', 'b', 'c'])
    # Add a categorical column
    df['cat'] = ['x'] * 100 + ['y'] * 3
    return df


def test_pipeline_synthetic():
    df = make_synthetic()
    df2 = simple_impute(df)
    X, feature_names, scaler = encode_and_select(df2)
    assert X.shape[0] == df.shape[0]
    assert X.shape[1] >= 3

    model, scores = train_isolation_forest(X, contamination=0.05)
    assert len(scores) == X.shape[0]
    # Expect at least one anomaly found
    preds = model.predict(X)
    assert (preds == -1).sum() >= 1
