import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_csv(path, nrows=None):
    """Load CSV into a DataFrame."""
    return pd.read_csv(path, nrows=nrows)


def simple_impute(df):
    """Impute missing values: numeric -> median, object/category -> 'missing'."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            df[col] = df[col].fillna('missing')
    return df


def encode_and_select(df, requested_features=None):
    """
    Encode categorical columns with one-hot (if not too many categories) and return X (numpy array) and feature names.
    If requested_features is provided, prefer those (if present).
    """
    df = df.copy()

    # If user provided requested features, try to use them
    if requested_features:
        present = [c for c in requested_features if c in df.columns]
        if present:
            df = df[present].copy()

    # Convert boolean-like strings to 0/1
    bool_map = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0}
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].str.lower().replace(bool_map)
            if lowered.isin([0, 1]).all():
                df[col] = lowered.astype(float)

    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # One-hot encode categorical columns with limited cardinality
    max_cardinality = 20
    encoded = []
    encoded_names = []
    for c in cat_cols:
        if df[c].nunique() <= 1:
            # skip constant column
            continue
        if df[c].nunique() <= max_cardinality:
            d = pd.get_dummies(df[c], prefix=c, dummy_na=False)
            encoded.append(d)
            encoded_names.extend(d.columns.tolist())
        else:
            # Too many categories: try label encoding of top categories
            top = df[c].value_counts().nlargest(max_cardinality).index
            d = df[c].where(df[c].isin(top), other='__other__')
            dummies = pd.get_dummies(d, prefix=c, dummy_na=False)
            encoded.append(dummies)
            encoded_names.extend(dummies.columns.tolist())

    parts = []
    if numeric_cols:
        parts.append(df[numeric_cols])
    if encoded:
        parts.extend(encoded)

    if not parts:
        raise ValueError("No usable columns after encoding/selection. Provide a CSV with numeric or categorical columns.")

    Xdf = pd.concat(parts, axis=1)
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf.values)
    feature_names = Xdf.columns.tolist()
    return X, feature_names, scaler
