# backend_service/custom_transformers.py
import pandas as pd
import numpy as np

def cast_to_object(X):
    return X.astype(object)

def cast_to_bool(X):
    def convert_element(val):
        if isinstance(val, str):
            return val.lower() in ['true', '1', 'yes', 't']
        return bool(val)
    
    if isinstance(X, pd.DataFrame):
        return X.apply(lambda s: s.map(convert_element) if isinstance(s, pd.Series) else [convert_element(v) for v in s], axis=0).astype(bool)
    elif isinstance(X, pd.Series):
        return X.map(convert_element).astype(bool)
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            return np.array([convert_element(val) for val in X], dtype=bool)
        else:
            return np.array([[convert_element(val) for val in row] for row in X], dtype=bool)
    else:
        return pd.Series(X).map(convert_element).astype(bool).to_numpy()

def cast_to_string(X):
    return X.astype(str)
