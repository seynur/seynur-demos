# ml_sidecar/utils.py

import os
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
import numpy as np


def ensure_dir(path: str):
    """
    Create a directory if it does not exist.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_float(value):
    """
    Safely convert to float, fallback 0.0
    """
    try:
        return float(value)
    except:
        return 0.0


def scale_features(X: np.ndarray) -> (np.ndarray, StandardScaler):
    """
    Fit MinMax/Standard scaler on X and return (scaled_X, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply existing scaler to new feature vectors.
    """
    return scaler.transform(X)