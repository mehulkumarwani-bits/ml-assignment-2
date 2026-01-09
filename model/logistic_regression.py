"""
logistic_regression.py
---------------------------------
Module providing a Logistic Regression estimator and helper functions.

API:
- get_estimator(random_state=42) -> returns an untrained sklearn LogisticRegression
- train_and_save(X_train, y_train, save_dir) -> trains estimator and saves joblib
- load_trained(save_dir) -> loads trained model if present, otherwise trains and saves

This module is intended to be imported by `train.py` or `streamlit_app.py` so the
project refers to model implementations by module rather than only raw joblib files.
"""

import os
from joblib import dump, load
from sklearn.linear_model import LogisticRegression


MODEL_FILENAME = "LogisticRegression.joblib"


def get_estimator(random_state: int = 42):
    """Return an untrained LogisticRegression estimator configured for this task."""
    return LogisticRegression(max_iter=1000, random_state=random_state)


def train_and_save(X_train, y_train, save_dir: str = "model"):
    os.makedirs(save_dir, exist_ok=True)
    model = get_estimator()
    model.fit(X_train, y_train)
    path = os.path.join(save_dir, MODEL_FILENAME)
    dump(model, path)
    return model


def load_trained(save_dir: str = "model"):
    path = os.path.join(save_dir, MODEL_FILENAME)
    if os.path.exists(path):
        return load(path)
    # If not present, raise FileNotFoundError so callers can decide to train
    raise FileNotFoundError(path)
