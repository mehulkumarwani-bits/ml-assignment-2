"""
gaussian_nb.py
---------------------------------
Module providing a Gaussian Naive Bayes classifier and helper functions.
"""

import os
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB


MODEL_FILENAME = "GaussianNB.joblib"


def get_estimator():
    return GaussianNB()


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
    raise FileNotFoundError(path)
