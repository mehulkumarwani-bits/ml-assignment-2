"""
knn.py
---------------------------------
Module providing a K-Nearest Neighbors classifier and helper functions.
"""

import os
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier


MODEL_FILENAME = "KNN.joblib"


def get_estimator(n_neighbors: int = 5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


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
