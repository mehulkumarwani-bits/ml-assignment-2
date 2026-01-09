"""
random_forest.py
---------------------------------
Module providing a RandomForest classifier and helper functions.
"""

import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier


MODEL_FILENAME = "RandomForest.joblib"


def get_estimator(random_state: int = 42, n_estimators: int = 100):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


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
