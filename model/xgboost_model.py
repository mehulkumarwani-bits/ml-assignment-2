"""
xgboost_model.py
---------------------------------
Module providing an XGBoost classifier and helper functions.
If `xgboost` is not installed, the module will raise ImportError on import.
"""

import os
from joblib import dump, load

from xgboost import XGBClassifier


MODEL_FILENAME = "XGBoost.joblib"


def get_estimator(random_state: int = 42):
    return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)


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
