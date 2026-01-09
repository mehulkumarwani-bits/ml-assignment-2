"""
train.py
---------------------------------
Implements a full training pipeline for the assignment dataset.

This script does the following:
- Loads the UCI Breast Cancer Wisconsin dataset (meets >=12 features and >=500 instances)
- Splits the data into train/test sets
- Scales features using `StandardScaler`
- Trains six classifiers: Logistic Regression, Decision Tree, KNN, GaussianNB,
  RandomForest, and XGBoost
- Evaluates each model using Accuracy, AUC, Precision, Recall, F1, and MCC
- Saves trained models and a metrics CSV into the `model/` directory

The code is heavily commented to explain each step (as requested in the assignment).
"""

import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
import importlib
from model import MODEL_MODULES

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

try:
    # XGBoost is optional; will raise ImportError if not installed
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False


def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """Load the Breast Cancer dataset from scikit-learn.

    This dataset is a good fit for the assignment: 30 features and 569 instances.
    Returns feature matrix X, target vector y, and feature names list.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names


def preprocess(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Split data and scale features using StandardScaler.

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def get_models(random_state: int = 42) -> Dict[str, object]:
    """Return a dictionary with model name -> estimator instances.

    We instantiate each model with default but set random_state where applicable
    for reproducibility.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    else:
        # Keep placeholder so that pipeline remains transparent if XGBoost isn't installed
        models["XGBoost"] = None

    return models


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Compute required evaluation metrics for a fitted model.

    The assignment requests: Accuracy, AUC, Precision, Recall, F1, and MCC.
    """
    y_pred = model.predict(X_test)

    # Try to get probability for positive class if available for AUC
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        # If predict_proba is not available, try decision_function
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    if y_proba is not None:
        # For binary classification AUC is computed with positive-class probabilities
        try:
            metrics["AUC"] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics["AUC"] = float("nan")
    else:
        metrics["AUC"] = float("nan")

    return metrics


def train_and_save_models(save_dir: str = "model") -> pd.DataFrame:
    """Full pipeline: train all models, evaluate, save models and metrics.

    Returns a pandas DataFrame with metrics for each model.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess data
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

    # Save scaler for later use in deployment
    dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    # Discover model modules via `model.MODEL_MODULES` mapping and import them.
    results = []
    for name, module_fname in MODEL_MODULES.items():
        try:
            mod = importlib.import_module(f"model.{module_fname}")
        except Exception as e:
            print(f"Could not import module for {name}: {e}")
            continue

        print(f"Processing model module: {name} ({module_fname})")

        # Attempt to train using the module's train_and_save helper.
        try:
            # Each module defines `train_and_save(X_train, y_train, save_dir)`
            trained = mod.train_and_save(X_train, y_train, save_dir=save_dir)
            model_obj = trained
        except Exception as e:
            print(f"Failed to train {name} via module {module_fname}: {e}")
            continue

        # Evaluate
        metrics = evaluate_model(model_obj, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

        print(f"Saved {name} via module {module_fname}")

    # Consolidate metrics into DataFrame and save
    if results:
        df = pd.DataFrame(results)
        # Reorder columns for readability
        cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        # Some columns may be missing if metrics were NaN; use intersection
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        metrics_csv = os.path.join(save_dir, "metrics_summary.csv")
        df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics summary to {metrics_csv}")
        return df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # When executed as a script, run training and show the metrics table.
    df_metrics = train_and_save_models(save_dir="model")
    if not df_metrics.empty:
        print(df_metrics.to_string(index=False))
    else:
        print("No models were trained. Check that dependencies are installed.")
