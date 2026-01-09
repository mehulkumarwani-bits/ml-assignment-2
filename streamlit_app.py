"""
streamlit_app.py
---------------------------------
Simple Streamlit app for interactive demo of trained models.

Features:
- Loads feature names and default values from the Breast Cancer dataset
- Loads `scaler.joblib` and model `*.joblib` files from the `model/` directory
- Lets user select a model, modify feature values, and run prediction
- Displays predicted class, probability (if available), and model metrics

Run with:
    streamlit run streamlit_app.py

Note: install `streamlit` and other dependencies from `requirements.txt` before running.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import importlib
from joblib import load
from sklearn.datasets import load_breast_cancer
from model import MODEL_MODULES, get_module_name


MODEL_DIR = "model"


@st.cache_data
def load_feature_info() -> Dict[str, object]:
    """Load feature names and default values (median) from sklearn dataset.

    Using the same dataset as the training pipeline ensures consistent ordering
    of features and reasonable defaults for sliders.
    """
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    X = data.data
    defaults = np.median(X, axis=0)
    # Provide reasonable ranges for sliders using percentiles
    mins = np.percentile(X, 1, axis=0)
    maxs = np.percentile(X, 99, axis=0)
    return {
        "feature_names": feature_names,
        "defaults": defaults,
        "mins": mins,
        "maxs": maxs,
    }


@st.cache_data
def available_models(model_dir: str = MODEL_DIR) -> List[str]:
    """Return list of supported model display names from `model` package mapping."""
    # Use the mapping defined in `model/__init__.py` so UI shows same names.
    return sorted(list(MODEL_MODULES.keys()))


def load_model(name: str, model_dir: str = MODEL_DIR):
    """Load a trained model by importing its module and calling `load_trained`.

    If the trained artifact is missing the module should raise FileNotFoundError
    and the caller can prompt the user to run the training script.
    """
    module_fname = get_module_name(name)
    if module_fname is None:
        raise RuntimeError(f"Unknown model: {name}")
    try:
        mod = importlib.import_module(f"model.{module_fname}")
    except Exception as e:
        raise RuntimeError(f"Could not import module for {name}: {e}")

    # Each model module exposes `load_trained(save_dir)` which raises FileNotFoundError
    return mod.load_trained(save_dir=model_dir)


def load_scaler(model_dir: str = MODEL_DIR):
    path = os.path.join(model_dir, "scaler.joblib")
    if not os.path.exists(path):
        return None
    return load(path)


def load_metrics_df(model_dir: str = MODEL_DIR) -> pd.DataFrame:
    path = os.path.join(model_dir, "metrics_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    st.title("ML Assignment - Model Demo")
    st.markdown("This demo loads trained models from the `model/` directory.")

    info = load_feature_info()
    feature_names = info["feature_names"]
    defaults = info["defaults"]
    mins = info["mins"]
    maxs = info["maxs"]

    models = available_models()
    if not models:
        st.warning("No trained models found in `model/`. Run `python app.py train` first.")
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    model_choice = st.sidebar.selectbox("Choose model", models)

    metrics_df = load_metrics_df()
    if not metrics_df.empty and model_choice in metrics_df["Model"].values:
        row = metrics_df[metrics_df["Model"] == model_choice].iloc[0]
        st.sidebar.write("**Model Metrics (test set)**")
        for col in [c for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"] if c in row.index]:
            st.sidebar.write(f"{col}: {row[col]:.3f}")

    st.sidebar.markdown("---")
    st.sidebar.write("Modify feature values below and press **Predict**")

    # Build feature inputs in the main area using sliders (or number_input for wide ranges)
    st.header("Input features")
    inputs = []
    with st.form(key="feature_form"):
        for i, fname in enumerate(feature_names):
            # Use percentiles to set slider ranges; default to median
            min_val = float(mins[i])
            max_val = float(maxs[i])
            default_val = float(defaults[i])
            # Choose slider step based on range
            step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01
            val = st.number_input(f"{fname}", value=default_val, min_value=min_val, max_value=max_val, step=step, format="%.6f")
            inputs.append(val)

        submit = st.form_submit_button("Predict")

    if submit:
        # Convert inputs to array and scale
        X_input = np.array(inputs, dtype=float).reshape(1, -1)
        scaler = load_scaler()
        if scaler is not None:
            X_scaled = scaler.transform(X_input)
        else:
            X_scaled = X_input

        try:
            model = load_model(model_choice)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        # Predict class and probability if available
        try:
            y_proba = model.predict_proba(X_scaled)
            prob_pos = float(y_proba[0][1]) if y_proba.shape[1] > 1 else float(y_proba[0][0])
        except Exception:
            prob_pos = None

        y_pred = model.predict(X_scaled)[0]

        st.subheader("Prediction")
        st.write(f"Predicted class: **{int(y_pred)}**")
        if prob_pos is not None:
            st.write(f"Predicted probability (pos class): **{prob_pos:.3f}**")
        else:
            st.write("Predicted probability: not available for this model")

        # Show model coefficients or feature importances if available
        st.subheader("Model details")
        if hasattr(model, "coef_"):
            st.write("Model coefficients (first 10 shown):")
            coefs = np.ravel(model.coef_)
            table = pd.DataFrame({"feature": feature_names, "coef": coefs})
            st.dataframe(table.sort_values(by="coef", key=lambda s: s.abs(), ascending=False).head(10))
        elif hasattr(model, "feature_importances_"):
            st.write("Feature importances (top 10):")
            imps = model.feature_importances_
            table = pd.DataFrame({"feature": feature_names, "importance": imps})
            st.dataframe(table.sort_values(by="importance", ascending=False).head(10))
        else:
            st.write("No coefficient or feature importance attributes available for this model.")


if __name__ == "__main__":
    main()
