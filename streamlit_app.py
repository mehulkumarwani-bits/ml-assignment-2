"""
streamlit_app.py
---------------------------------
Simple Streamlit app for interactive demo of trained models.

Features:
- Loads feature names and default values from the Breast Cancer dataset
- Loads `scaler.joblib` and model `*.joblib` files from the `model/` directory
- Lets user select a model, modify feature values, and run prediction
- Displays predicted class, probability (if available), and model metrics

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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Page Configuration
    st.set_page_config(page_title="ML Assignment 2", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .header-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 1rem; }
        .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
        <div class="header-section">
            <h1>🔬 Breast Cancer Detection - ML Model Demo</h1>
            <p><strong>Student ID:</strong> 2025AA05133 | <strong>Name:</strong> Mehul Kumar Wani</p>
            <p><strong>Dataset:</strong> UCI Breast Cancer Wisconsin | <strong>Samples:</strong> 569 | <strong>Features:</strong> 30</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    info = load_feature_info()
    feature_names = info["feature_names"]
    defaults = info["defaults"]
    mins = info["mins"]
    maxs = info["maxs"]

    models = available_models()
    if not models:
        st.warning("No trained models found in `model/`. Run `python app.py train` first.")
        return

    # Sidebar controls with improved styling
    st.sidebar.markdown("### ⚙️ Model Selection")
    model_choice = st.sidebar.selectbox("Choose a model:", models, help="Select the ML model to use for predictions")

    # Display model metrics in sidebar
    metrics_df = load_metrics_df()
    if not metrics_df.empty and model_choice in metrics_df["Model"].values:
        row = metrics_df[metrics_df["Model"] == model_choice].iloc[0]
        with st.sidebar.expander(f"📊 {model_choice} Metrics", expanded=True):
            col1, col2 = st.sidebar.columns(2)
            metrics_to_show = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
            for i, col_name in enumerate(metrics_to_show):
                if col_name in row.index:
                    target_col = col1 if i % 2 == 0 else col2
                    target_col.metric(col_name, f"{row[col_name]:.4f}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.info("👇 Use the sections below to:\n\n1. **Evaluate Models** - Upload test data and compare model performance\n\n2. **Make Predictions** - Enter feature values to predict individual cases")

    # Model Evaluation and Test Data Section with Tabs
    tab1, tab2 = st.tabs(["🧪 Model Evaluation", "🎯 Single Prediction"])
    
    with tab1:
        st.markdown("### Test Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Download Sample Test Data")
            test_csv_path = os.path.join("data", "test_data.csv")
            try:
                with open(test_csv_path, "rb") as f:
                    st.download_button(
                        label="📥 Download test_data.csv",
                        data=f,
                        file_name="test_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except FileNotFoundError:
                st.error("Sample test CSV not found. Run training first.")
        
        with col2:
            st.markdown("#### Upload Your Test Data")
            uploaded_file = st.file_uploader(
                'Choose a CSV file',
                type='csv',
                key='test_data_upload',
                help="Upload your test data CSV file (last column should be target)"
            )

        if uploaded_file is not None:
            test_data = pd.read_csv(uploaded_file)
            
            with st.expander("📊 Test Data Preview", expanded=True):
                st.dataframe(test_data.head(10), use_container_width=True)
                st.caption(f"Total records: {len(test_data)} | Features: {test_data.shape[1]}")
            
            st.markdown("### Evaluate Model Performance")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_model = st.selectbox(
                    'Select a model for evaluation',
                    available_models(),
                    help="Choose which model to evaluate"
                )
            
            with col3:
                evaluate_btn = st.button('🚀 Evaluate Model', use_container_width=True, key='eval_btn')

            if evaluate_btn:
                with st.spinner("Evaluating model..."):
                    try:
                        model = load_model(selected_model)
                        scaler = load_scaler()
                        
                        # Extract features and target from test data
                        X_test = test_data.iloc[:, :-1].values
                        y_test = test_data.iloc[:, -1].values
                        
                        if scaler is not None:
                            X_test = scaler.transform(X_test)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{accuracy:.4f}")
                        col2.metric("Precision", f"{precision:.4f}")
                        col3.metric("Recall", f"{recall:.4f}")
                        col4.metric("F1 Score", f"{f1:.4f}")
                        
                        # Display Classification Report
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("#### Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(5, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                                       xticklabels=['Malignant', 'Benign'],
                                       yticklabels=['Malignant', 'Benign'])
                            ax.set_xlabel('Predicted', fontsize=11)
                            ax.set_ylabel('Actual', fontsize=11)
                            ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=12, fontweight='bold')
                            st.pyplot(fig, use_container_width=True)
                        
                        st.success(f"✅ Evaluation complete for {selected_model}")
                        
                    except FileNotFoundError as e:
                        st.error(f"❌ Model not found: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error during evaluation: {str(e)}")
        else:
            st.info("👆 Upload a test CSV file to start model evaluation")

    
    with tab2:
        st.markdown("### Make Individual Predictions")
        st.markdown("Adjust the feature values below and click **Predict** to get a prediction for a single case.")
        
        # Build feature inputs in the main area using sliders (or number_input for wide ranges)
        st.markdown("#### Enter Feature Values")
        inputs = []
        with st.form(key="feature_form"):
            cols = st.columns(3)
            for i, fname in enumerate(feature_names):
                # Use percentiles to set slider ranges; default to median
                min_val = float(mins[i])
                max_val = float(maxs[i])
                default_val = float(defaults[i])
                # Choose slider step based on range
                step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01
                col_idx = i % 3
                with cols[col_idx]:
                    val = st.number_input(
                        f"{fname}",
                        value=default_val,
                        min_value=min_val,
                        max_value=max_val,
                        step=step,
                        format="%.4f"
                    )
                inputs.append(val)

            col1, col2, col3 = st.columns([1, 1, 2])
            with col2:
                submit = st.form_submit_button("🚀 Predict", use_container_width=True)

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

            # Display prediction with styling
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            class_label = "Benign (1)" if int(y_pred) == 1 else "Malignant (0)"
            color_class = "#10b981" if int(y_pred) == 1 else "#ef4444"
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                    <div style="background-color: {color_class}20; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {color_class};">
                        <h3 style="color: {color_class}; margin: 0;">Predicted Class: {class_label}</h3>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Using {model_choice}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            if prob_pos is not None:
                with col2:
                    st.metric("Confidence", f"{prob_pos:.1%}")
            
            # Show model coefficients or feature importances if available
            with st.expander("📈 Model Details", expanded=False):
                if hasattr(model, "coef_"):
                    st.write("**Model Coefficients (Top 10)**")
                    coefs = np.ravel(model.coef_)
                    table = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
                    table = table.sort_values(by="Coefficient", key=lambda s: s.abs(), ascending=False).head(10)
                    st.dataframe(table, use_container_width=True)
                elif hasattr(model, "feature_importances_"):
                    st.write("**Feature Importances (Top 10)**")
                    imps = model.feature_importances_
                    table = pd.DataFrame({"Feature": feature_names, "Importance": imps})
                    table = table.sort_values(by="Importance", ascending=False).head(10)
                    st.dataframe(table, use_container_width=True)
                else:
                    st.write("No coefficient or feature importance attributes available for this model.")


if __name__ == "__main__":
    main()
