"""
`model` package initializer.

This module exposes a mapping between display model names and their module
filenames. Other parts of the project (e.g. `train.py`, `streamlit_app.py`) can
import this mapping to discover and load model modules programmatically.

The mapping keys are the friendly names used in the rest of the project and the
values are the Python module filenames (without the `.py` suffix) under this
package.
"""

MODEL_MODULES = {
	"LogisticRegression": "logistic_regression",
	"DecisionTree": "decision_tree",
	"KNN": "knn",
	"GaussianNB": "gaussian_nb",
	"RandomForest": "random_forest",
	"XGBoost": "xgboost_model",
}

def get_module_name(display_name: str):
	return MODEL_MODULES.get(display_name)

