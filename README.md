# Machine Learning Assignment 2

## Problem statement

Build and evaluate supervised machine learning classifiers to detect whether
breast tumors are malignant or benign using the UCI Breast Cancer Wisconsin
(diagnostic) dataset. The tasks include:

- Load and preprocess the dataset (train/test split, feature scaling).
- Train multiple classifiers (Logistic Regression, Decision Tree, KNN,
  GaussianNB, Random Forest, and XGBoost when available).
- Evaluate models using Accuracy, AUC, Precision, Recall, F1-score and
  Matthews Correlation Coefficient (MCC).
- Save trained models and a metrics summary to the `model/` directory.
- Provide a simple Streamlit app to interactively load models, run
  predictions, and display results.

The repository contains the training pipeline (`train.py`), a CLI entrypoint
(`app.py`) and an interactive demo (`streamlit_app.py`). Models and metrics
are written to the `model/` folder after running the training pipeline.

## Dataset description

This project uses the UCI Breast Cancer Wisconsin (Diagnostic) dataset, which
is available through scikit-learn as `load_breast_cancer()`.

- Number of samples: 569
- Number of features: 30 numeric features (mean, standard error, and "worst"/largest
  values for 10 real-valued measurements)
- Classes: binary label (malignant = 0, benign = 1) — class distribution is
  212 malignant and 357 benign instances
- Feature examples: radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension (each with mean/SE/worst)

The dataset consists of measurements computed from digitized images of fine
needle aspirates of breast masses. It is commonly used for binary
classification benchmarking and is the dataset used by the training pipeline in
`train.py`.

## Models used

Comparison Table with the evaluation metrics calculated for all the 6 models as below:

| ML Model Name            | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression      |   0.9825 | 0.9954 |    0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            |   0.9123 | 0.9157 |    0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN                      |   0.9561 | 0.9788 |    0.9589 | 0.9722 | 0.9655 | 0.9054 |
| GaussianNB               |   0.9298 | 0.9868 |    0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) |   0.9561 | 0.9939 |    0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble)       |   0.9561 | 0.9901 |    0.9467 | 0.9861 | 0.9660 | 0.9058 |

Values are rounded to 4 decimal places for readability.

## Observations

Observations on the performance of each model on the chosen dataset as below:

| ML Model Name            | Observation about model performance                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression      | Best overall performer with highest accuracy and AUC; balanced precision and recall indicating reliable, generalizable predictions.                                                        |
| Decision Tree            | Lower overall accuracy and AUC; high precision but noticeably lower recall — tends to be more conservative and may miss some positive cases (higher variance/overfitting risk).            |
| KNN                      | Strong, balanced performance with high precision and recall; effective for this dataset but may be sensitive to feature scaling and k choice.                                              |
| Naive Bayes              | Good AUC despite slightly lower accuracy; fast and robust baseline, may be limited by its feature independence assumption.                                                                 |
| Random Forest (Ensemble) | High AUC and balanced metrics similar to KNN; strong ensemble performance with lower variance than a single decision tree.                                                                 |
| XGBoost (Ensemble)       | Competitive with RandomForest and Logistic Regression — high AUC and recall, slightly lower precision than the very best model; good choice when tuning gradient boosting hyperparameters. |
