# Diabetes-Risk-Prediction-ML

# Diabetes Risk Classifier

## Project Overview
This repository contains a machine learning project focused on predicting the risk of diabetes based on diagnostic measurements. The goal is to develop and evaluate several classification models to identify individuals at high risk for diabetes.

## Dataset
The project utilizes the PIMA Indian Diabetes Dataset (`diabetes.csv`), which includes medical diagnostic information for Pima Indian women. The dataset contains 8 clinical features and one outcome variable (0 for non-diabetic, 1 for diabetic).

## Data Preprocessing
- Handled missing values (represented as zeros in some columns) by replacing them with the median of the respective column, grouped by the `Outcome` variable.
- Data was split into training and testing sets.
- Features were scaled using `StandardScaler` to normalize the input data for better model performance.

## Models Implemented
Three different classification models were trained and evaluated:
1.  **Logistic Regression**: A baseline model to establish a performance benchmark.
2.  **Random Forest Classifier**: An ensemble learning method known for its robustness.
3.  **XGBoost Classifier**: A highly efficient and effective gradient boosting framework.

## Model Evaluation
Models were evaluated using various metrics, including Accuracy, Precision, Recall (emphasized as most important for this medical context), F1-Score, and ROC-AUC. Cross-validation was used during training to assess model generalization.

**Key Evaluation Metrics (on Test Set):**
- **Logistic Regression (Baseline)**: ROC-AUC: `0.827`
- **Random Forest**: ROC-AUC: `0.948`
- **XGBoost**: ROC-AUC: `0.954` (Best performing model)

## Feature Importance
An analysis of feature importance was conducted using the best-performing XGBoost model to understand which features contribute most significantly to the predictions.

## Visualizations
The notebook includes several visualizations to understand the data and model performance:
- Class distribution of the target variable.
- Histograms of feature distributions.
- Box plots showing feature distributions by outcome.
- ROC Curve comparison for all trained models (`roc_curves.png`).
- Feature Importance plot for the XGBoost model (`feature_importance.png`).

## Saved Artifacts
The trained model, scaler, and feature names are saved for future use and deployment:
- `model/diabetes_model.pkl`: The best-performing XGBoost model.
- `model/scaler.pkl`: The `StandardScaler` fitted on the training data.
- `model/feature_names.pkl`: A list of feature names used by the model.

## Usage (for future deployment or inference)
To use the saved model for inference, you can load these artifacts and apply them to new data:

```python
import joblib
import numpy as np
import pandas as pd

# Load the saved model, scaler, and feature names
loaded_model = joblib.load('model/diabetes_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')
loaded_feature_names = joblib.load('model/feature_names.pkl')

# Example new patient data (ensure order matches feature_names)
sample_data = pd.DataFrame([[
    6, 148, 72, 35, 169.5, 33.6, 0.627, 50
]], columns=loaded_feature_names)

# Scale the new data
sample_scaled = loaded_scaler.transform(sample_data)

# Make a prediction
prediction = loaded_model.predict(sample_scaled)[0]
probability = loaded_model.predict_proba(sample_scaled)[:, 1][0]

print(f"Prediction: {'Diabetic' if prediction == 1 else 'Not diabetic'}")
print(f"Probability of Diabetes: {probability:.2f}")
