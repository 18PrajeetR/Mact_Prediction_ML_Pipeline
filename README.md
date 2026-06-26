# 🚗 MACT Compensation Prediction using Machine Learning

A Machine Learning pipeline for predicting **Motor Accident Claims Tribunal (MACT) compensation** for **death cases** using **Ridge Regression with automatic hyperparameter tuning (RidgeCV)**.

The project preprocesses accident claim data, performs feature engineering, trains a RidgeCV regression model, evaluates its performance, and exports the trained artifacts for deployment in a Streamlit application.

---

## 📌 Project Overview

Motor Accident Claims Tribunal (MACT) compensation is determined based on several legal and demographic factors such as:

- Victim Age
- Monthly Income
- Number of Dependents
- Employment Type
- Marital Status
- Insurance Validity
- Vehicle Type
- State
- Negligence Percentage
- Deduction Fraction

This project predicts the **Total Compensation Awarded** using historical data and a robust regression pipeline.

The model is specifically designed for **Death Cases** and does not predict compensation for injury cases.

---

## ✨ Features

- Complete ML pipeline
- Automatic preprocessing
- One-Hot Encoding of categorical variables
- Feature scaling using StandardScaler
- Log transformation of target variable
- Ridge Regression with automatic alpha tuning (RidgeCV)
- 5-Fold Cross Validation
- Model evaluation using multiple regression metrics
- Export trained model for deployment
- Streamlit-ready prediction artifacts

---

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- Joblib

---

## 📂 Project Structure

```
.
│
├── MACT_ML_Project.ipynb
├── dataset.csv
│
├── mact_ridgecv_model.pkl
├── mact_scaler_death.pkl
├── mact_feature_names.pkl
│
└── README.md
```

---

## 📊 Machine Learning Pipeline

### 1. Data Loading

The dataset is loaded using Pandas.

```python
pd.read_csv(...)
```

---

### 2. Data Preprocessing

The preprocessing pipeline performs:

- Removes redundant columns
- Keeps only **Death** cases
- Drops Injury cases
- Log transforms compensation amount
- One-hot encodes categorical features

Removed columns:

- S_No
- Multiplier
- Future_Prospects_Pct
- Dependent_Parents

---

### 3. Train-Test Split

- 80% Training
- 20% Testing

Random State:

```
42
```

---

### 4. Feature Scaling

Only numerical columns are standardized using:

```
StandardScaler
```

---

### 5. Model

The model used is:

# RidgeCV

Instead of manually selecting the regularization parameter (alpha), RidgeCV automatically chooses the best alpha using **5-Fold Cross Validation**.

Candidate alpha values:

```
0.01
0.1
1
5
10
50
100
1000
```

---

### 6. Cross Validation

The model performs

- 5-Fold Cross Validation

Evaluation Metric:

- R² Score

---

### 7. Model Evaluation

Predictions are converted back from logarithmic scale to actual rupee values before evaluation.

Metrics used:

- RMSE (₹)
- R² Score
- MAPE (%)

---

## 📈 Output Metrics

The pipeline reports:

- Best Alpha
- R² Score
- RMSE
- Mean Absolute Percentage Error (MAPE)

Example output:

```
Model : RidgeCV

Alpha : Auto Selected

R² : xxxx

RMSE : ₹xxxx

MAPE : xx.xx%
```

(The exact values depend on the dataset used.)

---

## 💾 Saved Artifacts

After training, the following files are generated.

### Trained Model

```
mact_ridgecv_model.pkl
```

Contains the trained RidgeCV model.

---

### Standard Scaler

```
mact_scaler_death.pkl
```

Used to normalize numerical features during inference.

---

### Feature Names

```
mact_feature_names.pkl
```

Stores the ordered feature list to ensure incoming prediction data matches the training feature order.

---

## 🔮 Prediction Pipeline

For new accident cases:

1. Prepare input data
2. One-hot encode categorical values
3. Align feature columns
4. Scale numerical columns
5. Predict log compensation
6. Convert back to rupee value using exponential transformation

The prediction helper automatically performs these steps.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/mact-compensation-prediction.git
```

Navigate into the project

```bash
cd mact-compensation-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

Execute the notebook or run the training pipeline.

```python
run_pipeline("dataset.csv")
```

---

## 📦 Dependencies

```
numpy
pandas
scikit-learn
joblib
```

Install manually:

```bash
pip install numpy pandas scikit-learn joblib
```

---

## 📌 Future Improvements

- Support Injury Case prediction
- Compare multiple regression models (XGBoost, Random Forest, LightGBM)
- Explainability using SHAP
- Hyperparameter optimization
- REST API deployment
- Docker support
- Cloud deployment
- Interactive Streamlit dashboard

---

## 👨‍💻 Author

Developed as a Machine Learning solution for predicting MACT compensation awards using Ridge Regression with automated model selection.

---

## 📄 License

This project is intended for academic and research purposes.
