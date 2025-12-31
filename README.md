# Fraud Detection Project – Bank & E-Commerce Transactions
---

## Project Overview
This project aims to **detect fraudulent transactions** in e-commerce and credit card datasets using machine learning. It includes **data preprocessing, feature engineering, predictive modeling, and model explainability** to provide actionable insights for fraud prevention.

---

## Datasets
1. **Fraud_Data.csv** – 151,112 e-commerce transactions with user, device, time, and IP info.  
2. **creditcard.csv** – 284,807 credit card transactions with PCA-transformed numerical features.  

Both datasets are highly imbalanced, requiring careful modeling.

---

## Workflow Summary

### Task 1 – Data Analysis & Preprocessing
- Cleaned datasets, removed duplicates, corrected data types.  
- Explored class imbalance and transaction patterns.  
- Feature engineering: time-based, behavioral, velocity, geolocation, and interactions.  
- Scaled numerical features, one-hot encoded categorical features.  
- Handled class imbalance using **SMOTE** for training data.  

**Outcome:** Model-ready, clean, and feature-rich datasets.

### Task 2 – Modeling & Evaluation
- Models: Logistic Regression (baseline) and Random Forest (tuned).  
- Metrics: F1-score, Precision-Recall AUC, confusion matrices, stratified 5-fold CV.  
- Random Forest outperformed Logistic Regression.  
- Top predictive features identified using built-in feature importance.

### Task 3 – Model Explainability (SHAP)
- SHAP analysis for global and local explanations.  
- Fraud Dataset Top Features: `time_since_signup`, `short_account`, `purchase_value`, `purchase_velocity`, `hour_of_day`.  
- Credit Dataset Top PCA Features: `V14`, `V10`, `V17`, `V12`, `V4`.  
- Force plots generated for True Positive, False Positive, and False Negative examples.  

**Business Recommendations:**  
- Extra verification for new or short-lived accounts.  
- Monitor high-risk hours and countries.  
- Flag high-velocity or unusual transactions.  

**Outcome:** Predictions are interpretable and actionable.

---

## Project Structure
```bash
project-root/
│
├── data/
│   ├── raw/                  # Original CSV datasets
│   └── processed/            # Cleaned & feature-engineered CSVs
│
├── notebooks/
│   ├── eda-fraud-data.ipynb  # Cleaning and EDA for Fraud_Data.csv
│   ├── eda-creditcard.ipynb  # Cleaning and EDA for creditcard.csv
│   ├── feature-engineering_creditcard.ipynb # Feature engineering for both datasets
│   ├── modeling.ipynb  # Cleaning and EDA for Fraud_Data.csv
│   ├── shap-explainability.ipynb
│
├── scripts/                  # Optional: reusable functions/modules
│   ├── data_cleaning.py
│   └── feature_engineering.py
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview (this file)

````


---

## How to Run
1. Clone repository and install dependencies.  
2. Run notebooks in order: Task1 → Task2 → Task3.  
3. Outputs (plots, models) will be saved in `results/` and `models/`.

---

## References
- SHAP Documentation: [https://shap.readthedocs.io](https://shap.readthedocs.io)  
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)  
- SMOTE: Chawla et al., 2002
