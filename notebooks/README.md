Fraud Detection Project – E-Commerce & Credit Card Transactions
===============================================================

**1\. Project Overview**
------------------------

This project aims to **improve the detection of fraudulent transactions** for e-commerce and bank credit card datasets. Fraud detection is critical for financial institutions and e-commerce platforms to **prevent financial loss** and **maintain customer trust**.

The solution leverages:

*   **Data cleaning and exploratory data analysis (EDA)**
    
*   **Feature engineering** (time-based, transaction velocity, geolocation, interaction terms)
    
*   **Scaling and encoding of features**
    
*   **Class imbalance handling using SMOTE**
    
*   Preprocessing pipelines for **robust and reproducible modeling**
    

**2\. Business Objective**
--------------------------

*   Identify fraudulent transactions early to **reduce financial losses**.
    
*   Provide actionable insights to **risk management teams**.
    
*   Develop a **scalable and reproducible pipeline** for transaction monitoring.
    

**3\. Dataset Description**
---------------------------

### **Fraud\_Data.csv**

*   Transactions from an e-commerce platform.
    
*   Features include:
    
    *   user\_id, signup\_time, purchase\_time, purchase\_value
        
    *   device\_id, source, browser, sex, age, ip\_address
        
    *   class (0 = Legitimate, 1 = Fraud)
        
*   Enriched with **IP-to-country mapping** using IpAddress\_to\_Country.csv.
    

### **creditcard.csv**

*   Credit card transaction dataset.
    
*   Features:
    
    *   Time, Amount, anonymized PCA features (V1–V28)
        
    *   Class (0 = Legit, 1 = Fraud)
        

**4\. Repository Structure**
----------------------------
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
│   └── feature-engineering.ipynb # Feature engineering for both datasets
│
├── scripts/                  # Optional: reusable functions/modules
│   ├── data_cleaning.py
│   └── feature_engineering.py
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview (this file)

````

**5\. Workflow**
----------------

1.  **Data Cleaning**
    
    *   Remove duplicates
        
    *   Handle missing values
        
    *   Correct datatypes (especially datetime columns)
        
2.  **Exploratory Data Analysis (EDA)**
    
    *   Univariate: distribution of key features
        
    *   Bivariate: feature vs. fraud label
        
    *   Class imbalance analysis
        
3.  **Feature Engineering**
    
    *   Time-based: hour\_of\_day, day\_of\_week, time\_since\_signup
        
    *   Transaction frequency/velocity: transactions per user, rolling windows
        
    *   Geolocation-derived features: high-risk country flag
        
    *   Interaction terms: e.g., amount × hour\_of\_day
        
    *   Scaling and encoding (StandardScaler, One-Hot Encoding)
        
4.  **Train/Test Split**
    
    *   Stratified split to preserve class distribution
        
    *   Avoids leakage before SMOTE
        
5.  **Class Imbalance Handling**
    
    *   SMOTE applied **only on training set**
        
    *   Optional: compare with class weighting or undersampling
        
6.  **Modeling & Evaluation**
    
    *   Prepared datasets ready for modeling
        
    *   Precision–recall and business cost metrics considered
        

**6\. How to Run**
------------------

1.  Clone the repository:
    

```bash 
git clone <your-repo-url>
cd project-root
````

1.  Create and activate a Python environment:
    

```bash 
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

1.  Install dependencies:
    

```bash 
pip install -r requirements.txt
```

1.  Run notebooks in order:
    
    1.  eda-fraud-data.ipynb
        
    2.  eda-creditcard.ipynb
        
    3.  feature-engineering.ipynb
        
2.  Processed datasets will be saved in data/processed/ and are ready for modeling.
    

**7\. Key Highlights**
----------------------

*   Robust cleaning and EDA for **both datasets**
    
*   Comprehensive feature engineering:
    
    *   Time-based, transaction frequency, velocity, geolocation, interaction features
        
*   Leakage-safe train/test split with **SMOTE applied correctly**
    
*   Modular code structure for **reuse and maintainability**
    
*   Clear documentation for business stakeholders
    

**8\. Future Work**
-------------------

*   Compare alternative class imbalance strategies (e.g., class weights, undersampling)
    
*   Evaluate model performance using **business-specific cost metrics**
    
*   Extract reusable functions into standalone Python scripts for full pipeline automation
    
*   Implement basic **error handling and logging** for production readiness

## Task 2: Model Building & Training

In this task, we built, trained, and evaluated classification models to detect fraudulent transactions for both e-commerce and bank datasets. Special attention was given to handling class imbalance and selecting appropriate evaluation metrics.

### 1. Data Preparation
- Used processed train/test datasets:
  - E-commerce: `fraud_X_train`, `fraud_X_test`, `fraud_y_train`, `fraud_y_test`
  - Bank Credit: `credit_X_train`, `credit_X_test`, `credit_y_train`, `credit_y_test`
- Stratified splits preserved class distribution.

### 2. Baseline Model: Logistic Regression
- Trained with `class_weight='balanced'`.
- **Fraud Data (E-commerce)**:
  - F1-Score: 0.671
  - PR-AUC: 0.579
  - Confusion Matrix:  
    ```
    [[23120   256]
     [ 1087  1367]]
    ```
- **Credit Data**:
  - F1-Score: 0.233
  - PR-AUC: 0.759

### 3. Ensemble Model: Random Forest
- Trained with `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`.
- **Fraud Data (E-commerce)**:
  - F1-Score: 0.698
  - PR-AUC: 0.645
  - Confusion Matrix:  
    ```
    [[23306    70]
     [ 1101  1353]]
    ```
- **Credit Data**:
  - F1-Score: 0.670
  - PR-AUC: 0.796
  - Confusion Matrix:  
    ```
    [[56593    58]
     [   18    77]]
    ```
- **Top 10 Features (Fraud Data)**:  
  `short_account`, `time_since_signup`, `country_United States`, `age`, `sex_M`, `purchase_velocity`, `source_Direct`, `purchase_value`, `source_SEO`, `country_China`

### 4. Cross-Validation (Fraud Data)
- Stratified 5-fold CV F1 scores:
  - Logistic Regression: 0.855 ± 0.001
  - Random Forest: 0.723 ± 0.003

### 5. Model Comparison
| Model                | F1     | PR-AUC |
|----------------------|--------|--------|
| Logistic Regression  | 0.671  | 0.579  |
| Random Forest        | 0.698  | 0.645  |

**Observations:**  
- Random Forest outperformed Logistic Regression on both datasets, capturing complex patterns while handling class imbalance.  
- Logistic Regression provides interpretability, while Random Forest offers higher predictive power.  

### 6. Key Challenges
- **Class Imbalance:** Mitigated with `class_weight='balanced'` and appropriate metrics (F1, PR-AUC).  
- **Feature Relevance:** Evaluated using Random Forest feature importances.  
- **Interpretability vs Performance:** Logistic Regression offers transparency; Random Forest improves predictive performance.
