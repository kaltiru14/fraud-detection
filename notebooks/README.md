Task 1: Data Analysis and Preprocessing
---------------------------------------

### Objective

* Understand the structure and characteristics of the e-commerce fraud dataset.
    
* Identify and handle class imbalance.
    
* Engineer features to make the dataset modeling-ready.
    

Dataset
-------

**1\. Fraud\_Data.csv**

*   Contains e-commerce transaction data.
    
*   Key columns: user\_id, signup\_time, purchase\_time, purchase\_value, device\_id, source, browser, sex, age, ip\_address, class.
    

**2\. IpAddress\_to\_Country.csv**

*   Maps IP address ranges to countries.
    
*   Columns: lower\_bound\_ip\_address, upper\_bound\_ip\_address, country.
    

Steps Performed
---------------

### 1\. Data Cleaning

*   Checked for missing values → none found.
    
*   Removed duplicate transactions.
    
*   Converted signup\_time and purchase\_time to datetime objects.
    
*   Converted ip\_address to integer for geolocation mapping.
    

### 2\. Exploratory Data Analysis (EDA)

*   **Univariate analysis:** distribution of purchase values, user age, etc.
    
*   **Bivariate analysis:** features vs target variable.
    
*   Quantified **class imbalance**: only ~10% of transactions are fraudulent.
    
*   Fraud patterns by source, browser, and country were analyzed.
    

### 3\. Geolocation Integration

*   Converted IP addresses to integer.
    
*   Merged Fraud\_Data.csv with IpAddress\_to\_Country.csv using range-based lookup.
    
*   Identified countries with high fraud rates.
    

### 4\. Feature Engineering

*   **Time-based features:**
    
    *   hour\_of\_day, day\_of\_week, time\_since\_signup (in hours), short\_account (<24h old).
        
*   **Transaction frequency/velocity:**
    
    *   user\_txn\_count (total transactions per user).
        
    *   txn\_in\_24h (transactions in the last 24 hours per user).
        
*   **Categorical encoding:** one-hot encoding for source, browser, sex, country.
    
*   **Numeric scaling:** StandardScaler applied to purchase\_value, age, time\_since\_signup, user\_txn\_count, txn\_in\_24h.
    

### 5\. Handling Class Imbalance

*   Applied **SMOTE** on the dataset to demonstrate resampling.
    
*   ✅ **Important:** SMOTE will be applied only on the training set during modeling to prevent data leakage.
    

### 6\. Saved Processed Dataset

*   Cleaned and feature-engineered dataset saved as:../data/processed/fraud\_features.csv
    

Key Insights from Task 1
------------------------

*   Fraudulent transactions tend to have **higher purchase values**.
    
*   Short-lived accounts and certain traffic sources are more prone to fraud.
    
*   Some countries show disproportionately high fraud rates — useful for **risk-based verification**.
    
*   Severe **class imbalance** necessitates careful choice of evaluation metrics (F1-score, AUC-PR).
    

Next Steps
----------

*   **Task 2 — Model Building and Training:**
    
    *   Train baseline and ensemble models.
        
    *   Evaluate using metrics suitable for imbalanced classification.
        
    *   Perform hyperparameter tuning and cross-validation.
        

References
----------

*   [Kaggle: Fraud Detection Dataset](https://www.kaggle.com)
    
*   [Handling Imbalanced Data — imbalanced-learn](https://imbalanced-learn.org/)
    
*   [IP Geolocation Processing in Python](https://www.geeksforgeeks.org/ip-address-to-country-in-python/)