import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Setup
# -----------------------------
print("Loading Data & Model...")

X_test_fraud = pd.read_csv("../data/processed/fraud_X_test.csv")
y_test_fraud = pd.read_csv("../data/processed/fraud_y_test.csv").values.ravel()

best_model_fraud = joblib.load("../models/best_model_fraud.pkl")

os.makedirs("./results", exist_ok=True)

# -----------------------------
# 2. Sample to make SHAP fast
# -----------------------------
SAMPLE_SIZE = 500

if len(X_test_fraud) > SAMPLE_SIZE:
    X_sample = X_test_fraud.sample(SAMPLE_SIZE, random_state=42)
else:
    X_sample = X_test_fraud

print(f"Using {len(X_sample)} rows for SHAP analysis")

# -----------------------------
# 3. Compute SHAP values
# -----------------------------
print("Computing SHAP values...")

explainer = shap.TreeExplainer(best_model_fraud)
shap_values = explainer.shap_values(X_sample)

# -----------------------------
# 4. Save Summary Plot (Beeswarm)
# -----------------------------
print("Saving summary plots...")

plt.title("SHAP Summary Plot — Fraud Model")
shap.summary_plot(shap_values[1], X_sample, show=False)
plt.savefig("./results/shap_summary_fraud.png", dpi=200, bbox_inches="tight")
plt.clf()

# -----------------------------
# 5. Save Bar Plot
# -----------------------------
shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
plt.savefig("./results/shap_bar_fraud.png", dpi=200, bbox_inches="tight")
plt.clf()

print("Done! Files saved to ./results/")
