import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
# -----------------------------
# CREDIT DATA — SHAP ANALYSIS
# -----------------------------
print("\nLoading Credit Data & Model...")

X_test_credit = pd.read_csv("../data/processed/credit_X_test.csv")
y_test_credit = pd.read_csv("../data/processed/credit_y_test.csv").values.ravel()

best_model_credit = joblib.load("../models/best_model_credit.pkl")

# -----------------------------
# Sample to make SHAP fast
# -----------------------------
SAMPLE_SIZE = 500

if len(X_test_credit) > SAMPLE_SIZE:
    X_sample_credit = X_test_credit.sample(SAMPLE_SIZE, random_state=42)
else:
    X_sample_credit = X_test_credit

print(f"Using {len(X_sample_credit)} rows for SHAP analysis (Credit)")

# -----------------------------
# Compute SHAP values
# -----------------------------
print("Computing SHAP values for Credit model...")

explainer_credit = shap.TreeExplainer(best_model_credit)
shap_values_credit = explainer_credit.shap_values(X_sample_credit)

# -----------------------------
# Save Summary Plot (Beeswarm)
# -----------------------------
print("Saving summary plots for Credit model...")

plt.title("SHAP Summary Plot — Credit Model")
shap.summary_plot(shap_values_credit[1], X_sample_credit, show=False)
plt.savefig("./results/shap_summary_credit.png", dpi=200, bbox_inches="tight")
plt.clf()

# -----------------------------
# Save Bar Plot
# -----------------------------
shap.summary_plot(shap_values_credit[1], X_sample_credit, plot_type="bar", show=False)
plt.savefig("./results/shap_bar_credit.png", dpi=200, bbox_inches="tight")
plt.clf()

print("Done! Files saved to ./results/ (Credit)")
