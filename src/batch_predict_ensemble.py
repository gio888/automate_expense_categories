import os
import pandas as pd
import numpy as np
import joblib
from scipy.stats import mode

# ✅ Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "Cash 2024-06.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed_cash_2024-06.csv")

# ✅ Load Models and TF-IDF Vectorizer
print("🔍 Loading models and vectorizer...")
lgbm_model = joblib.load("lgbm_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
catboost_model = joblib.load("catboost_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ✅ Step 1: Load Data
print("📂 Loading transaction data...")
df = pd.read_csv(DATA_FILE)

# ✅ Step 2: Predict Categories Using Ensemble
def predict_category_ensemble(description):
    """Predict categories using an ensemble of models."""
    # Transform text description to TF-IDF features
    features = tfidf_vectorizer.transform([description])

    # Get predictions from all models
    preds = [
        lgbm_model.predict(features),
        xgb_model.predict(features),
        catboost_model.predict(features),
    ]
    
    # Majority voting for final category
    return mode(preds, axis=0).mode[0]

print("🔍 Predicting categories with ensemble...")
df["Category"] = df["Description"].apply(predict_category_ensemble)

# ✅ Step 3: Calculate "Amount (Negated)" and Format Amounts
df["Amount (Negated)"] = np.where(df["Out"].notna(), df["Out"], df["In"])
df["Amount"] = np.where(df["In"].notna(), df["In"], df["Out"])

# ✅ Step 4: Keep Only Required Columns
df = df[["Date", "Description", "Category", "Amount (Negated)", "Amount"]]

# ✅ Step 5: Save Processed Data
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Batch processing complete! Results saved to: {OUTPUT_FILE}")
