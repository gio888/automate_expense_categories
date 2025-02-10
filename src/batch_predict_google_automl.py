import os
import pandas as pd
import numpy as np
from google.cloud import automl_v1beta1
import google.auth
from google.auth.transport.requests import Request
from google.auth.credentials import Credentials

# ✅ Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "Cash 2024-06.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed_cash_2024-06.csv")

# ✅ Google AutoML Configuration
PROJECT_ID = "gnucash-predict"  # 🔹 Replace with your actual project ID
MODEL_ID = "kitty_2019_2024"  # 🔹 Replace with your trained AutoML model ID

# ✅ Step 1: Authenticate using Web Login (No `client_secret.json` needed)
def authenticate_google():
    """Authenticate with Google Cloud using a web-based login."""
    creds, _ = google.auth.default()

    # If credentials are invalid, refresh and re-authenticate
    if not creds or not creds.valid:
        creds.refresh(Request())  # 🔹 Opens a browser for login if needed

    return creds

# ✅ Authenticate
creds = authenticate_google()

# ✅ Step 2: Connect to Google AutoML
client = automl_v1beta1.PredictionServiceClient(credentials=creds)
model_full_id = f"projects/{PROJECT_ID}/locations/us-central1/models/{MODEL_ID}"

# ✅ Step 3: Load the CSV file
print("📂 Loading transaction data...")
df = pd.read_csv(DATA_FILE)

# ✅ Step 4: Send transactions to AutoML for prediction
def predict_category(description):
    """Send a single transaction description to Google AutoML for prediction."""
    payload = {"text_snippet": {"content": description, "mime_type": "text/plain"}}
    request = automl_v1beta1.PredictRequest(name=model_full_id, payload=payload)
    response = client.predict(request=request)
    
    # Extract the top predicted category
    if response.payload:
        return response.payload[0].display_name
    return "Unknown"

print("🔍 Predicting categories using Google AutoML...")
df["Category"] = df["Description"].apply(predict_category)

# ✅ Step 5: Calculate "Amount (Negated)" and format Amounts correctly
df["Amount (Negated)"] = np.where(df["Out"].notna(), df["Out"], df["In"])
df["Amount"] = np.where(df["In"].notna(), df["In"], df["Out"])

# ✅ Step 6: Keep only required columns
df = df[["Date", "Description", "Category", "Amount (Negated)", "Amount"]]

# ✅ Step 7: Save processed transactions
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Batch processing complete! Results saved to: {OUTPUT_FILE}")
