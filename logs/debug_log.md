# Debug Log

## **Session Log - 2025-02-19**

### **Summary of Actions**
- Ran batch prediction on `Statement UNIONBANK Visa 7152 2025-01 - Sheet2.csv`.
- Processed `151` transactions for `credit_card` source.
- Successfully loaded models and vectorizers for `credit_card`.
- Encountered issues related to missing household models, GCS client initialization, low confidence scores, and metric logging errors.

---

## **Issues Found & Analysis**

### **🚨 GCS Client Initialization Failure**
- **Error:** `Failed to initialize GCS client: expected str, bytes or os.PathLike object, not NoneType`
- **Root Cause:**
  - `GCP_CREDENTIALS_PATH` is missing or not set properly, causing failure in `ModelStorage` initialization.
- **Fix Needed:**
  - Ensure `GCP_CREDENTIALS_PATH` is set before running batch predictions:
    ```bash
    export GCP_CREDENTIALS_PATH=/path/to/your/gcp_credentials.json
    ```
  - Modify the code to provide a default or fail-safe credentials path.

### **⚠️ Household Models Not Found**
- **Warning:** `No models found for household`
- **Root Cause:**
  - Household-related models (`lgbm, xgboost, catboost, tfidf vectorizer, label encoder`) are either not trained, missing from storage, or not registered in the model registry.
- **Fix Needed:**
  - Check if `household` models exist in `model_registry.json`.
  - If missing, train and register models for the `household` transaction source.
  - Verify that household models are properly saved in `ModelStorage`.

### **🔍 High Rate of Low Confidence Predictions (56.29%)**
- **Warning:** `Low average confidence score: 46.15%`
- **Root Cause:**
  - Possible insufficient training data for certain categories.
  - Model hyperparameters may need adjustment.
  - Ensemble model weighting may need tuning.
- **Fix Needed:**
  - Re-evaluate training data distribution to ensure all categories have sufficient samples.
  - Consider adjusting ensemble model weighting to favor better-performing models.
  - Log and analyze misclassified samples separately.

### **🛑 Error Logging Metrics**
- **Error:** `Error logging metrics: No active exception to reraise`
- **Root Cause:**
  - `PredictionMonitor.log_metrics()` encountered an issue when trying to log prediction statistics.
- **Fix Needed:**
  - Add exception handling inside `log_metrics()` to catch errors gracefully.
  - Check if `prediction_metrics.jsonl` file is writable and correctly formatted.

---

## **📌 Next Steps (To-Do List)**

1. **Fix GCS Client Issue**
   - Verify `GCP_CREDENTIALS_PATH` and update code to handle missing credentials gracefully.
2. **Train and Register Household Models**
   - If models are missing, train and save them for household transactions.
3. **Improve Model Confidence**
   - Investigate low-confidence predictions and consider re-training models.
4. **Fix Metric Logging Issue**
   - Debug `log_metrics()` function to ensure metrics are properly saved.

---

## **Best Practice for Documentation**
- **Use a "Session Log" or "Work Log" Document**
  - This file serves as a record of debugging sessions.
  - Each session should include:
    - **Date & Time**
    - **Summary of Actions**
    - **Issues Found**
    - **Fixes Applied (or To-Do)**
- This ensures continuity and makes it easier to resume work without missing details.

---

## **Session Log - 2025-02-19**

### **Summary of Changes**

1. **Fixed the `credentials_path` error**:
   - Added a check for the `GCP_CREDENTIALS_PATH` environment variable in the `BatchPredictor` class.

2. **Fixed the `check_metrics` method**:
   - Removed the `raise` statement at the end of the `check_metrics` method to avoid the `No active exception to reraise` error.

## **📌 Next Steps (To-Do List)**

1. **Improve credit card ml model**
   - improve [model](../src/auto_model_ensemble.py) for better accuracy.
2. **Improve Inter-relations between python files**
   - Update the relations with updated code. Make sure all codes are necesssary and interdependecies identified
3. **Improve Model Confidence**
   - Investigate low-confidence predictions and consider re-training models.
4. **Fix Metric Logging Issue**
   - Debug `log_metrics()` function to ensure metrics are properly saved.