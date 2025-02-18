# Project File Dependencies

## 1. `transaction_types.py`
- Defines an `Enum` (`TransactionSource`) for categorizing transaction sources (`HOUSEHOLD` and `CREDIT_CARD`).
- Used by other modules to classify transactions.

### **Dependencies:**
- **Used in:**  
  - `auto_model_ensemble.py`
  - `batch_predict_ensemble.py`

---

## 2. `model_storage.py`
- Handles saving/loading models both locally and on Google Cloud Storage (GCS).
- Implements retry mechanisms for uploading models to GCS.
- Used for saving trained models and retrieving stored models.

### **Dependencies:**
- **Used in:**
  - `auto_model_ensemble.py`
  - `batch_predict_ensemble.py` (for loading models)

- **Depends on:**
  - `google.cloud.storage`
  - `joblib` for model serialization

---

## 3. `model_registry.py`
- Maintains a registry of all trained models, their versions, performance metrics, and storage locations.
- Can store the registry locally or back it up to GCS.
- Provides functions to register, retrieve, and update model information.

### **Dependencies:**
- **Used in:**
  - `auto_model_ensemble.py`
  - `merge_training_data.py`

- **Depends on:**
  - `google.cloud.storage` for cloud backup

---

## 4. `merge_training_data.py`
- Validates and merges transaction corrections into the training dataset.
- Fetches the latest model version from `model_registry.py` to align training data with the model in production.
- Validates transaction categories, amounts, and descriptions.

### **Dependencies:**
- **Uses:** `model_registry.py` to get the latest trained model version.
- **Produces:** Cleaned and validated training datasets used by `auto_model_ensemble.py`.

---

## 5. `auto_model_ensemble.py`
- The core model training script.
- Loads transaction data, vectorizes text, and trains multiple models (`lgbm`, `xgboost`, `catboost`) using `flaml.AutoML`.
- Saves trained models using `model_storage.py` and registers them with `model_registry.py`.

### **Dependencies:**
- **Uses:**
  - `transaction_types.py` for transaction source classification.
  - `model_storage.py` to store trained models.
  - `model_registry.py` to log model versions.
- **Consumes:** The validated training dataset produced by `merge_training_data.py`.
- **Produces:** Trained models that `batch_predict_ensemble.py` will use for inference.

---

## 6. `batch_predict_ensemble.py`
- Loads the latest trained models and vectorizers.
- Runs batch predictions for transactions using multiple models.
- Uses an ensemble approach, including majority voting across `lgbm`, `xgboost`, and `catboost` models.
- Saves results with confidence scores.

### **Dependencies:**
- **Uses:**
  - `transaction_types.py` for transaction classification.
  - `model_storage.py` to load models.
- **Consumes:** Models trained by `auto_model_ensemble.py`.
- **Produces:** Categorized transaction data for further analysis.

---

## How Everything is Connected

1. **Training Data Handling (`merge_training_data.py`)**  
   - Cleans and validates transaction data.
   - Ensures alignment with the latest trained model version.
   - Produces updated training data.

2. **Model Training (`auto_model_ensemble.py`)**  
   - Loads the validated training data.
   - Trains and saves models.
   - Registers models in `model_registry.py`.

3. **Batch Prediction (`batch_predict_ensemble.py`)**  
   - Loads trained models and vectorizers.
   - Applies ensemble predictions to categorize transactions.

4. **Model Management (`model_storage.py` & `model_registry.py`)**  
   - Stores models locally and on GCS.
   - Keeps track of trained model versions and performance.

5. **Transaction Classification (`transaction_types.py`)**  
   - Standardizes transaction sources for consistency across training and prediction.

---

## High-Level Flow

### **Training Pipeline:**
- `merge_training_data.py` → `auto_model_ensemble.py` → `model_storage.py` & `model_registry.py`

### **Prediction Pipeline:**
- `model_registry.py` & `model_storage.py` → `batch_predict_ensemble.py`

