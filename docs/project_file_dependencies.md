# Project File Dependencies

## Core Pipeline Scripts (src/)

### `transaction_types.py`
**Purpose**: Defines transaction source types (`HOUSEHOLD`, `CREDIT_CARD`) and their configurations
- **Used by**: All pipeline scripts for source classification
- **Dependencies**: None
- **Key exports**: `TransactionSource` enum, source configuration methods

### `model_storage.py` 
**Purpose**: Handles local and cloud model storage with fallback strategies
- **Used by**: `auto_model_ensemble.py`, `batch_predict_ensemble.py`, `verify_models.py`
- **Dependencies**: `google.cloud.storage`, `joblib`
- **Features**: Local/cloud sync, retry mechanisms, caching

### `model_registry.py`
**Purpose**: Tracks model versions, metrics, locations, and metadata
- **Used by**: `auto_model_ensemble.py`, `merge_training_data.py`, `batch_predict_ensemble.py`
- **Dependencies**: `google.cloud.storage` (optional cloud backup)
- **Storage**: `models/registry/model_registry.json`

### `transform_monthly_household_transactions.py`
**Purpose**: Converts Google Sheets exports to ML-ready format (household only)
- **Input**: `"House Kitty Transactions - Cash YYYY-MM.csv"`
- **Output**: `"House Kitty Transactions - Cash - Corrected YYYY-MM.csv"`
- **Dependencies**: None
- **Note**: Only used for household transactions, not credit card

### `batch_predict_ensemble.py`
**Purpose**: Runs ensemble predictions using source-specific models
- **Uses**: `transaction_types.py`, `model_storage.py`, `model_registry.py`
- **Input**: CSV files with Description column and transaction_source
- **Output**: `processed_{filename}_v{version}_{timestamp}.csv`
- **Features**: Ensemble voting, confidence scoring, source-aware model loading

### `merge_training_data.py`
**Purpose**: Validates corrections and merges with existing training data
- **Uses**: `model_registry.py`, `transaction_types.py`
- **Input**: Manually corrected CSV files
- **Output**: `training_data_{source}_v{version}_{timestamp}.csv`
- **Validation**: Categories, amounts, dates, descriptions, transaction sources

### `auto_model_ensemble.py`
**Purpose**: Trains source-specific ensemble models using FLAML AutoML
- **Uses**: `transaction_types.py`, `model_storage.py`, `model_registry.py`
- **Input**: Training data from `merge_training_data.py`
- **Output**: Models, vectorizers, label encoders per source
- **Models**: LightGBM, XGBoost, CatBoost (per transaction source)

### `verify_models.py`
**Purpose**: Checks model availability and downloads from cloud if needed
- **Uses**: `model_registry.py`, `model_storage.py`
- **Dependencies**: Same as model storage
- **Use case**: Troubleshooting missing models

## Utility Scripts (root/)

### `check_latest_data.py`
**Purpose**: Utility to examine latest data files
- **Dependencies**: Standard libraries only

### `check_registry.py`
**Purpose**: Utility to inspect model registry contents
- **Uses**: `src/model_registry.py`

## Data Flow Architecture

### Household Expense Flow
```
Google Sheets Export → transform_monthly_household_transactions.py → 
Corrected CSV → batch_predict_ensemble.py → Predictions → 
Manual Corrections → merge_training_data.py → Training Data → 
auto_model_ensemble.py → New Models
```

### Credit Card Flow
```
Bank CSV Export → batch_predict_ensemble.py → Predictions → 
Manual Corrections → merge_training_data.py → Training Data → 
auto_model_ensemble.py → New Models
```

### Model Management Flow
```
Training → model_storage.py (save) → model_registry.py (register) → 
GCS Backup → model_storage.py (load) → batch_predict_ensemble.py
```

## Key Dependencies by Function

### **Text Processing**
- `scikit-learn`: TF-IDF vectorization, label encoding
- `pandas`: Data manipulation and validation
- `numpy`: Numerical operations

### **Machine Learning**
- `flaml`: AutoML framework for model training
- `lightgbm`, `xgboost`, `catboost`: Ensemble algorithms
- `shap`: Model interpretability (optional)

### **Storage & Persistence**
- `joblib`: Model serialization
- `google.cloud.storage`: Cloud backup
- `json`: Registry and configuration files

### **Data Validation**
- `pandas`: Data type validation and cleaning
- Custom validation logic in `merge_training_data.py`

## Source-Specific Architecture

### **Household Models**
- **Vectorizer**: Word-level n-grams (1,2) for item descriptions
- **Models**: `household_lgbm_model_v{N}.pkl`, etc.
- **Categories**: Item-level expenses (groceries, utilities, etc.)

### **Credit Card Models**
- **Vectorizer**: Character-level n-grams (1,3) for merchant names
- **Models**: `credit_card_lgbm_model_v{N}.pkl`, etc.
- **Categories**: Merchant-level expenses (shopping, dining, etc.)

## File Naming Conventions

### **Models**
- `{source}_{algorithm}_model_v{version}.pkl`
- `{source}_tfidf_vectorizer_model_v{version}.pkl`
- `{source}_label_encoder_model_v{version}.pkl`

### **Training Data**
- `training_data_{source}_v{version}_{date}.csv`

### **Predictions**
- `processed_{filename}_v{version}_{timestamp}.csv`

## Critical Dependencies
1. **Valid Categories**: `data/valid_categories.txt` required by merge script
2. **Registry File**: `models/registry/model_registry.json` tracks all models
3. **GCS Credentials**: Required for cloud backup functionality
4. **Python Packages**: See `requirements.txt` for complete list

## Troubleshooting Dependencies
- **Model loading fails**: Run `verify_models.py` to check/restore models
- **Validation errors**: Check `data/valid_categories.txt` exists and is readable
- **Training fails**: Verify all dependencies installed and GCS credentials configured
- **Import errors**: Ensure all scripts run from project root directory