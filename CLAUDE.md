# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated expense categorization ML pipeline that uses ensemble machine learning models (LightGBM, XGBoost, CatBoost) to classify financial transactions from household expenses and credit card statements. The system implements human-in-the-loop feedback for continuous improvement.

## Architecture

### Core Components
- **Source-specific models**: Separate trained models for household vs credit card transactions (`TransactionSource` enum)
- **Ensemble prediction**: Majority voting across multiple ML algorithms with confidence scoring
- **Model versioning**: Complete audit trail via `ModelRegistry` and cloud backup via `ModelStorage`
- **Training data evolution**: Human corrections automatically integrated back into training data

### Key Files
- `src/batch_predict_ensemble.py` - Main prediction engine with ensemble voting
- `src/auto_model_ensemble.py` - Model training pipeline with FLAML AutoML
- `src/model_registry.py` - Version tracking and model metadata management
- `src/model_storage.py` - Local/cloud storage handling with Google Cloud Storage
- `src/transaction_types.py` - Transaction source definitions and configurations
- `src/merge_training_data.py` - Integrates human corrections into training data

### Data Flow
1. Raw transactions → Transform (household only) → Predict → Manual correction → Merge corrections → Retrain models
2. Models are versioned and backed up to Google Cloud Storage automatically
3. Predictions include confidence scores for human review prioritization

## Common Development Commands

### Training and Prediction
```bash
# Household expense workflow (5 steps)
python src/transform_monthly_household_transactions.py  # Transform raw Google Sheets export
python src/batch_predict_ensemble.py                   # Generate predictions
# [Manual review and correction step]
python src/merge_training_data.py                      # Integrate corrections
python src/auto_model_ensemble.py --source household   # Retrain models

# Credit card workflow (4 steps)
python src/batch_predict_ensemble.py                   # Generate predictions
# [Manual review and correction step]  
python src/merge_training_data.py                      # Integrate corrections
python src/auto_model_ensemble.py --source credit_card # Retrain models
```

### Diagnostics
```bash
python src/verify_models.py        # Check model availability and versions
python src/check_registry.py       # Validate model registry integrity
```

### Testing
```bash
python -m unittest tests/test_transaction_types.py  # Run unit tests
```

## File Naming Conventions

### Input Files
- Household: `"House Kitty Transactions - Cash YYYY-MM.csv"`
- Credit Card: `"For Automl Statement UNIONBANK Visa YYYY-MM.csv"`

### Output Files
- Predictions: `processed_{filename}_v{version}_{timestamp}.csv`
- Training Data: `training_data_{source}_v{version}_{date}.csv`
- Models: `{source}_{algorithm}_model_v{version}.pkl`

## Environment Setup

### Dependencies
Install via: `pip install -r requirements.txt`
Key ML libraries: scikit-learn, lightgbm, xgboost, catboost, flaml, shap, pyyaml

### Configuration System
The project uses a centralized configuration system that loads settings from:
1. **Environment variables** (highest priority)
2. **config.yaml file** (medium priority)
3. **Default values** (lowest priority)

### Required Setup
1. **Copy configuration template:** `cp config.example.yaml config.yaml`
2. **Set up GCP credentials:** Create service account and download JSON key
3. **Configure settings:** Edit `config.yaml` or use environment variables
4. **Validate setup:** Run `python src/config.py` to test configuration
5. **Required data files:** Ensure `data/valid_categories.txt` exists

### Environment Variables
Key environment variables for configuration:
- `GCP_CREDENTIALS_PATH` - Path to service account JSON key
- `GCP_BUCKET_NAME` - GCS bucket for model backups  
- `GCP_PROJECT_ID` - Optional GCP project ID
- `DATA_DIR`, `MODELS_DIR`, `LOGS_DIR` - Override default paths

## Model Management

### Training Process
- Uses FLAML AutoML for hyperparameter optimization
- Trains separate ensemble models per transaction source
- Automatic version incrementing and cloud backup
- Performance metrics logged to `logs/training.log`

### Prediction Confidence
- High (>70%): Usually correct
- Medium (50-70%): Review recommended  
- Low (<50%): Likely needs correction

## Important Notes

- Models are source-specific (household vs credit_card) with different TF-IDF configurations
- Training requires minimum samples per category (5 for household, 3 for credit card)
- **Configuration is required before use** - copy `config.example.yaml` to `config.yaml` and customize
- All file paths are now configurable via the configuration system
- Cloud backup requires valid GCP credentials and proper configuration
- Manual correction step is critical for model improvement
- Use `python src/config.py` to validate your configuration setup