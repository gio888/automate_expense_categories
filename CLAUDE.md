# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated expense categorization ML pipeline that uses ensemble machine learning models (LightGBM, XGBoost, CatBoost) to classify financial transactions from household expenses and credit card statements. The system supports both CSV and Excel file formats and implements human-in-the-loop feedback for continuous improvement.

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
- `src/utils/excel_processor.py` - Excel file processing and standardization
- `src/utils/file_detector.py` - Intelligent file detection for CSV and Excel formats

### Data Flow
1. **File Processing**: Excel/CSV files → Automatic format detection → Standardized data format
2. **ML Pipeline**: Raw transactions → Transform (household only) → Predict → Manual correction → Merge corrections → Retrain models
3. **Storage & Versioning**: Models are versioned and backed up to Google Cloud Storage automatically
4. **Quality Control**: Predictions include confidence scores for human review prioritization

## Common Development Commands

### Web Interface (Primary)
```bash
# Start the web interface (recommended for users)
python start_web_server.py
# Then open http://localhost:8000 in browser
```

**Supported File Formats:**
- **Excel files**: `.xlsx`, `.xls` (Direct upload from bank statements)
- **CSV files**: Legacy format support with full backward compatibility

### Command Line Interface (Development)

#### Training and Prediction
```bash
# Household expense workflow (5 steps)
python src/transform_monthly_household_transactions.py  # Transform raw Google Sheets export
python src/batch_predict_ensemble.py                   # Generate predictions
# [Manual review and correction step]
python src/merge_training_data.py                      # Integrate corrections
python src/auto_model_ensemble.py --source household   # Retrain models

# Credit card workflow (4 steps) - now supports Excel files directly
python src/batch_predict_ensemble.py --file statement.xlsx  # Generate predictions from Excel
# [Manual review and correction step]  
python src/merge_training_data.py                           # Integrate corrections
python src/auto_model_ensemble.py --source credit_card      # Retrain models
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

The system uses a standardized naming convention: `{source}_{period}_{stage}_{timestamp}.csv`

### Components
- **Source**: `unionbank_visa`, `household_cash`, `bpi_mastercard` (standardized, no spaces)
- **Period**: `YYYY-MM` (statement period, not processing date)  
- **Stage**: `input`, `predictions`, `corrected`, `accounting`, `training`
- **Timestamp**: `YYYYMMDD_HHMMSS` (when processed, for uniqueness)

### Input Files - Multiple Formats Supported
**Excel Files (Recommended):**
- Credit Card: `"Statement UNIONBANK Visa 8214 YYYY-MM.xlsx"`
- Household: Excel exports from accounting systems

**CSV Files (Legacy Format - Auto-Converted):**
- Household: `"House Kitty Transactions - Cash YYYY-MM.csv"`
- Credit Card: `"For Automl Statement UNIONBANK Visa YYYY-MM.csv"`

### Output Files (New Standardized Format)
- **Predictions**: `unionbank_visa_2025-07_predictions_20250808_194140.csv`
- **Corrected**: `unionbank_visa_2025-07_corrected_20250808_194140.csv`
- **Accounting**: `unionbank_visa_2025-07_accounting_20250808_194140.csv` 
- **Training Data**: `unionbank_visa_2025-07_training_20250808_194140.csv`
- **Models**: `{source}_{algorithm}_model_v{version}.pkl` (unchanged)

### Benefits
✅ **Purpose clear** from filename  
✅ **Source preserved** (Union Bank Visa)  
✅ **Statement period preserved** (2025-07)  
✅ **Processing time tracked** (timestamps)  
✅ **Chronological sorting** works naturally  
✅ **No spaces or special characters**

## Environment Setup

### Dependencies
Install via: `pip install -r requirements.txt`
Key libraries:
- **ML**: scikit-learn, lightgbm, xgboost, catboost, flaml, shap
- **Data**: pandas, numpy, pyyaml
- **Excel Support**: openpyxl (for .xlsx/.xls file processing)
- **Web**: fastapi, uvicorn (for web interface)

### Personal Configuration System
The project uses a **personal data architecture** that separates your private data from the public codebase:

**Configuration Structure:**
```
personal/              # Your private data (ignored by git)
├── config.yaml       # Your GCP credentials and settings
├── accounts.yaml     # Your bank accounts, staff, loans
└── categories.yaml   # Your expense category preferences

personal.example/     # Templates for new users (tracked by git)
├── config.yaml       # Template configuration
├── accounts.yaml     # Example account definitions
└── categories.yaml   # Example category structure
```

### Required Setup
1. **Copy personal configuration templates:**
   ```bash
   cp -r personal.example/ personal/
   ```

2. **Configure your personal settings:**
   - Edit `personal/config.yaml` with your GCP credentials
   - Edit `personal/accounts.yaml` with your bank accounts and staff
   - Edit `personal/categories.yaml` with your expense preferences

3. **Generate your category definitions:**
   ```bash
   python generate_categories.py
   ```

4. **Validate setup:** Run `python src/config.py` to test configuration

5. **Test category generation:** Run `python generate_categories.py` to create `data/valid_categories.txt`

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

## Usage Patterns

### For End Users
- **Primary interface**: Web interface (`python start_web_server.py`)
- **Complete workflow**: Upload → Process → Correct → Retrain (all in browser)
- **File formats**: Drag & drop Excel (.xlsx, .xls) and CSV files, automatic format detection
- **Real-time feedback**: Progress bars, confidence scores, searchable corrections
- **Direct bank statement upload**: No need for manual Excel to CSV conversion

### For Developers/Automation
- **CLI commands**: Use the batch processing scripts for automation (supports both Excel and CSV)
- **Model development**: Direct access to training and validation scripts
- **Integration**: FastAPI endpoints available for custom integrations
- **File processing**: Automatic Excel → standardized format conversion in background

## Important Notes

- **Web interface dependencies**: Requires `fastapi>=0.104.0`, `uvicorn>=0.24.0`, and `openpyxl>=3.1.0`
- Models are source-specific (household vs credit_card) with different TF-IDF configurations
- Training requires minimum samples per category (5 for household, 3 for credit card)
- **Configuration is required before use** - copy `config.example.yaml` to `config.yaml` and customize
- All file paths are now configurable via the configuration system
- Cloud backup requires valid GCP credentials and proper configuration
- Manual correction step is critical for model improvement
- Use `python src/config.py` to validate your configuration setup