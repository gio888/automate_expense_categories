# Expense Categorization ML Pipeline

## Overview
Automated machine learning pipeline for categorizing financial transactions from household expenses and credit card statements. Supports both Excel (.xlsx/.xls) and CSV file formats with automatic format detection. Uses ensemble ML models (LightGBM, XGBoost, CatBoost) with human-in-the-loop feedback for continuous improvement.

## Key Features
- **Multi-format support**: Direct Excel (.xlsx/.xls) and CSV file processing
- **Automatic currency cleaning**: Removes currency symbols (PHP, $, etc.) from amounts
- **Source-specific models**: Separate trained models for household vs credit card transactions
- **Ensemble predictions**: Combines multiple algorithms for better accuracy
- **Intelligent file detection**: Automatic format and transaction type detection
- **Human feedback loop**: Incorporates manual corrections to continuously improve model performance
- **Cloud backup**: Automatic model backup to Google Cloud Storage
- **Version tracking**: Complete audit trail of models, training data, and performance metrics
- **Web interface**: User-friendly browser-based interface for all operations

## Project Structure
```
automate_expense_categories/
├── README.md                          # This file
├── quick_reference.md                 # Daily command reference
├── project_file_dependencies.md       # Technical dependencies
├── workflows/                         # Detailed step-by-step guides
│   ├── household_workflow.md          # Complete household expense workflow
│   └── credit_card_workflow.md        # Complete credit card workflow
├── src/                              # Core ML pipeline scripts
│   ├── transform_monthly_household_transactions.py
│   ├── batch_predict_ensemble.py      # Main prediction engine
│   ├── merge_training_data.py         # Correction integration
│   ├── auto_model_ensemble.py         # Model training
│   ├── model_registry.py             # Model version management
│   ├── model_storage.py              # Local/cloud storage handling
│   ├── transaction_types.py          # Source type definitions
│   ├── verify_models.py              # Model availability checker
│   ├── utils/                        # Utility modules
│   │   ├── excel_processor.py        # Excel file processing and standardization
│   │   └── file_detector.py          # Intelligent file format detection
│   └── web/                          # Web interface components
│       ├── app.py                    # FastAPI web application
│       └── static/                   # Frontend assets
├── data/                             # Training data and predictions
├── models/                           # Trained models and registry
├── logs/                            # Training and prediction logs
└── docs/                            # Technical documentation
```

## Quick Start

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt` (includes Excel processing support)
- Configure your personal settings (see **Configuration Setup** below)

### Configuration Setup
**IMPORTANT**: This project requires personal configuration before use.

#### Personal Configuration Architecture
The project uses a `personal/` directory to keep your private data separate from the codebase:

1. **Copy personal configuration templates:**
   ```bash
   cp -r personal.example/ personal/
   ```

2. **Configure your settings:**
   ```bash
   # Edit your GCP credentials and settings
   nano personal/config.yaml

   # Define your bank accounts and staff (if applicable)
   nano personal/accounts.yaml

   # Customize your expense categories
   nano personal/categories.yaml
   ```

3. **Set up Google Cloud Storage:**
   - Create a GCP project and enable Cloud Storage API
   - Create a service account with "Storage Admin" role
   - Download the service account key JSON file
   - Create your own GCS bucket for model backups
   - Update `personal/config.yaml` with your GCP credentials path and bucket name

4. **Generate category definitions:**
   ```bash
   python generate_categories.py
   ```

5. **Validate configuration:**
   ```bash
   python setup_validator.py
   ```

6. **Alternative: Use environment variables:**
   ```bash
   export GCP_CREDENTIALS_PATH="/path/to/your/service-account-key.json"
   export GCP_BUCKET_NAME="your-bucket-name"
   ```

**Note:** The `personal/` directory is gitignored to protect your private data.

## Usage

### Web Interface (Recommended)
The easiest way to process your transactions:

```bash
python start_web_server.py
```

Then open **http://localhost:8000** in your browser.

**Supports:**
- Excel files (.xlsx, .xls) - Direct upload from bank statements  
- CSV files - Legacy format support
- Automatic currency symbol removal (PHP, $, etc.)
- Interactive correction interface with searchable categories
- Multiple download formats (CSV, accounting system format)
- Automatic model retraining with corrections

### Command Line Interface

#### Household Expenses (5-step workflow)
```bash
cd $PROJECT_ROOT

# 1. Transform monthly Google Sheets export
python src/transform_monthly_household_transactions.py

# 2. Predict categories using ensemble models
python src/batch_predict_ensemble.py

# 3. [Manual step] Review and correct predictions

# 4. Merge corrections with training data
python src/merge_training_data.py

# 5. Retrain models with new data
python src/auto_model_ensemble.py --source household
```

#### Credit Card Transactions (4-step workflow)
```bash
cd $PROJECT_ROOT

# 1. Predict categories using ensemble models
# Supports Excel (.xlsx, .xls) and CSV files - automatic format detection
python src/batch_predict_ensemble.py --file "Statement UNIONBANK Visa 2025-08.xlsx"
# OR
python src/batch_predict_ensemble.py --file "Statement UNIONBANK Visa 2025-08.csv"

# 2. [Manual step] Review and correct predictions

# 3. Merge corrections with training data
python src/merge_training_data.py

# 4. Retrain models with new data
python src/auto_model_ensemble.py --source credit_card
```

## How It Works

### 1. **Data Processing**
- **Household**: Google Sheets exports (Excel/CSV) → Transform → Predict
- **Credit Card**: Bank statements (Excel/CSV) → Automatic format detection → Predict directly
- **Automatic cleaning**: Currency symbols (PHP, $, etc.) removed automatically

### 2. **Machine Learning**
- **Text Vectorization**: TF-IDF with source-specific optimizations
- **Ensemble Models**: LightGBM + XGBoost + CatBoost per transaction source
- **Prediction**: Majority voting with confidence scores

### 3. **Continuous Improvement**
- Manual correction of predictions
- Automatic integration of corrections into training data
- Retraining with accumulated feedback
- Version tracking of all models and data

### 4. **Model Management**
- Local storage with cloud backup
- Version registry with performance metrics
- Automatic model loading with fallback strategies

## File Types & Naming Conventions

### Standardized Naming Format
The system uses a standardized naming convention for all generated files:

**Format**: `{source}_{period}_{stage}_{timestamp}.csv`

Where:
- **source**: `unionbank_visa`, `household_cash`, `bpi_mastercard` (standardized, no spaces)
- **period**: `YYYY-MM` (statement period, not processing date)
- **stage**: `input`, `predictions`, `corrected`, `accounting`, `training`
- **timestamp**: `YYYYMMDD_HHMMSS` (when processed, for uniqueness)

### Supported Input Files

#### Excel Files (Recommended)
- **Credit Card**: `"Statement UNIONBANK Visa 8214 YYYY-MM.xlsx"` (direct from bank)
- **Household**: Excel exports from accounting systems
- Automatic currency symbol removal (PHP, $, etc.)
- Direct processing without manual conversion

#### CSV Files (Legacy Format - Still Supported)
- **Household**: `"House Kitty Transactions - Cash YYYY-MM.csv"`
- **Credit Card**: `"For Automl Statement UNIONBANK Visa YYYY-MM.csv"`
- Automatically converted to standardized format

### Output Files (Standardized Format)

**Examples:**
- **Predictions**: `unionbank_visa_2025-07_predictions_20250809_101640.csv`
- **Corrected**: `unionbank_visa_2025-07_corrected_20250809_101640.csv`
- **Accounting**: `unionbank_visa_2025-07_accounting_20250809_101640.csv`
- **Training Data**: `training_data_credit_card_v{version}_{timestamp}.csv`
- **Models**: `{source}_{algorithm}_model_v{version}.pkl` (unchanged)

**Benefits:**
- Purpose is clear from filename
- Source and statement period preserved
- Processing time tracked with timestamps
- Chronological sorting works naturally
- No spaces or special characters

## Documentation

### Daily Usage
- **[Quick Reference](docs/quick_reference.md)**: Essential commands
- **[Household Workflow](workflows/household_workflow.md)**: Complete step-by-step guide
- **[Credit Card Workflow](workflows/credit_card_workflow.md)**: Complete step-by-step guide

### Technical Details
- **[File Dependencies](docs/project_file_dependencies.md)**: Technical architecture  
- **[Output Files](docs/ML_Pipeline_Output_Files.md)**: File reference

## Troubleshooting

### Common Issues
| Problem | Solution |
|---------|----------|
| "Models not found" error | Run `python src/verify_models.py` |
| Validation failures | Check categories match `data/valid_categories.txt` |
| Training failures | Review `logs/training.log` for details |
| Low prediction confidence | Focus manual corrections on these transactions |

### Performance Indicators
- **High confidence**: >70% (usually correct)
- **Medium confidence**: 50-70% (review recommended)
- **Low confidence**: <50% (likely needs correction)

## System Requirements
- **Storage**: ~500MB for models and training data
- **Memory**: 4GB RAM minimum for training
- **Processing**: Training takes 15-20 minutes per source
- **Network**: Required for Google Cloud Storage backup

## Data Sources
- **Household Expenses**: Monthly Google Sheets exports
- **Credit Card**: UnionBank statement CSV downloads
- **Categories**: Maintained in `data/valid_categories.txt`

---

**For detailed workflows and step-by-step instructions, see the `workflows/` directory.**