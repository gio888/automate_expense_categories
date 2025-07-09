# Expense Categorization ML Pipeline

## Overview
Automated machine learning pipeline for categorizing financial transactions from household expenses and credit card statements. Uses ensemble ML models (LightGBM, XGBoost, CatBoost) with human-in-the-loop feedback for continuous improvement.

## Key Features
- **Source-specific models**: Separate trained models for household vs credit card transactions
- **Ensemble predictions**: Combines multiple algorithms for better accuracy
- **Human feedback loop**: Incorporates manual corrections to continuously improve model performance
- **Cloud backup**: Automatic model backup to Google Cloud Storage
- **Version tracking**: Complete audit trail of models, training data, and performance metrics

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
│   └── verify_models.py              # Model availability checker
├── data/                             # Training data and predictions
├── models/                           # Trained models and registry
├── logs/                            # Training and prediction logs
└── docs/                            # Technical documentation
```

## Quick Start

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`
- Ensure `data/valid_categories.txt` exists with approved expense categories
- Google Cloud Storage credentials configured (for model backup)

### Household Expenses (5-step workflow)
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

### Credit Card Transactions (4-step workflow)
```bash
cd $PROJECT_ROOT

# 1. Predict categories using ensemble models
python src/batch_predict_ensemble.py

# 2. [Manual step] Review and correct predictions

# 3. Merge corrections with training data
python src/merge_training_data.py

# 4. Retrain models with new data
python src/auto_model_ensemble.py --source credit_card
```

## How It Works

### 1. **Data Processing**
- **Household**: Google Sheets exports → Transform → Predict
- **Credit Card**: Bank CSV exports → Predict directly

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

### Input Files
- **Household**: `"House Kitty Transactions - Cash YYYY-MM.csv"`
- **Credit Card**: `"For Automl Statement UNIONBANK Visa YYYY-MM.csv"`

### Output Files
- **Predictions**: `processed_{filename}_v{version}_{timestamp}.csv`
- **Training Data**: `training_data_{source}_v{version}_{date}.csv`
- **Models**: `{source}_{algorithm}_model_v{version}.pkl`

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