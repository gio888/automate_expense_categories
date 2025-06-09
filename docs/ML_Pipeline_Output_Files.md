# ML Pipeline Output Files Reference

## Prediction Files
| Filename Pattern | Script | Description | Location |
|---|---|---|---|
| `processed_{input_basename}_v{version}_{timestamp}.csv` | `src/batch_predict_ensemble.py` | Transactions with predicted categories and confidence scores | `data/` |

**Example**: `processed_For Automl Statement UNIONBANK Visa 2025-05_v1.0_20250608_190731.csv`

## Training Data Files
| Filename Pattern | Script | Description | Location |
|---|---|---|---|
| `training_data_{source}_v{version}_{date}.csv` | `src/merge_training_data.py` | Source-specific training data with merged corrections | `data/` |

**Examples**: 
- `training_data_household_v6_20250503.csv`
- `training_data_credit_card_v4_20250609.csv`

## Model Files
| Filename Pattern | Script | Description | Location |
|---|---|---|---|
| `{source}_{algorithm}_model_v{version}.pkl` | `src/auto_model_ensemble.py` | Trained ML model for specific source and algorithm | `models/` |
| `{source}_tfidf_vectorizer_model_v{version}.pkl` | `src/auto_model_ensemble.py` | TF-IDF vectorizer for specific transaction source | `models/` |
| `{source}_label_encoder_model_v{version}.pkl` | `src/auto_model_ensemble.py` | Label encoder for mapping categories to numbers | `models/` |

**Examples**:
- `household_lgbm_model_v2.pkl`
- `credit_card_xgboost_model_v4.pkl`
- `household_tfidf_vectorizer_model_v2.pkl`

## Log Files
| Filename Pattern | Script | Description | Location |
|---|---|---|---|
| `batch_predict_{datetime}.log` | `src/batch_predict_ensemble.py` | Logs from batch prediction process | `logs/` |
| `correction_validation_{datetime}.log` | `src/merge_training_data.py` | Logs from correction validation process | `logs/` |
| `training.log` | `src/auto_model_ensemble.py` | Logs from model training process | `logs/` |
| `classification_report_{model_name}.txt` | `src/auto_model_ensemble.py` | Classification performance metrics for each model | `logs/` |
| `shap_summary_{model_name}.png` | `src/auto_model_ensemble.py` | SHAP explanations visualization (if enabled) | `logs/` |
| `prediction_metrics.jsonl` | `src/batch_predict_ensemble.py` | JSON lines with prediction quality metrics | `logs/` |

## Registry & Configuration Files
| Filename Pattern | Script | Description | Location |
|---|---|---|---|
| `model_registry.json` | `src/model_registry.py` | Registry of all model versions and metadata | `models/registry/` |
| `valid_categories.txt` | Manual | List of approved expense categories | `data/` |

## Transformed Data Files
| Filename Pattern | Script | Description | Location |
|---|---|---|---|
| `House Kitty Transactions - Cash - Corrected {YYYY-MM}.csv` | `src/transform_monthly_household_transactions.py` | Household data converted to ML format | Google Drive folder |

## File Naming Key
- `{source}`: `household` or `credit_card`
- `{algorithm}`: `lgbm`, `xgboost`, or `catboost`
- `{version}`: Incremental version number (1, 2, 3, ...)
- `{timestamp}`: Format `YYYYMMDD_HHMMSS`
- `{date}`: Format `YYYYMMDD`
- `{datetime}`: Format `YYYYMMDD_HHMMSS`

## File Lifecycle
```
Raw Data → Transform (household only) → Predict → Manual Corrections → 
Merge → Train → New Models → Registry Update
```

## Cleanup Guidelines
### Safe to Delete
- Prediction files older than 30 days
- Log files older than 90 days
- Old model versions (keep latest 2-3 versions)

### Keep Permanently  
- `model_registry.json`
- `valid_categories.txt`
- Latest training data files
- Latest model files

### Archive Periodically
- Old training data versions
- Historical prediction files
- Old model versions (for rollback capability)