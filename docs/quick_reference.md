# Transaction Classification Pipeline: Quick Reference

## Core Commands

### Household Expenses
```bash
# 1. Transform raw data
python src/transform_monthly_household_transactions.py

# 2. Predict categories
python src/batch_predict_ensemble.py

# 3. After manual corrections, merge & retrain
python src/merge_training_data.py
python src/auto_model_ensemble.py --source household
```

### Credit Card Transactions
```bash
# 1. Predict categories
python src/batch_predict_ensemble.py

# 2. After manual corrections, merge & retrain
python src/merge_training_data.py
python src/auto_model_ensemble.py --source credit_card
```

## Output Files
| File Pattern | Purpose |
|---|---|
| `processed_{filename}_v{version}_{timestamp}.csv` | Predictions with confidence scores |
| `training_data_{source}_v{version}_{date}.csv` | Updated training data |
| `{source}_{model}_model_v{version}.pkl` | Trained models |

## Troubleshooting
| Issue | Solution |
|---|---|
| Model loading errors | `python src/verify_models.py` |
| Validation failures | Check categories match `data/valid_categories.txt` |
| Training failures | Review `logs/training.log` |
| Low confidence predictions | Focus manual corrections on these transactions |

## File Selection Tips
- **Transform script**: Select "House Kitty Transactions - Cash YYYY-MM.csv"
- **Prediction script**: Select transformed or "For Automl" files
- **Merge script**: Select manually corrected files

## Quality Indicators
- **Good confidence**: >70%
- **Review needed**: 50-70%
- **Likely wrong**: <50%