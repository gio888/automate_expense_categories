# Transaction Classification Pipeline: Quick Reference

This guide provides step-by-step instructions for running the transaction classification pipeline.

## 1. Batch Prediction

Run the batch prediction script:
```bash
python src/batch_predict_ensemble.py
```

The script will:
1. Show available CSV files in the data directory
2. Prompt you to select a file to process
3. Load appropriate models based on transaction source
4. Generate predictions with confidence scores
5. Save results to data directory as `processed_[filename]_v[version]_[timestamp].csv`

### Common Issues:
- If you get model loading errors, run `python src/verify_models.py` to check model availability
- Low confidence predictions (< 0.5) should be reviewed carefully

## 2. Manual Correction

1. Open the processed file generated in the previous step
2. Review transactions and correct any misclassified categories
3. Save with prefix "corrected_" (e.g., `corrected_Cash_2024-10.csv`)

### Best Practices:
- Focus first on transactions with low confidence scores
- Ensure the corrected file maintains the same structure as the processed file
- Verify that all categories are valid according to your category system

## 3. Merge Corrections

Run the merge script:
```bash
python src/merge_training_data.py
```

The script will:
1. Show available CSV files
2. Prompt you to select the corrections file
3. Validate corrections and merge with existing training data
4. Create a new versioned training dataset (`training_data_[source]_v[version]_[date].csv`)

### Common Issues:
- If validation fails, check for invalid categories or data format issues
- Ensure the transaction_source column is correctly set in your data

## 4. Retrain Models

Run the training script with the appropriate source parameter:
```bash
python src/auto_model_ensemble.py --source household
# OR
python src/auto_model_ensemble.py --source credit_card
```

The script will:
1. Load the latest training data for the specified source
2. Train ensemble models (LightGBM, XGBoost, CatBoost)
3. Save and register models with the next version number

### Notes:
- Training can take several minutes depending on dataset size
- Check the logs directory for detailed training results
- New models will automatically be used in future batch predictions