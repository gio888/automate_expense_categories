## ML Pipeline Output Files

| Filename Syntax | Script | Description | Folder |
|---|---|---|---|
| `processed_{input_basename}_v{predictor.version}_{timestamp}.csv` | batch_predict_ensemble.py | Transactions with predicted categories and confidence scores | data/ |
| `training_data_v{new_version}_{timestamp}.csv` | merge_training_data.py | New version of training data with merged corrections | data/ |
| `{source.value}_{model_type}_model_v{version}.pkl` | auto_model_ensemble.py | Trained ML model for specific source and algorithm | models/ |
| `{source.value}_tfidf_vectorizer_model_v{version}.pkl` | auto_model_ensemble.py | TF-IDF vectorizer for specific transaction source | models/ |
| `{source.value}_label_encoder_model_v{version}.pkl` | auto_model_ensemble.py | Label encoder for mapping categories to numbers | models/ |
| `classification_report_{model_name}.txt` | auto_model_ensemble.py | Classification performance metrics for each model | logs/ |
| `shap_summary_{model_name}.png` | auto_model_ensemble.py | SHAP explanations visualization | logs/ |
| `batch_predict_{datetime}.log` | batch_predict_ensemble.py | Logs from batch prediction process | logs/ |
| `correction_validation_{datetime}.log` | merge_training_data.py | Logs from correction validation process | logs/ |
| `training.log` | auto_model_ensemble.py | Logs from model training process | logs/ |
| `prediction_metrics.jsonl` | batch_predict_ensemble.py (PredictionMonitor) | JSON lines with prediction quality metrics | logs/ |
| `model_registry.json` | model_registry.py | Registry of all model versions and metadata | models/registry/ |