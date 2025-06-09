# ML Pipeline Documentation: Transaction Category Classification

## Overview
This document outlines the machine learning pipeline for automatically classifying financial transactions into categories. The pipeline implements an iterative learning process with human feedback incorporation.

## Pipeline Components

### 1. Data Management
#### Training Data Structure
- Primary features: Transaction Description (text)
- Target: Transaction Category
- Additional fields stored but not used in training: Date, Amount, Amount (Negated), Amount (Raw), Entered, Reconciled
- Data versioning format: `training_data_v{version_number}.csv`

#### Data Validation
```python
def validate_training_data(df):
    """
    Validates training data before processing
    Returns: (bool, list of validation messages)
    """
    validations = []
    
    # Check for required columns
    required_columns = ["Description", "Category"]
    if not all(col in df.columns for col in required_columns):
        validations.append("Missing required columns")
    
    # Check for empty descriptions
    empty_descriptions = df["Description"].isna().sum()
    if empty_descriptions > 0:
        validations.append(f"Found {empty_descriptions} empty descriptions")
    
    # Check for empty categories
    empty_categories = df["Category"].isna().sum()
    if empty_categories > 0:
        validations.append(f"Found {empty_categories} empty categories")
    
    # Category distribution check
    category_dist = df["Category"].value_counts(normalize=True)
    rare_categories = category_dist[category_dist < 0.01]
    if not rare_categories.empty:
        validations.append(f"Found {len(rare_categories)} categories with <1% representation")
    
    return len(validations) == 0, validations
```

### 2. Model Training Pipeline
#### Pre-processing
1. Text Vectorization
   - TF-IDF vectorization
   - n-gram range: (1, 2)
   - max_features: 1000
   - min_df: 2
   - max_df: 0.95

2. Data Splitting
   - 80% training, 20% testing
   - Stratified split based on category distribution
   - Random seed for reproducibility

#### Model Training
- Ensemble of models using FLAML AutoML
- Models: LightGBM, XGBoost, CatBoost
- Metric optimization: macro F1-score
- Time budget: 300 seconds per model
- Cross-validation: 5 folds

#### Model Artifacts
Storage structure in `models/` directory:
```
models/
├── lgbm_model_v{version}.pkl
├── xgboost_model_v{version}.pkl
├── catboost_model_v{version}.pkl
└── tfidf_vectorizer_v{version}.pkl
```

### 3. Logging System
#### Training Logs
Location: `logs/training_logs.csv`
Fields:
- version: Model version
- training_data: Training data file name
- models: List of model files
- macro_f1: Average macro F1 score
- roc_auc: Average ROC AUC score
- pr_auc: Average PR AUC score
- timestamp: Training timestamp

#### Detailed Logs
Location: `logs/training.log`
Contents:
- Data validation results
- Training progress
- Model performance metrics
- Error tracking

### 4. Batch Prediction System
```python
def batch_predict(descriptions, model_version):
    """
    Batch prediction with confidence scores
    Returns: DataFrame with predictions and confidence scores
    """
    results = {
        'description': descriptions,
        'predicted_category': [],
        'confidence_score': [],
        'model_version': model_version
    }
    
    # Load models and vectorizer
    models = load_ensemble_models(model_version)
    vectorizer = load_vectorizer(model_version)
    
    # Transform text
    X = vectorizer.transform(descriptions)
    
    # Get predictions and confidence scores
    predictions = []
    confidences = []
    for model in models:
        pred = model.predict(X)
        prob = model.predict_proba(X)
        predictions.append(pred)
        confidences.append(np.max(prob, axis=1))
    
    # Ensemble voting and confidence averaging
    final_predictions = vote_predictions(predictions)
    final_confidences = np.mean(confidences, axis=0)
    
    results['predicted_category'] = final_predictions
    results['confidence_score'] = final_confidences
    
    return pd.DataFrame(results)
```

### 5. Feedback and Correction System
#### Correction Interface Requirements
- Display transaction details and predicted category
- Allow quick category corrections
- Track confidence scores
- Flag low-confidence predictions for review
- Export corrections in standard format

#### Correction Data Structure
```python
correction_schema = {
    'description': str,
    'predicted_category': str,
    'corrected_category': str,
    'confidence_score': float,
    'model_version': str,
    'timestamp': datetime,
    'correction_notes': str  # Optional
}
```

### 6. Training Data Update Process
1. Merge corrections with existing training data
2. Validate merged dataset
3. Version new training dataset
4. Archive previous version
5. Update data registry

### 7. Performance Monitoring
#### Metrics to Track
- Overall accuracy
- Per-category F1 scores
- Confusion matrix
- Prediction confidence distribution
- Correction rate
- Category distribution drift

#### Monitoring Dashboard Requirements
- Historical performance trends
- Category-wise performance
- Data drift indicators
- Correction patterns
- Model version comparison

## Pipeline Execution Flow
1. Data Preparation & Validation
2. Model Training & Evaluation
3. Batch Prediction
4. Human Review & Correction
5. Feedback Integration
6. Performance Monitoring
7. Retraining Decision

## Best Practices
1. Version Control
   - All code in version control
   - Model versions tracked
   - Training data versions tracked

2. Documentation
   - Model cards for each version
   - Training configurations
   - Data preprocessing steps
   - Performance reports

3. Testing
   - Unit tests for preprocessing
   - Integration tests for pipeline
   - Data validation tests
   - Model sanity checks

4. Monitoring
   - Regular performance checks
   - Data drift monitoring
   - Error pattern analysis
   - Resource usage tracking

## Future Improvements
1. Feature Engineering
   - Amount-based features
   - Temporal features
   - Merchant name extraction

2. Model Enhancements
   - Neural network integration
   - Active learning implementation
   - Uncertainty estimation

3. Pipeline Optimization
   - Automated retraining triggers
   - Enhanced data validation
   - Improved confidence estimation