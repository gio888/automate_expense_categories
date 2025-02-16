# Multi-Source Transaction Classification: Implementation Plan

## Technical Context

### Current System Architecture
- Implemented ML pipeline using FLAML AutoML for expense categorization
- Uses ensemble approach (LightGBM, XGBoost, CatBoost)
- Text processing optimized for household descriptions (groceries, supplies)
- Includes model versioning, registry system, and GCS backup
- Active learning system for continuous model improvement
- Current text vectorization tuned for item-level descriptions

### Technical Challenges
- Different text patterns in credit card transactions (merchant names vs items)
- Current TF-IDF vectorization may not be optimal for merchant names
- Need to maintain single category taxonomy across sources
- Model performance metrics need source-specific tracking
- Risk of code duplication if creating separate pipelines
- Must preserve existing feedback and retraining mechanisms

### Implementation Approach
- Extend existing codebase with source-specific processing
- Add configurable text vectorization per source type
- Maintain unified category system in database
- Implement source-aware model training and evaluation
- Create flexible configuration system for source settings
- Design for modularity to support future source types
- Keep single feedback loop with source context

## Overview
This document outlines the steps to extend the current ML pipeline to handle multiple transaction sources (household and credit card expenses) while maintaining a unified category system and feedback loop. The implementation leverages existing infrastructure while adding source-specific optimizations to handle different transaction description patterns effectively.

## Implementation Steps

### 1. Transaction Source Management
```python
# Create new file: src/transaction_types.py

from enum import Enum

class TransactionSource(Enum):
    HOUSEHOLD = 'household'
    CREDIT_CARD = 'credit_card'
    
    @classmethod
    def get_source_display_name(cls, source):
        return {
            cls.HOUSEHOLD: 'Household Expenses',
            cls.CREDIT_CARD: 'Credit Card Transactions'
        }.get(source, source.value)
```

### 2. Enhanced Text Vectorization
```python
# Update src/auto_model_ensemble.py

def create_vectorizer(transaction_source: TransactionSource):
    """Create source-specific vectorizer configurations"""
    base_config = {
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.95
    }
    
    if transaction_source == TransactionSource.CREDIT_CARD:
        return TfidfVectorizer(
            **base_config,
            ngram_range=(1, 3),  # Capture longer vendor names
            analyzer='char_wb',   # Better for business names
            strip_accents='unicode'
        )
    else:
        return TfidfVectorizer(
            **base_config,
            ngram_range=(1, 2),
            analyzer='word'
        )
```

### 3. Modified Model Training Pipeline
```python
# Update src/auto_model_ensemble.py

def train_models(df: pd.DataFrame, 
                training_file: str, 
                compute_shap: bool = True, 
                min_category_samples: int = 5):
    """Train models for each transaction source"""
    
    # Group data by source
    sources = df['transaction_source'].unique()
    
    models = {}
    vectorizers = {}
    metrics = {}
    
    for source in sources:
        logger.info(f"\nTraining models for {TransactionSource.get_source_display_name(source)}")
        source_data = df[df['transaction_source'] == source]
        
        # Create source-specific vectorizer
        vectorizer = create_vectorizer(source)
        vectorizers[source] = vectorizer
        
        # Train source-specific models
        X = source_data["Description"].fillna("")
        y = source_data["Category"]
        
        source_models, source_metrics = train_source_models(
            X, y, vectorizer, min_category_samples
        )
        
        models[source] = source_models
        metrics[source] = source_metrics
        
        # Save source-specific artifacts
        save_source_models(source, source_models, vectorizer, source_metrics, training_file)
    
    return models, vectorizers, metrics
```

### 4. Updated Prediction System
```python
# Update src/batch_predict_ensemble.py

class BatchPredictor:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.models = {}
        self.vectorizers = {}
        self.load_latest_models()
    
    def load_latest_models(self):
        """Load latest models for each source"""
        for source in TransactionSource:
            try:
                self.load_source_models(source)
            except Exception as e:
                logger.error(f"Failed to load models for {source.value}: {str(e)}")
    
    def predict_batch(self, 
                     descriptions: List[str], 
                     source: TransactionSource) -> Tuple[List[str], List[float]]:
        """Process predictions using source-specific models"""
        if source not in self.vectorizers:
            raise ValueError(f"No models loaded for source: {source.value}")
            
        vectorizer = self.vectorizers[source]
        models = self.models[source]
        
        features = vectorizer.transform(descriptions)
        predictions = []
        confidences = []
        
        for model in models:
            pred = model.predict(features)
            prob = model.predict_proba(features)
            predictions.append(pred)
            confidences.append(np.max(prob, axis=1))
        
        return self._combine_predictions(predictions, confidences)
```

### 5. Extended Model Registry
```python
# Update src/model_registry.py

class ModelRegistry:
    def register_model(self, 
                      model_name: str, 
                      version: int, 
                      source: TransactionSource,
                      locations: Dict[str, str], 
                      metrics: Dict[str, float],
                      training_data: str,
                      additional_metadata: Optional[Dict] = None) -> None:
        """Register a new model version with source information"""
        
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {}
        
        version_entry = {
            'version': version,
            'source': source.value,
            'locations': locations,
            'metrics': metrics,
            'training_data': training_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }
        
        if additional_metadata:
            version_entry.update(additional_metadata)
        
        self.registry['models'][model_name][str(version)] = version_entry
        self._save_registry()
        logging.info(f"✅ Registered {model_name} v{version} for {source.value}")
```

### 6. Configuration Management
```python
# Create new file: src/config.py

from src.transaction_types import TransactionSource

class Config:
    SOURCES = {
        TransactionSource.HOUSEHOLD: {
            'min_samples_per_category': 5,
            'confidence_threshold': 0.6,
            'vectorizer_config': {
                'ngram_range': (1, 2),
                'analyzer': 'word'
            }
        },
        TransactionSource.CREDIT_CARD: {
            'min_samples_per_category': 3,
            'confidence_threshold': 0.7,
            'vectorizer_config': {
                'ngram_range': (1, 3),
                'analyzer': 'char_wb'
            }
        }
    }
    
    @classmethod
    def get_source_config(cls, source: TransactionSource) -> dict:
        """Get configuration for specific source"""
        return cls.SOURCES.get(source, cls.SOURCES[TransactionSource.HOUSEHOLD])
```

## Implementation Order

1. **Phase 1: Basic Source Management**
   - Create TransactionSource enum
   - Add source field to training data
   - Update data loading and validation

2. **Phase 2: Source-Specific Vectorization**
   - Implement create_vectorizer
   - Update training pipeline for source handling
   - Test with sample data from both sources

3. **Phase 3: Model Registry Updates**
   - Extend registry with source tracking
   - Update model versioning
   - Test model saving and loading

4. **Phase 4: Prediction System**
   - Update BatchPredictor
   - Add source-specific prediction
   - Test prediction flow

5. **Phase 5: Configuration System**
   - Implement Config class
   - Add source-specific settings
   - Test configuration management

6. **Phase 6: Testing and Validation**
   - Test complete pipeline
   - Validate predictions
   - Compare performance across sources

## Notes

- Keep unified category system
- Maintain single feedback/correction pipeline
- Each phase should include appropriate unit tests
- Update documentation as changes are implemented
- Monitor performance metrics for each source separately