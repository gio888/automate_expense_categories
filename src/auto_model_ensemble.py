# First, only the essential imports for path setup
import os
import sys
from pathlib import Path

# Setup the path BEFORE any other imports
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
import logging
import shap
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, roc_auc_score, average_precision_score
from flaml import AutoML
from google.cloud import storage

# Import our custom classes
from src.model_storage import ModelStorage
from src.model_registry import ModelRegistry

# ✅ Define Paths and Configuration
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REGISTRY_DIR = MODELS_DIR / "registry"

# GCS Configuration
BUCKET_NAME = "expense-categorization-ml-models-backup"
CREDENTIALS_PATH = "/Users/gio/.gcp/gnucash-predict-f9187964a5aa.json"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)

# ✅ Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize storage and registry
model_storage = ModelStorage(BUCKET_NAME, CREDENTIALS_PATH, MODELS_DIR)
model_registry = ModelRegistry(REGISTRY_DIR, BUCKET_NAME, CREDENTIALS_PATH)

def save_model(model, name: str, metrics: dict, training_data: str) -> dict:
    """
    Save model using ModelStorage and register it
    
    Args:
        model: Trained model to save
        name: Model name
        metrics: Performance metrics
        training_data: Training data file used
    
    Returns:
        dict: Model save locations
    """
    try:
        # Get next version
        version = model_registry.get_latest_version(name)
        version = (version or 0) + 1
        
        # Save model using storage class
        locations = model_storage.save_model(model, name, version)
        
        # Register the model
        model_registry.register_model(
            model_name=name,
            version=version,
            locations=locations,
            metrics=metrics,
            training_data=training_data,
            additional_metadata={
                'framework': 'flaml',
                'python_version': '3.9'  # You might want to get this dynamically
            }
        )
        
        logger.info(f"✅ Model {name} v{version} saved and registered successfully")
        return locations
        
    except Exception as e:
        logger.error(f"❌ Error saving {name} model: {str(e)}")
        # Fallback to local save
        local_path = MODELS_DIR / f"{name}_model_fallback.pkl"
        joblib.dump(model, local_path)
        logger.info(f"✅ Model saved locally as fallback: {local_path}")
        return {'local_path': str(local_path)}

def load_data():
    """Load the latest training data file"""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith("training_data_")]
        if not files:
            raise FileNotFoundError("No training data files found")
        
        latest_file = sorted(files)[-1]
        logger.info(f"📂 Loading training data: {latest_file}")
        df = pd.read_csv(DATA_DIR / latest_file)
        
        return df, latest_file
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, compute_extra_metrics=True):
    """Evaluate model performance with various metrics"""
    metrics = {
        'macro_f1': f1_score(y_test, model.predict(X_test), average='macro')
    }
    
    if compute_extra_metrics:
        try:
            y_pred_proba = model.predict_proba(X_test)
            y_test_dummies = pd.get_dummies(y_test)
            
            metrics.update({
                'roc_auc': roc_auc_score(y_test_dummies, y_pred_proba, multi_class='ovr'),
                'pr_auc': average_precision_score(y_test_dummies, y_pred_proba)
            })
        except Exception as e:
            logger.warning(f"Could not compute probability-based metrics: {str(e)}")
    
    return metrics

def train_models(df, training_file, compute_shap=True, min_category_samples=5):
    """Train multiple models using FLAML AutoML"""
    # [Previous data preparation code remains the same...]
    
    models = {}
    model_metrics = {}
    training_errors = {}
    
    for model_name in ["lgbm", "xgboost", "catboost"]:
        try:
            logger.info(f"\n{'='*30}\n🚀 Training {model_name} model...\n{'='*30}")
            
            # Log training parameters
            logger.info(f"Training parameters for {model_name}:")
            logger.info(f"- Features: {X_train_tfidf.shape[1]} columns")
            logger.info(f"- Training samples: {X_train_tfidf.shape[0]}")
            logger.info(f"- Unique categories: {len(np.unique(y_train))}")
            
            automl = AutoML()
            automl.fit(
                X_train_tfidf,
                y_train,
                task="classification",
                metric="macro_f1",
                time_budget=300,
                estimator_list=[model_name],
                verbose=2
            )
            
            models[model_name] = automl
            
            # Evaluate and log detailed metrics
            metrics = evaluate_model(automl, X_test_tfidf, y_test, compute_extra_metrics=True)
            model_metrics[model_name] = metrics
            
            logger.info(f"\n📊 {model_name} Performance Metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"- {metric_name}: {value:.4f}")
            
            # Save and register model with detailed metadata
            save_model(
                model=automl,
                name=model_name,
                metrics=metrics,
                training_data=training_file
            )
            
            logger.info(f"✅ {model_name} training completed successfully")
            
        except Exception as e:
            error_msg = f"❌ Error training {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.error("Stack trace:", exc_info=True)
            
            # Store error details
            training_errors[model_name] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log error to separate file
            error_log_path = os.path.join(LOGS_DIR, f"training_error_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            with open(error_log_path, 'w') as f:
                f.write(f"Error Type: {type(e).__name__}\n")
                f.write(f"Error Message: {str(e)}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                import traceback
                f.write("\nStack Trace:\n")
                traceback.print_exc(file=f)
            
            # Consider failing training if critical models fail
            if model_name in ['lgbm', 'xgboost']:  # Critical models
                raise Exception(f"Critical model {model_name} failed: {str(e)}")
            continue
    
    # Log final training summary
    logger.info("\n📋 Training Summary:")
    logger.info(f"Successfully trained models: {list(models.keys())}")
    if training_errors:
        logger.warning("⚠️ Training Errors:")
        for model_name, error in training_errors.items():
            logger.warning(f"- {model_name}: {error['error_type']}: {error['error_message']}")
    
    return models, tfidf_vectorizer, X_test_tfidf, X_train_tfidf, model_metrics

def compute_shap_values(models, X_test_tfidf, X_train_tfidf):
    """Compute and save SHAP explanations for model interpretability"""
    logger.info("\n🔍 Computing SHAP explanations...")
    background_data = X_train_tfidf[:50]
    
    for model_name, model in models.items():
        try:
            explainer = shap.Explainer(model.predict, background_data)
            shap_values = explainer(X_test_tfidf)
            
            # Save SHAP plot
            shap.summary_plot(
                shap_values, 
                X_test_tfidf,
                show=False,
                plot_size=(12, 8)
            )
            shap_file = LOGS_DIR / f"shap_summary_{model_name}.png"
            shap.save_html(shap_file)
            logger.info(f"✅ SHAP summary saved: {shap_file}")
            
        except Exception as e:
            logger.error(f"❌ SHAP computation failed for {model_name}: {str(e)}")

def main():
    """Main training pipeline"""
    try:
        logger.info("\n" + "="*50 + "\n📚 Starting model training pipeline\n" + "="*50)
        
        # Load data
        df, training_file = load_data()
        logger.info(f"Loaded {len(df)} records from {training_file}")
        
        # Train models
        models, vectorizer, X_test, X_train, metrics = train_models(
            df,
            training_file,
            compute_shap=True
        )
        
        # Log final metrics summary
        logger.info("\n📊 Final Performance Summary:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Check and log model registry status
        logger.info("\n📦 Model Registry Status:")
        for model_name in models.keys():
            version = model_registry.get_latest_version(model_name)
            locations = model_registry.get_model_locations(model_name, version)
            logger.info(f"\n{model_name} v{version}:")
            for loc_type, path in locations.items():
                logger.info(f"  {loc_type}: {path}")
        
        logger.info("\n✅ Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()