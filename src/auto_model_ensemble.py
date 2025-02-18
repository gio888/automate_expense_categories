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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, roc_auc_score, average_precision_score
from flaml import AutoML
from google.cloud import storage

# Import our custom classes
from src.model_storage import ModelStorage
from src.model_registry import ModelRegistry

#Import transacion source to handle different types of transactions
from src.transaction_types import TransactionSource

# ✅ Define Paths and Configuration
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REGISTRY_DIR = MODELS_DIR / "registry"

# GCS Configuration
BUCKET_NAME = "expense-categorization-ml-models-backup"
CREDENTIALS_PATH = "***CREDENTIALS-REMOVED***"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)

# ✅ Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(str(LOGS_DIR / "training.log")),
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
        if version is None:
            version = 1
        else:
            version += 1
        
        # Save model using storage class
        locations = model_storage.save_model(model, name, version)
        
        # Register the model
        model_registry.register_model(
            model_name=name,
            version=version,
            locations=locations,
            metrics=metrics,
            training_data=training_data,
            additional_metadata={  # ✅ This must wrap extra fields properly
                'framework': 'flaml',
                'python_version': sys.version.split(" ")[0]  # ✅ Dynamically retrieved
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
    
def load_data(source: TransactionSource):
    """
    Load the latest training data file for specified source
    """
    
    try:
        source_prefix = f"training_data_{source.value}_"
        files = [
            f for f in os.listdir(DATA_DIR) 
            if f.startswith(source_prefix)
        ]
        if not files:
            raise FileNotFoundError(
                f"No training data files found for source: {source.value}"
            )
        
        latest_file = sorted(
            files, key=lambda x: os.path.getmtime(DATA_DIR / x)
        )[-1]
        logger.info(
            f"📂 Loading {source.value} training data: {latest_file}"
        )
        df = pd.read_csv(DATA_DIR / latest_file)
        
        if 'transaction_source' not in df.columns:
            raise ValueError(
                f"Training data missing transaction_source column: {latest_file}"
            )
            
        if not all(df['transaction_source'] == source.value):
            mismatched = df[df['transaction_source'] != source.value]
            raise ValueError(
                f"Found {len(mismatched)} records with incorrect source in {latest_file}"
            )
        
        logger.info(f"✅ Loaded {len(df)} records from {latest_file}")
        return df, latest_file
    except Exception as e:
        import traceback
        logger.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")
        raise

def evaluate_model(model, X_test, y_test, compute_extra_metrics=True):
    """
    Evaluate model performance with various metrics.
    
    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: True labels for the test set.
        compute_extra_metrics: Whether to compute additional metrics like ROC AUC and PR AUC.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
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

def train_models(df, training_file, source, compute_shap=True, min_category_samples=5):
    """Train multiple models using FLAML AutoML"""
    # Prepare initial data
    X = df["Description"].fillna("")
    y = df["Category"].astype(str).fillna("Other")
    
    # Analyze initial category distribution
    category_counts = pd.Series(y).value_counts()
    logger.info("\n📊 Initial category distribution:")
    for category, count in category_counts.items():
        logger.info(f"{category}: {count} samples ({count/len(y)*100:.2f}%)")
    
    # Filter categories based on minimum samples threshold
    valid_categories = category_counts[category_counts >= min_category_samples].index
    mask = y.isin(valid_categories)
    X = X[mask]
    y = y[mask]
    
    # Log filtering results
    excluded_categories = set(category_counts.index) - set(valid_categories)
    removed_samples = len(df) - len(X)
    logger.info(f"\n📊 Category filtering results:")
    logger.info(f"Minimum samples threshold: {min_category_samples}")
    logger.info(f"Removed categories: {len(excluded_categories)}")
    logger.info(f"Removed samples: {removed_samples} ({removed_samples/len(df)*100:.1f}%)")
    if excluded_categories:
        logger.info("\nExcluded categories (insufficient samples):")
        for cat in excluded_categories:
            logger.info(f"- {cat}: {category_counts.get(cat, 0)} samples")
    
    # ✅ Apply Label Encoding BEFORE splitting the data
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Convert category names to numbers

    # After fitting the label encoder
    save_model(
        model=label_encoder,
        name=f"{source.value}_label_encoder",
        metrics={},
        training_data=training_file
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,  # ⬅️ Now using encoded labels
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    # Compute balanced class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    balanced_weights = dict(zip(np.unique(y_train), class_weights))

    print("\n🔍 Computed Class Weights:")
    print(balanced_weights)  # Optional: Print to verify class weight distribution

    # ✅ ADD DEBUGGING CODE
    print("\n🔍 DEBUG: First 10 values of y_train before training:", y_train[:10])
    print("Type of y_train:", type(y_train))
    print("Unique values in y_train:", set(y_train))
    # Ensure y_train is numerical before passing it to AutoML
    if isinstance(y_train[0], str):
        print("\n❌ ERROR: y_train is still in string format before training!")
    else:
        print("\n✅ y_train is correctly encoded before training!")
    
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"✅ X_train_tfidf shape: {X_train_tfidf.shape}")  # Debugging feature count

    # ✅ Save the vectorized training data for SHAP
    X_train_tfidf_path = "models/X_train_tfidf_v5.pkl"  # Change versioning dynamically if needed
    joblib.dump(X_train_tfidf, X_train_tfidf_path)
    print(f"✅ Saved vectorized dataset: {X_train_tfidf_path}")
    
    models = {}
    model_metrics = {}
    
    for model_name in ["lgbm", "xgboost", "catboost"]:
        try:
            logger.info(f"\n{'='*30}\n🚀 Training {model_name} model...\n{'='*30}")
            automl = AutoML()
            automl.fit(
                X_train_tfidf,
                y_train,
                task="classification",
                metric="macro_f1",
                time_budget=300,
                estimator_list=[model_name],
                verbose=2,
                sample_weight=np.array([balanced_weights[label] for label in y_train])  # ✅ Convert to NumPy array
            )
            
            models[model_name] = automl
            
            # Evaluate model
            metrics = evaluate_model(automl, X_test_tfidf, y_test, compute_extra_metrics=True)
            model_metrics[model_name] = metrics
            
            # Save and register model
            save_model(
                model=automl,
                name=f"{source.value}_{model_name}",
                metrics=metrics,
                training_data=training_file
            )
            
            # Generate and save classification report
            y_pred = automl.predict(X_test_tfidf)
            report = classification_report(y_test, y_pred)
            with open(LOGS_DIR / f"classification_report_{model_name}.txt", "w") as f:
                f.write(report)
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            continue
    
    # Save vectorizer
    try:
        save_model(
            model=tfidf_vectorizer,
            name=f"{source.value}_tfidf_vectorizer",
            metrics={},  # Vectorizer doesn't have metrics
            training_data=training_file
        )
    except Exception as e:
        logger.error(f"❌ Error saving vectorizer: {str(e)}")
        joblib.dump(tfidf_vectorizer, MODELS_DIR / "tfidf_vectorizer_fallback.pkl")
    
    if compute_shap and models:
        compute_shap_values(models, X_test_tfidf, X_train_tfidf)
    
    return models, tfidf_vectorizer, X_test_tfidf, X_train_tfidf, model_metrics

def compute_shap_values(models, X_test_tfidf, X_train_tfidf):
    """Compute and save SHAP explanations for model interpretability"""
    print("\n🔍 Computing SHAP explanations...")
    
    # ✅ Use a proper background dataset
    background_data = X_train_tfidf[:50]  # Use multiple samples instead of one

    for model_name, automl in models.items():
        try:
            # ✅ Extract the actual LightGBM/XGBoost/CatBoost model
            if hasattr(automl, "model") and hasattr(automl.model, "estimator"):
                model = automl.model.estimator  # Get the core ML model
                print(f"✅ Extracted Model for SHAP: {type(model)}")
            else:
                raise ValueError(f"⚠️ Could not extract underlying model for {model_name}, skipping SHAP.")

            # ✅ Initialize SHAP Explainer with the correct model
            explainer = shap.TreeExplainer(model.predict, background_data)

            # ✅ Compute SHAP values on a small batch
            shap_values = explainer(X_test_tfidf[:5])

            # ✅ Save SHAP summary (optional)
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_test_tfidf[:50])
            shap_file = LOGS_DIR / f"shap_summary_{model_name}.png"
            plt.savefig(shap_file)
            plt.close()
            logger.info(f"✅ SHAP summary saved: {shap_file}")

        except Exception as e:
            logger.error(f"❌ SHAP computation failed for {model_name}: {str(e)}")

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Train models for specific transaction source')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['household', 'credit_card'],
                       help='Transaction source to train models for')
    return parser.parse_args()

def main():
    """Main training pipeline"""
    try:
        args = parse_args()
        source = TransactionSource(args.source)
        
        logger.info("\n" + "="*50)
        logger.info(f"📚 Starting model training pipeline for {source.value}")
        logger.info("="*50 + "\n")

        # ✅ Load source-specific data
        try:
            df, training_file = load_data(source)
            if df.empty:
                logger.error(f"❌ No data found for {source.value}. Skipping training.")
                return  # Stop execution if no data is available
            logger.info(f"✅ Loaded {len(df)} records from {training_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load data for {source.value}: {str(e)}")
            return  # Stop execution if data cannot be loaded

        # ✅ Train models (handling potential failure)
        models, metrics = train_models(
            df,
            training_file,
            source,
            compute_shap=True
        )
        if not models:
            logger.error(f"❌ No models were trained for {source.value}. Skipping registry updates.")
            return  # Stop execution if training failed

        # ✅ Log final metrics summary
        logger.info("\n📊 Final Performance Summary:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        # ✅ Check and log model registry status (only if models were trained)
    # ✅ Check and log model registry status (only if models were trained)
    if models:
        logger.info("\n📦 Model Registry Status:")
        for model_name in models.keys():
            source_model_name = f"{source.value}_{model_name}"
            version = model_registry.get_latest_version(source_model_name)
            if version is not None:
                locations = model_registry.get_model_locations(source_model_name, version)
                if locations:
                    logger.info(f"\n{source_model_name} v{version}:")
                    for loc_type, path in locations.items():
                        logger.info(f"  {loc_type}: {path}")
                else:
                    logger.warning(f"No locations found for {source_model_name} v{version}")
            else:
                logger.warning(f"No version found for {source_model_name}")

    logger.info("\n✅ Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"❌ Unexpected failure in training pipeline: {str(e)}")
        logger.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main()