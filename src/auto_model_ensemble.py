import os
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

# ✅ Define Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ✅ Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the latest training data file"""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith("training_data_")]
        if not files:
            raise FileNotFoundError("No training data files found")
        
        DATA_FILE = sorted(files)[-1]
        logger.info(f"📂 Loading training data: {DATA_FILE}")
        df = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE))
        
        # Log DVC reminder
        logger.info(f"🔔 Reminder: Run `dvc add {os.path.join(DATA_DIR, DATA_FILE)}` to track dataset changes.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def consolidate_rare_categories(df, min_samples=5):
    """Consolidate categories with fewer than min_samples into an 'Other' category"""
    category_counts = df["Category"].value_counts()
    rare_categories = category_counts[category_counts < min_samples].index
    
    logger.info(f"\n📊 Before consolidation: {len(category_counts)} categories")
    logger.info(f"Found {len(rare_categories)} categories with fewer than {min_samples} samples")
    
    # Create a mapping dictionary
    category_mapping = {cat: "Other" for cat in rare_categories}
    
    # Apply mapping to create new column
    df["Category_Consolidated"] = df["Category"].map(lambda x: category_mapping.get(x, x))
    
    # Log the changes
    new_counts = df["Category_Consolidated"].value_counts()
    logger.info(f"After consolidation: {len(new_counts)} categories")
    logger.info("\nCategory distribution after consolidation:")
    for cat, count in new_counts.items():
        logger.info(f"{cat}: {count} samples ({count/len(df)*100:.2f}%)")
    
    return df

def analyze_category_distribution(y):
    """Analyze and log category distribution"""
    category_counts = pd.Series(y).value_counts()
    total_samples = len(y)
    
    logger.info("\n📊 Category distribution:")
    for category, count in category_counts.items():
        percentage = (count/total_samples) * 100
        logger.info(f"{category}: {count} samples ({percentage:.2f}%)")
    
    return category_counts

def save_model(model, name):
    """Save a model to disk"""
    path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(model, path)
    logger.info(f"✅ Model saved: {path}")
    return path

def evaluate_model(model, X_test, y_test, compute_extra_metrics=False):
    """Evaluate model performance with various metrics"""
    metrics = {
        'macro_f1': f1_score(y_test, model.predict(X_test), average='macro')
    }
    
    if compute_extra_metrics:
        y_pred_proba = model.predict_proba(X_test)
        y_test_dummies = pd.get_dummies(y_test)
        
        metrics.update({
            'roc_auc': roc_auc_score(y_test_dummies, y_pred_proba, multi_class='ovr'),
            'pr_auc': average_precision_score(y_test_dummies, y_pred_proba)
        })
    
    return metrics

def log_metrics(metrics, model_name):
    """Log model performance metrics"""
    logger.info(f"📊 {model_name} performance: {metrics}")
    
    if os.getenv('PROD_ENVIRONMENT'):
        metrics_file = os.path.join(LOGS_DIR, f"{model_name}_metrics.csv")
        pd.DataFrame([metrics]).to_csv(
            metrics_file,
            mode='a',
            header=not os.path.exists(metrics_file)
        )

def train_models(df, compute_shap=True, metrics_callback=None):
    """Train multiple models using FLAML AutoML"""
    # Consolidate rare categories
    df = consolidate_rare_categories(df, min_samples=5)
    
    # Prepare data
    X = df["Description"].fillna("")
    y = df["Category_Consolidated"].astype(str).fillna("Other")
    
    # Analyze category distribution
    analyze_category_distribution(y)
    
    # Encode labels before splitting
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save category mapping
    category_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    joblib.dump(category_mapping, os.path.join(MODELS_DIR, "category_mapping.pkl"))
    logger.info("✅ Category mapping saved.")
    
    # Split data with stratification
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    logger.info(f"🔍 Unique categories in training set: {len(np.unique(y_train_encoded))}")
    logger.info(f"🔍 Unique categories in test set: {len(np.unique(y_test_encoded))}")
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    logger.info("✅ Label encoder saved.")
    
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Verify data alignment
    assert X_train_tfidf.shape[0] == len(y_train_encoded), "Training data/label mismatch"
    assert X_test_tfidf.shape[0] == len(y_test_encoded), "Test data/label mismatch"
    
    logger.info(f"🔍 Training data shape: {X_train_tfidf.shape}, Labels shape: {len(y_train_encoded)}")
    logger.info(f"🔍 Test data shape: {X_test_tfidf.shape}, Labels shape: {len(y_test_encoded)}")
    
    # Save vectorizer
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    logger.info("✅ TF-IDF vectorizer saved.")
    
    models = {}
    for model_name in ["lgbm", "xgboost", "catboost"]:
        try:
            logger.info(f"🚀 Training {model_name} model...")
            automl = AutoML()
            automl.fit(
                X_train_tfidf,
                y_train_encoded,
                task="classification",
                metric="macro_f1",
                time_budget=300,
                estimator_list=[model_name],
                verbose=2
            )
            
            models[model_name] = automl
            save_model(automl, model_name)
            
            metrics = evaluate_model(automl, X_test_tfidf, y_test_encoded, compute_extra_metrics=True)
            log_metrics(metrics, model_name)
            
            if metrics_callback:
                metrics_callback(metrics)
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
    
    return models, tfidf_vectorizer, X_test_tfidf, X_train_tfidf, label_encoder

def compute_shap_explanations(models, X_test_tfidf, X_train_tfidf):
    """Compute and save SHAP explanations for model interpretability"""
    logger.info("🔍 Computing SHAP explanations...")
    background_data = X_train_tfidf[:50]  # Use subset of training data as background
    
    for model_name, model in models.items():
        try:
            explainer = shap.Explainer(model.predict, background_data)
            shap_values = explainer(X_test_tfidf)
            shap.summary_plot(shap_values, X_test_tfidf, show=False)
            shap_file = os.path.join(LOGS_DIR, f"shap_summary_{model_name}.png")
            shap.save(shap_file)
            logger.info(f"✅ SHAP summary saved: {shap_file}")
        except Exception as e:
            logger.error(f"❌ SHAP computation failed for {model_name}: {str(e)}")
            logger.error("Stack trace:", exc_info=True)

def main():
    """Main training pipeline"""
    try:
        # Load and preprocess data
        df = load_data()
        
        # Train models
        models, tfidf_vectorizer, X_test_tfidf, X_train_tfidf, label_encoder = train_models(
            df,
            compute_shap=True
        )
        
        # Generate model explanations
        compute_shap_explanations(models, X_test_tfidf, X_train_tfidf)
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()