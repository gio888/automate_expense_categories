import os
import pandas as pd
import numpy as np
import joblib
import glob
import logging
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report
from flaml import AutoML

# ✅ Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAINING_LOG_FILE = os.path.join(LOGS_DIR, "training_logs.csv")

# ✅ Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ✅ Setup logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "training.log")),
        logging.StreamHandler()  # This ensures output goes to console too
    ]
)
logger = logging.getLogger()

# ✅ Function to get the latest training data file
def get_latest_training_file(data_dir: str, prefix: str = "training_data", extension: str = "csv") -> str:
    """Get the latest version of the training data file."""
    training_files = glob.glob(os.path.join(data_dir, f"{prefix}_*.{extension}"))
    if not training_files:
        raise FileNotFoundError(f"No training files found in {data_dir} with prefix '{prefix}'.")
    
    # Sort by last modified time
    training_files.sort(key=os.path.getmtime)
    return training_files[-1]

try:
    # ✅ Load and prepare data
    DATA_FILE = get_latest_training_file(DATA_DIR)
    logger.info(f"\n{'='*50}\n📂 Starting new training run\n{'='*50}")
    logger.info(f"Loading training data from: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)[["Description", "Category"]]
    logger.info(f"Loaded {len(df)} transactions for training")
    df["Category"] = df["Category"].fillna("Unknown")
    
    X = df["Description"]
    y = df["Category"].astype(str)
    
    # ✅ Split data into training and testing sets
    logger.info("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ✅ Convert text to TF-IDF features
    logger.info("🔍 Transforming text using TF-IDF...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()
    
    # ✅ Determine version for this run
    existing_versions = [int(f.split("_v")[-1].split(".pkl")[0]) 
                        for f in glob.glob(os.path.join(MODELS_DIR, "lgbm_model_v*.pkl"))]
    version_number = max(existing_versions) + 1 if existing_versions else 1
    version = f"v{version_number}"
    
    # ✅ Save TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer_{version}.pkl")
    joblib.dump(tfidf_vectorizer, tfidf_path)
    logger.info(f"✅ TF-IDF vectorizer saved to: {tfidf_path}")
    
    # ✅ Train ensemble models
    models = []
    model_names = ["lgbm", "xgboost", "catboost"]
    metrics = {}
    
    for model_name in model_names:
        logger.info(f"\n{'='*30}\n🚀 Training {model_name} model...\n{'='*30}")
        try:
            logger.info(f"This may take several minutes...")
            automl = AutoML()
            automl.fit(
                X_train_tfidf, y_train,
                task="classification",
                metric="macro_f1",
                time_budget=300,
                estimator_list=[model_name],
                verbose=2
            )
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{model_name}_model_{version}.pkl")
            joblib.dump(automl, model_path)
            logger.info(f"✅ {model_name} model saved to: {model_path}")
            models.append(automl)
            
            # Evaluate
            y_pred = automl.predict(X_test_tfidf)
            y_pred = [str(label) for label in y_pred]
            
            # Save classification report
            report = classification_report(y_test, y_pred)
            report_path = os.path.join(LOGS_DIR, f"classification_report_{model_name}_{version}.txt")
            with open(report_path, "w") as f:
                f.write(report)
            
            # Calculate metrics
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            metrics[model_name] = {"macro_f1": macro_f1}
            
            try:
                proba = automl.predict_proba(X_test_tfidf)
                metrics[model_name].update({
                    "roc_auc": roc_auc_score(pd.get_dummies(y_test), proba, average="macro", multi_class="ovr"),
                    "pr_auc": average_precision_score(pd.get_dummies(y_test), proba, average="macro")
                })
            except:
                logger.warning(f"Probability scores not available for {model_name}")
                metrics[model_name].update({"roc_auc": None, "pr_auc": None})
            
            logger.info(f"📈 {model_name} Performance: {metrics[model_name]}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}", exc_info=True)
            continue
    
    # ✅ Log training results
    timestamp = datetime.now().isoformat()
    log_entry = {
        "version": version,
        "training_data": os.path.basename(DATA_FILE),
        "models": "|".join([f"{name}_model_{version}.pkl" for name in model_names]),
        "macro_f1": np.mean([metrics[m]["macro_f1"] for m in metrics]),
        "roc_auc": np.mean([metrics[m]["roc_auc"] for m in metrics if metrics[m]["roc_auc"] is not None]),
        "pr_auc": np.mean([metrics[m]["pr_auc"] for m in metrics if metrics[m]["pr_auc"] is not None]),
        "timestamp": timestamp
    }
    
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(TRAINING_LOG_FILE, mode="a", header=not os.path.exists(TRAINING_LOG_FILE), index=False)
    logger.info(f"✅ Training logs updated in: {TRAINING_LOG_FILE}")
    
    # ✅ Clean up memory
    gc.collect()
    logger.info("✅ Training pipeline completed successfully!")
    
except Exception as e:
    logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
    raise