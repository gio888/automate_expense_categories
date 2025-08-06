import sys
import os
import logging
from pathlib import Path

# Ensure src directory is in the system path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.model_registry import ModelRegistry
from src.model_storage import ModelStorage

# Load configuration
config = get_config()
if not config.validate_required():
    print("Configuration validation failed. Please check your settings.")
    sys.exit(1)

# Get configuration values
gcp_config = config.get_gcp_config()
paths_config = config.get_paths_config()

# Define paths from configuration
MODELS_DIR = Path(paths_config['models_dir'])
REGISTRY_DIR = MODELS_DIR / "registry"

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize model registry and storage with configuration
model_registry = ModelRegistry(str(REGISTRY_DIR))
model_storage = ModelStorage(
    bucket_name=gcp_config['bucket_name'],
    credentials_path=gcp_config['credentials_path'],
    models_dir=str(MODELS_DIR)
)

def verify_latest_model():
    """Check if the latest model and vectorizer are available, or retrieve them if missing."""
    
    # Debugging: Print the raw contents of the registry file
    print("DEBUG: Expected registry path:", model_registry.registry_file)
    with open(model_registry.registry_file, "r") as f:
        print("DEBUG: Raw registry file contents:", f.read())

    # Debugging: Print available models in memory
    print("DEBUG: Registry data in memory:", model_registry.registry["models"])
    
    latest_version = model_registry.get_latest_version("lgbm")
    if latest_version is None:
        logger.error("No registered model versions found! Batch prediction cannot proceed.")
        return False
    
    logger.info(f"Latest model version: v{latest_version}")
    
    # Check model files
    model_types = ['lgbm', 'xgboost', 'catboost']
    all_models_exist = True
    
    for model_type in model_types:
        model_path = os.path.join(MODELS_DIR, f"{model_type}_model_v{latest_version}.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"{model_type} model missing locally. Attempting cloud retrieval...")
            try:
                model_storage.load_model(model_type, latest_version)
            except Exception as e:
                logger.error(f"Failed to retrieve {model_type} model: {str(e)}")
                all_models_exist = False
    
    # Check vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer_v{latest_version}.pkl")
    if not os.path.exists(vectorizer_path):
        logger.warning("TF-IDF vectorizer missing locally. Attempting cloud retrieval...")
        try:
            model_storage.load_model("tfidf_vectorizer", latest_version)
        except Exception as e:
            logger.error(f"Failed to retrieve TF-IDF vectorizer: {str(e)}")
            all_models_exist = False
    
    if all_models_exist:
        logger.info("✅ All required models and vectorizer are available.")
    else:
        logger.error("❌ Some models are missing. Fix issues before running batch prediction.")
    
    return all_models_exist

if __name__ == "__main__":
    success = verify_latest_model()
    if not success:
        exit(1)
