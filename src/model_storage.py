import os
import logging
from datetime import datetime
from google.cloud import storage
from google.api_core import retry
import tempfile
import joblib
from pathlib import Path

class ModelStorage:
    def __init__(self, bucket_name, credentials_path, models_dir, cache_dir=None):
        """
        Initialize model storage with both local and GCS capabilities
        
        Args:
            bucket_name (str): GCS bucket name
            credentials_path (str): Path to GCP credentials JSON
            models_dir (str): Local directory for model storage
            cache_dir (str, optional): Directory for caching cloud models
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.models_dir / 'cache'
        self.bucket_name = bucket_name
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize GCS client
        try:
            self.storage_client = storage.Client.from_service_account_json(credentials_path)
            self.bucket = self.storage_client.bucket(bucket_name)
            self.is_cloud_available = True
        except Exception as e:
            logging.error(f"Failed to initialize GCS client: {str(e)}")
            self.is_cloud_available = False

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def upload_to_gcs(self, local_path: Path, gcs_path: str) -> bool:
        """Upload file to GCS with retry logic"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            logging.info(f"✅ Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to upload {local_path} to GCS: {str(e)}")
            return False

    def save_model(self, model, name: str, version: int) -> dict:
        """
        Save model locally and to GCS
        
        Returns:
            dict: Locations where model was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        locations = {'version': version, 'timestamp': timestamp}
        
        # Save locally
        local_filename = f"{name}_model_v{version}.pkl"
        local_path = self.models_dir / local_filename
        try:
            joblib.dump(model, local_path)
            locations['local_path'] = str(local_path)
            logging.info(f"✅ Model saved locally: {local_path}")
        except Exception as e:
            logging.error(f"❌ Failed to save model locally: {str(e)}")
            raise

        # Upload to GCS if available
        if self.is_cloud_available:
            gcs_path = f"models/v{version}/{local_filename}"
            if self.upload_to_gcs(local_path, gcs_path):
                locations['gcs_path'] = f"gs://{self.bucket_name}/{gcs_path}"
        
        return locations

    def load_model(self, name: str, version: int) -> object:
        """
        Load model with fallback strategy:
        1. Try local
        2. Try cached
        3. Download from GCS and cache
        """
        local_path = self.models_dir / f"{name}_model_v{version}.pkl"
        cache_path = self.cache_dir / f"{name}_model_v{version}.pkl"
        
        # Try loading locally
        if local_path.exists():
            try:
                return joblib.load(local_path)
            except Exception as e:
                logging.warning(f"Failed to load local model: {str(e)}")
        
        # Try loading from cache
        if cache_path.exists():
            try:
                return joblib.load(cache_path)
            except Exception as e:
                logging.warning(f"Failed to load cached model: {str(e)}")
        
        # Try downloading from GCS
        if self.is_cloud_available:
            gcs_path = f"models/v{version}/{name}_model_v{version}.pkl"
            blob = self.bucket.blob(gcs_path)
            
            if blob.exists():
                try:
                    # Download to cache
                    blob.download_to_filename(str(cache_path))
                    model = joblib.load(cache_path)
                    logging.info(f"✅ Downloaded and cached model from GCS: {gcs_path}")
                    return model
                except Exception as e:
                    logging.error(f"Failed to download/load model from GCS: {str(e)}")
        
        raise FileNotFoundError(f"Could not find model {name} v{version} in any location")

# Example usage in auto_model_ensemble.py:
def get_model_storage():
    """Get configured ModelStorage instance"""
    return ModelStorage(
        bucket_name="expense-categorization-ml-models-backup",
        credentials_path="***CREDENTIALS-REMOVED***",
        models_dir=os.path.join(PROJECT_ROOT, "models")
    )

def save_model(model, name):
    """Enhanced save_model function"""
    storage = get_model_storage()
    version = get_next_version(name)
    locations = storage.save_model(model, name, version)
    
    # Update model registry
    update_model_registry(locations)
    return locations

# Example usage in batch_predict_ensemble_cld.py:
def load_latest_models():
    """Enhanced model loading with cloud fallback"""
    storage = get_model_storage()
    version = get_latest_model_version()
    
    models = {}
    for model_name in ['lgbm', 'xgboost', 'catboost']:
        try:
            models[model_name] = storage.load_model(model_name, version)
        except Exception as e:
            logging.error(f"Failed to load {model_name} model: {str(e)}")
            raise
    
    vectorizer = storage.load_model('tfidf_vectorizer', version)
    return models, vectorizer, version