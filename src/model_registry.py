import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
from google.cloud import storage

class ModelRegistry:
    def __init__(self, registry_dir: str, bucket_name: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize model registry with optional GCS backup
        
        Args:
            registry_dir: Directory to store registry files
            bucket_name: Optional GCS bucket for backup
            credentials_path: Optional path to GCS credentials
        """
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / 'model_registry.json'
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS if credentials provided
        self.bucket_name = bucket_name
        if bucket_name and credentials_path:
            try:
                self.storage_client = storage.Client.from_service_account_json(credentials_path)
                self.bucket = self.storage_client.bucket(bucket_name)
                self.is_cloud_available = True
            except Exception as e:
                logging.error(f"Failed to initialize GCS for registry: {str(e)}")
                self.is_cloud_available = False
        else:
            self.is_cloud_available = False
        
        # Initialize or load registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load or initialize the registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.registry = json.load(f)
            except Exception as e:
                logging.error(f"Error loading registry: {str(e)}")
                self.registry = {'models': {}, 'metadata': {'last_updated': None}}
        else:
            self.registry = {'models': {}, 'metadata': {'last_updated': None}}

    def _save_registry(self) -> None:
        """Save registry to file and backup to GCS if available"""
        # Update last modified timestamp
        self.registry['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save locally
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving registry locally: {str(e)}")
            raise
        
        # Backup to GCS if available
        if self.is_cloud_available:
            try:
                blob = self.bucket.blob('model_registry/model_registry.json')
                blob.upload_from_filename(str(self.registry_file))
                logging.info("✅ Registry backed up to GCS")
            except Exception as e:
                logging.error(f"Failed to backup registry to GCS: {str(e)}")

    def register_model(self, 
                      model_name: str, 
                      version: int, 
                      locations: Dict[str, str], 
                      metrics: Dict[str, float],
                      training_data: str,
                      additional_metadata: Optional[Dict] = None) -> None:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model
            version: Version number
            locations: Dict containing paths ('local_path', 'gcs_path')
            metrics: Dict of performance metrics
            training_data: Path or identifier of training data used
            additional_metadata: Optional additional metadata
        """
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {}
        
        # Create version entry
        version_entry = {
            'version': version,
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
        logging.info(f"✅ Registered {model_name} v{version}")

    def get_latest_version(self, model_name: str) -> Optional[int]:
        """Get the latest version number for a model"""
        if model_name not in self.registry['models']:
            return None
        
        versions = [int(v) for v in self.registry['models'][model_name].keys()]
        return max(versions) if versions else None

    def get_model_info(self, model_name: str, version: Optional[int] = None) -> Optional[Dict]:
        """Get information about a specific model version"""
        if model_name not in self.registry['models']:
            return None
        
        if version is None:
            version = self.get_latest_version(model_name)
            if version is None:
                return None
        
        return self.registry['models'][model_name].get(str(version))

    def get_model_locations(self, model_name: str, version: Optional[int] = None) -> Optional[Dict[str, str]]:
        """Get storage locations for a specific model version"""
        model_info = self.get_model_info(model_name, version)
        return model_info['locations'] if model_info else None

    def list_models(self) -> List[str]:
        """Get list of all registered models"""
        return list(self.registry['models'].keys())

    def get_model_versions(self, model_name: str) -> List[int]:
        """Get all versions for a specific model"""
        if model_name not in self.registry['models']:
            return []
        return sorted([int(v) for v in self.registry['models'][model_name].keys()])

    def get_model_metrics(self, model_name: str, version: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Get metrics for a specific model version"""
        model_info = self.get_model_info(model_name, version)
        return model_info['metrics'] if model_info else None

    def update_model_status(self, model_name: str, version: int, status: str) -> None:
        """Update the status of a model version (e.g., 'active', 'archived')"""
        if model_name in self.registry['models'] and str(version) in self.registry['models'][model_name]:
            self.registry['models'][model_name][str(version)]['status'] = status
            self._save_registry()
            logging.info(f"Updated {model_name} v{version} status to: {status}")

# Example usage:
if __name__ == "__main__":
    from config import get_config
    
    # Load configuration
    config = get_config()
    gcp_config = config.get_gcp_config()
    paths_config = config.get_paths_config()
    
    # Initialize registry with configuration
    registry = ModelRegistry(
        registry_dir=str(Path(paths_config['models_dir']) / "registry"),
        bucket_name=gcp_config['bucket_name'],
        credentials_path=gcp_config['credentials_path']
    )
    
    # Example: Register a model
    registry.register_model(
        model_name="lgbm",
        version=1,
        locations={
            "local_path": "models/lgbm_model_v1.pkl",
            "gcs_path": "gs://bucket/models/lgbm_model_v1.pkl"
        },
        metrics={
            "accuracy": 0.85,
            "f1_score": 0.83
        },
        training_data="training_data_v1.csv"
    )
    
    # Example: Get latest version info
    model_info = registry.get_model_info("lgbm")
    print(f"Latest model info: {model_info}")