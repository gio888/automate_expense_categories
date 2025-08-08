"""
Configuration management for expense categorization ML pipeline.

Loads configuration from:
1. Environment variables (highest priority)
2. config.yaml file (medium priority)  
3. Default values (lowest priority)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration management with multiple sources"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from multiple sources
        
        Args:
            config_file: Path to YAML config file (optional)
        """
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file or self.project_root / "config.yaml"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from all sources with proper priority"""
        # Start with defaults
        self._config = self._get_defaults()
        
        # Override with config file if it exists
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                self._merge_config(self._config, file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables (highest priority)
        self._load_from_environment()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Default configuration values"""
        return {
            'gcp': {
                'bucket_name': 'expense-categorization-ml-models-backup',
                'credentials_path': None,  # Must be provided by user
                'project_id': None
            },
            'paths': {
                'data_dir': str(self.project_root / 'data'),
                'models_dir': str(self.project_root / 'models'),
                'logs_dir': str(self.project_root / 'logs'),
                'cache_dir': str(self.project_root / 'cache')
            },
            'ml': {
                'batch_size': 1000,
                'min_samples_household': 5,
                'min_samples_credit_card': 3,
                'confidence_threshold_household': 0.6,
                'confidence_threshold_credit_card': 0.7
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # GCP settings
            'GCP_CREDENTIALS_PATH': ['gcp', 'credentials_path'],
            'GCP_BUCKET_NAME': ['gcp', 'bucket_name'],
            'GCP_PROJECT_ID': ['gcp', 'project_id'],
            
            # Path settings
            'DATA_DIR': ['paths', 'data_dir'],
            'MODELS_DIR': ['paths', 'models_dir'],
            'LOGS_DIR': ['paths', 'logs_dir'],
            'CACHE_DIR': ['paths', 'cache_dir'],
            
            # ML settings
            'ML_BATCH_SIZE': ['ml', 'batch_size'],
            'ML_CONFIDENCE_THRESHOLD_HOUSEHOLD': ['ml', 'confidence_threshold_household'],
            'ML_CONFIDENCE_THRESHOLD_CREDIT_CARD': ['ml', 'confidence_threshold_credit_card'],
            
            # Logging
            'LOG_LEVEL': ['logging', 'level']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if env_var == 'ML_BATCH_SIZE':
                    value = int(value)
                elif 'CONFIDENCE_THRESHOLD' in env_var:
                    value = float(value)
                
                self._set_nested_value(self._config, config_path, value)
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: Dict, path: list, value: Any):
        """Set a nested configuration value using a path list"""
        for key in path[:-1]:
            config = config.setdefault(key, {})
        config[path[-1]] = value
    
    def get(self, *path, default=None) -> Any:
        """
        Get configuration value using dot notation or path components
        
        Examples:
            config.get('gcp', 'bucket_name')
            config.get('paths.data_dir')
        """
        if len(path) == 1 and '.' in path[0]:
            path = path[0].split('.')
        
        value = self._config
        for key in path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def validate_required(self) -> bool:
        """
        Validate that all required configuration is present
        
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = [
            ('gcp', 'credentials_path')
        ]
        
        missing = []
        for field_path in required_fields:
            if not self.get(*field_path):
                missing.append('.'.join(field_path))
        
        if missing:
            logger.error("❌ Configuration validation failed!")
            logger.error(f"Missing required settings: {', '.join(missing)}")
            logger.error("")
            logger.error("💡 How to fix this:")
            logger.error("1. Copy template: cp config.example.yaml config.yaml")
            logger.error("2. Edit config.yaml with your GCP credentials path")
            logger.error("3. Or set environment variable: export GCP_CREDENTIALS_PATH=/path/to/key.json")
            logger.error("4. Run setup validation: python setup_validator.py")
            return False
        
        # Validate credentials file exists
        creds_path = self.get('gcp', 'credentials_path')
        if creds_path and not Path(creds_path).exists():
            logger.error("❌ GCP credentials file not found!")
            logger.error(f"Looking for: {creds_path}")
            logger.error("")
            logger.error("💡 How to fix this:")
            logger.error("1. Go to Google Cloud Console > IAM & Admin > Service Accounts")
            logger.error("2. Create service account with 'Storage Admin' role")
            logger.error("3. Download JSON key file")
            logger.error("4. Update config.yaml with correct path to the key file")
            logger.error("5. Or set GCP_CREDENTIALS_PATH environment variable")
            return False
        
        return True
    
    def get_gcp_config(self) -> Dict[str, Any]:
        """Get GCP-specific configuration"""
        return self.get('gcp', default={})
    
    def get_paths_config(self) -> Dict[str, str]:
        """Get paths configuration"""
        return self.get('paths', default={})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML-specific configuration"""
        return self.get('ml', default={})
    
    def get_file_handling_config(self) -> Dict[str, Any]:
        """Get file handling configuration"""
        config = self.get('file_handling', default={})
        
        # Provide sensible defaults if not configured
        defaults = {
            'default_directories': ['./data'],
            'file_patterns': ['*.csv'],
            'max_files_shown': 20,
            'sort_by': 'modified_date',
            'show_all_option': True
        }
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Expand home directory paths
        expanded_dirs = []
        for directory in config['default_directories']:
            expanded_path = os.path.expanduser(directory)
            expanded_dirs.append(expanded_path)
        config['default_directories'] = expanded_dirs
        
        return config
    
    def print_config(self, hide_sensitive: bool = True):
        """Print current configuration (for debugging)"""
        import json
        config_copy = self._config.copy()
        
        if hide_sensitive:
            # Hide sensitive information
            if 'gcp' in config_copy and 'credentials_path' in config_copy['gcp']:
                if config_copy['gcp']['credentials_path']:
                    config_copy['gcp']['credentials_path'] = '***HIDDEN***'
        
        print("Current Configuration:")
        print(json.dumps(config_copy, indent=2))


# Global configuration instance
_config_instance = None

def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_file: Path to config file (only used on first call)
    
    Returns:
        Config: Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance

def reload_config(config_file: Optional[str] = None):
    """Force reload of configuration (useful for testing)"""
    global _config_instance
    _config_instance = Config(config_file)
    return _config_instance


if __name__ == "__main__":
    # Example usage and testing
    config = get_config()
    
    print("=== Configuration Test ===")
    config.print_config()
    
    print(f"\nGCP Bucket: {config.get('gcp', 'bucket_name')}")
    print(f"Data Dir: {config.get('paths', 'data_dir')}")
    print(f"ML Batch Size: {config.get('ml', 'batch_size')}")
    
    print(f"\nValidation: {'✅ PASS' if config.validate_required() else '❌ FAIL'}")