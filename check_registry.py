import os
import sys
import glob
import json
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_FILE = REGISTRY_DIR / "model_registry.json"

# Set up path for importing from src directory
sys.path.insert(0, str(PROJECT_ROOT))

# Import the TransactionSource enum and ModelRegistry
from src.transaction_types import TransactionSource
from src.model_registry import ModelRegistry

# Initialize the model registry
registry = ModelRegistry(REGISTRY_DIR)

# Create a function to check both methods
def check_latest_version(source_value):
    source = TransactionSource(source_value)
    print(f"\n==== Checking {source.value.upper()} ====")
    
    # METHOD 1: Registry-based approach
    print("\n1. USING MODEL REGISTRY:")
    model_name = f"{source.value}_lgbm"
    version = registry.get_latest_version(model_name)
    if version is not None:
        print(f"Latest model version from registry: v{version}")
        model_info = registry.get_model_info(model_name, version)
        if model_info:
            print(f"Model info from registry: {json.dumps(model_info, indent=2)}")
        else:
            print("No model info found in registry")
    else:
        print(f"No model version found in registry for {model_name}")
    
    # METHOD 2: File-based approach
    print("\n2. USING FILE SCANNING:")
    
    # Check latest model version via file scanning
    model_files = glob.glob(str(MODELS_DIR / f"{source.value}_lgbm_model_v*.pkl"))
    if model_files:
        import re
        versions = [int(re.search(r'v(\d+)', f).group(1)) for f in model_files if re.search(r'v(\d+)', f)]
        if versions:
            latest_version = max(versions)
            print(f"Latest model version from files: v{latest_version}")
            file = [f for f in model_files if f'v{latest_version}' in f][0]
            print(f"Model file: {os.path.basename(file)}")
        else:
            print("Couldn't determine model version from filenames")
    else:
        print(f"No model files found for {source.value}")
    
    # Check latest training data via file scanning
    source_prefix = f"training_data_{source.value}_"
    training_files = glob.glob(str(DATA_DIR / f"{source_prefix}*.csv"))
    if training_files:
        latest_file = max(training_files, key=os.path.getmtime)
        print(f"\nLatest training data file: {os.path.basename(latest_file)}")
    else:
        print(f"\nNo training data files found for {source.value}")

# Check both sources
check_latest_version('credit_card')
check_latest_version('household')

# Check if registry is accessible
print("\n==== REGISTRY FILE CHECK ====")
if REGISTRY_FILE.exists():
    print(f"Registry file exists at: {REGISTRY_FILE}")
    try:
        with open(REGISTRY_FILE, 'r') as f:
            registry_data = json.load(f)
            print(f"Registry contains {len(registry_data.get('models', {}))} model entries")
            print(f"Models in registry: {list(registry_data.get('models', {}).keys())}")
    except Exception as e:
        print(f"Error reading registry file: {str(e)}")
else:
    print(f"Registry file does not exist at {REGISTRY_FILE}")