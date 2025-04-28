import os
import sys
import glob
from pathlib import Path

# Get project root directory (adjust if needed)
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"

# Set up path for importing from src directory
sys.path.insert(0, str(PROJECT_ROOT))

# Import the TransactionSource enum
from src.transaction_types import TransactionSource

# Find latest credit card training data
source_prefix = f"training_data_{TransactionSource.CREDIT_CARD.value}_"
training_files = glob.glob(str(DATA_DIR / f"{source_prefix}*.csv"))

if training_files:
    latest_file = max(training_files, key=os.path.getmtime)
    print(f"\nLatest credit card training data file:")
    print(f"- {os.path.basename(latest_file)}")
    print(f"- Last modified: {os.path.getmtime(latest_file)}")
    print(f"- Full path: {latest_file}")
else:
    print("\nNo credit card training data files found!")

# Find latest credit card model version (via file scanning)
MODELS_DIR = PROJECT_ROOT / "models"
model_files = glob.glob(str(MODELS_DIR / f"credit_card_lgbm_model_v*.pkl"))

if model_files:
    import re
    versions = [int(re.search(r'v(\d+)', f).group(1)) for f in model_files if re.search(r'v(\d+)', f)]
    if versions:
        latest_version = max(versions)
        print(f"\nLatest credit card model version: v{latest_version}")
        print(f"- Based on file: {os.path.basename([f for f in model_files if f'v{latest_version}' in f][0])}")
    else:
        print("\nCouldn't determine model version from filenames")
else:
    print("\nNo credit card model files found!")