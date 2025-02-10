import os
import sys
from pathlib import Path

print("\nDEBUG INFO:")

# Get project root
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Current directory: {os.getcwd()}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Python path before: {sys.path}")

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))
print(f"\nPython path after: {sys.path}")

# Try imports
try:
    from src.model_storage import ModelStorage
    print("\n✅ Successfully imported ModelStorage")
except ImportError as e:
    print(f"\n❌ Failed to import ModelStorage: {e}")

print("\nDirectory contents:")
print(f"src directory exists: {(PROJECT_ROOT / 'src').exists()}")
print(f"Files in src: {os.listdir(PROJECT_ROOT / 'src')}")