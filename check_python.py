#!/usr/bin/env python3
"""
Python Environment Checker
Shows which Python interpreter is being used and environment details
"""

import sys
import os
import platform
from pathlib import Path

def main():
    print("🐍 Python Environment Information")
    print("=" * 50)
    
    # Python executable path
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Python Version Info: {sys.version_info}")
    
    print("\n📂 Environment Details")
    print("-" * 30)
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Script Location: {Path(__file__).parent}")
    
    print("\n📍 Python Path")
    print("-" * 30)
    for i, path in enumerate(sys.path, 1):
        print(f"{i:2d}. {path}")
    
    print("\n🔧 Virtual Environment Check")
    print("-" * 30)
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in a virtual environment")
        print(f"   Virtual Env: {sys.prefix}")
        print(f"   Base Python: {sys.base_prefix}")
    else:
        print("⚠️  Running in system Python (not a virtual environment)")
    
    # Check for common virtual env indicators
    virtual_env = os.environ.get('VIRTUAL_ENV')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if virtual_env:
        print(f"   VIRTUAL_ENV: {virtual_env}")
    if conda_env:
        print(f"   CONDA_ENV: {conda_env}")
    
    print("\n🏠 Project Directory Check")
    print("-" * 30)
    project_files = ['config.example.yaml', 'requirements.txt', 'setup.py', 'src/']
    for file in project_files:
        file_path = Path(file)
        status = "✅ Found" if file_path.exists() else "❌ Missing"
        print(f"   {status}: {file}")

if __name__ == "__main__":
    main()