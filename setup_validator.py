#!/usr/bin/env python3
"""
Setup Validator for Expense Categorization ML Pipeline

This script validates that all prerequisites are properly configured
and provides actionable guidance for fixing any issues found.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m' 
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class SetupValidator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        
    def print_status(self, message: str, status: str, details: str = None):
        """Print formatted status message"""
        status_symbols = {
            'pass': f"{Colors.GREEN}✅{Colors.END}",
            'fail': f"{Colors.RED}❌{Colors.END}",
            'warn': f"{Colors.YELLOW}⚠️{Colors.END}",
            'info': f"{Colors.BLUE}ℹ️{Colors.END}"
        }
        
        symbol = status_symbols.get(status, "")
        print(f"{symbol} {message}")
        
        if details and (self.verbose or status == 'fail'):
            print(f"   {details}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_status(
                f"Python version: {version.major}.{version.minor}.{version.micro}", 
                'pass'
            )
            return True
        else:
            self.print_status(
                f"Python version: {version.major}.{version.minor}.{version.micro}", 
                'fail',
                "Requires Python 3.8 or higher"
            )
            self.issues.append("Update Python to version 3.8 or higher")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required Python packages are installed"""
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            self.print_status("Requirements file", 'fail', "requirements.txt not found")
            self.issues.append("Create requirements.txt file")
            return False
            
        # Read requirements
        with open(requirements_file) as f:
            lines = f.readlines()
        
        required_packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before ==, >=, etc.)
                package = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                required_packages.append((package, line))
        
        missing_packages = []
        installed_packages = []
        
        for package, requirement in required_packages:
            try:
                __import__(package.replace('-', '_'))
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(requirement)
        
        if missing_packages:
            self.print_status(
                f"Python packages: {len(installed_packages)}/{len(required_packages)} installed", 
                'fail',
                f"Missing: {', '.join(missing_packages)}"
            )
            self.issues.append(f"Install missing packages: pip install {' '.join(missing_packages)}")
            return False
        else:
            self.print_status(
                f"Python packages: All {len(required_packages)} packages installed", 
                'pass'
            )
            return True
    
    def check_configuration(self) -> bool:
        """Check configuration setup"""
        config_example = self.project_root / "config.example.yaml"
        config_file = self.project_root / "config.yaml"
        
        # Check if example exists
        if not config_example.exists():
            self.print_status("Configuration template", 'fail', "config.example.yaml not found")
            self.issues.append("Ensure config.example.yaml exists")
            return False
        
        # Check if user config exists
        if not config_file.exists():
            self.print_status("User configuration", 'fail', "config.yaml not found")
            self.issues.append("Copy config.example.yaml to config.yaml and customize it")
            return False
        
        # Try to load and validate configuration
        try:
            sys.path.insert(0, str(self.project_root))
            from src.config import get_config
            
            config = get_config()
            
            # Check GCP credentials
            gcp_config = config.get_gcp_config()
            creds_path = gcp_config.get('credentials_path')
            
            if not creds_path:
                self.print_status("GCP credentials path", 'fail', "Not configured in config.yaml")
                self.issues.append("Set gcp.credentials_path in config.yaml")
                return False
            
            if not Path(creds_path).exists():
                self.print_status("GCP credentials file", 'fail', f"File not found: {creds_path}")
                self.issues.append(f"Create GCP service account key file at {creds_path}")
                return False
            
            # Check bucket name
            bucket_name = gcp_config.get('bucket_name')
            if not bucket_name or bucket_name == 'your-expense-ml-models-backup':
                self.print_status("GCS bucket name", 'warn', "Using default/example bucket name")
                self.warnings.append("Update gcp.bucket_name with your own bucket")
            
            self.print_status("Configuration", 'pass', "config.yaml loaded successfully")
            return True
            
        except Exception as e:
            self.print_status("Configuration validation", 'fail', str(e))
            self.issues.append(f"Fix configuration error: {e}")
            return False
    
    def check_data_directory(self) -> bool:
        """Check data directory and required files"""
        data_dir = self.project_root / "data"
        
        if not data_dir.exists():
            self.print_status("Data directory", 'fail', "data/ directory not found")
            self.issues.append("Create data/ directory")
            return False
        
        # Check for valid categories file
        categories_file = data_dir / "valid_categories.txt"
        if not categories_file.exists():
            self.print_status("Categories file", 'fail', "data/valid_categories.txt not found")
            self.issues.append("Create data/valid_categories.txt with your expense categories")
            return False
        
        # Check categories content
        with open(categories_file) as f:
            categories = [line.strip() for line in f if line.strip()]
        
        if len(categories) < 3:
            self.print_status("Categories content", 'warn', f"Only {len(categories)} categories defined")
            self.warnings.append("Add more categories to data/valid_categories.txt for better ML performance")
        else:
            self.print_status(f"Categories: {len(categories)} defined", 'pass')
        
        # Check for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            self.print_status(f"Data files: {len(csv_files)} CSV files found", 'pass')
        else:
            self.print_status("Data files", 'warn', "No CSV files found in data/")
            self.warnings.append("Add transaction CSV files to data/ directory")
        
        return True
    
    def check_directories(self) -> bool:
        """Check that required directories exist"""
        required_dirs = ['models', 'logs', 'cache']
        all_exist = True
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.print_status(f"{dir_name}/ directory", 'pass')
            else:
                self.print_status(f"{dir_name}/ directory", 'fail', "Directory will be created automatically")
                # This is not a blocking issue as directories are created automatically
        
        return True
    
    def check_model_registry(self) -> bool:
        """Check if models are available"""
        models_dir = self.project_root / "models"
        registry_dir = models_dir / "registry"
        
        if not registry_dir.exists():
            self.print_status("Model registry", 'warn', "No trained models found")
            self.warnings.append("Train models using auto_model_ensemble.py after adding training data")
            return True
        
        registry_file = registry_dir / "model_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    registry = json.load(f)
                
                model_count = len(registry.get('models', {}))
                self.print_status(f"Trained models: {model_count} model types registered", 'pass')
            except Exception as e:
                self.print_status("Model registry", 'warn', f"Registry file corrupted: {e}")
                self.warnings.append("Registry file may need regeneration")
        
        return True
    
    def test_imports(self) -> bool:
        """Test that main modules can be imported"""
        test_modules = [
            'src.config',
            'src.model_storage', 
            'src.model_registry',
            'src.transaction_types'
        ]
        
        sys.path.insert(0, str(self.project_root))
        
        failed_imports = []
        for module in test_modules:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
        
        if failed_imports:
            self.print_status("Module imports", 'fail', "; ".join(failed_imports))
            self.issues.append("Fix import errors - ensure all dependencies are installed")
            return False
        else:
            self.print_status("Module imports", 'pass', "All core modules importable")
            return True
    
    def run_validation(self) -> bool:
        """Run all validation checks"""
        print(f"{Colors.BOLD}🔍 Expense ML Pipeline Setup Validation{Colors.END}")
        print("=" * 50)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Configuration", self.check_configuration),
            ("Data Setup", self.check_data_directory),
            ("Directories", self.check_directories),
            ("Model Registry", self.check_model_registry),
            ("Module Imports", self.test_imports)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            if self.verbose:
                print(f"\n{Colors.BLUE}Checking {check_name}...{Colors.END}")
            
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.print_status(f"{check_name}", 'fail', f"Check failed: {e}")
                all_passed = False
        
        return all_passed
    
    def print_summary(self, all_passed: bool):
        """Print validation summary and next steps"""
        print("\n" + "=" * 50)
        
        if all_passed and not self.issues:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ Setup Validation PASSED{Colors.END}")
            print("🎉 Your expense categorization system is ready to use!")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Minor Recommendations:{Colors.END}")
                for warning in self.warnings:
                    print(f"  • {warning}")
                    
            print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
            print("1. Add transaction CSV files to data/ directory")
            print("2. Run: python src/batch_predict_ensemble.py --help")
            print("3. Start with: python src/batch_predict_ensemble.py --list")
            
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Setup Validation FAILED{Colors.END}")
            print("The following issues must be resolved:\n")
            
            for i, issue in enumerate(self.issues, 1):
                print(f"{Colors.RED}{i}.{Colors.END} {issue}")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.END}")
                for warning in self.warnings:
                    print(f"  • {warning}")
            
            print(f"\n{Colors.BOLD}Fix these issues and run validation again:{Colors.END}")
            print("python setup_validator.py")

def main():
    parser = argparse.ArgumentParser(
        description="Validate expense categorization ML pipeline setup"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--fix',
        action='store_true', 
        help='Attempt to automatically fix common issues'
    )
    
    args = parser.parse_args()
    
    validator = SetupValidator(verbose=args.verbose)
    
    if args.fix:
        print("🔧 Auto-fix mode not yet implemented")
        print("Please resolve issues manually using the guidance provided")
        return
    
    success = validator.run_validation()
    validator.print_summary(success)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()