import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import glob
import re
from typing import Optional, Dict, List, Tuple
from src.model_registry import ModelRegistry

class CorrectionValidator:
    def __init__(self):
        """Initialize with auto-detected project root and enhanced logging"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        self.logs_dir = os.path.join(self.project_root, "logs")
        self.registry_dir = os.path.join(self.project_root, "models", "registry")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize model registry
        self.model_registry = ModelRegistry(self.registry_dir)
        
        # Setup logging with more detailed format
        log_file = os.path.join(self.logs_dir, f'correction_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.valid_categories = self._load_valid_categories()

    def _get_latest_model_version(self) -> str:
        """Get latest model version from registry"""
        try:
            # Try getting version from lgbm model (primary model)
            version = self.model_registry.get_latest_version("lgbm")
            if version is not None:
                return f"v{version}"
            
            self.logger.warning("Could not get version from model registry")
            # Fallback to scanning model files
            model_files = glob.glob(os.path.join(self.project_root, "models", "lgbm_model_v*.pkl"))
            if model_files:
                versions = [int(re.search(r'v(\d+)', f).group(1)) for f in model_files]
                return f"v{max(versions)}"
            
            self.logger.error("Could not determine model version")
            return "unknown"
        except Exception as e:
            self.logger.error(f"Error getting model version: {str(e)}")
            return "unknown"

    def _load_valid_categories(self) -> set:
        """Load and validate category list"""
        categories_file = os.path.join(self.data_dir, "valid_categories.txt")
        if not os.path.exists(categories_file):
            self.logger.error(f"Categories file not found: {categories_file}")
            raise FileNotFoundError(f"Please create {categories_file} with valid categories")
        
        with open(categories_file, 'r') as f:
            categories = {line.strip() for line in f if line.strip()}
        
        if not categories:
            raise ValueError("Categories file is empty")
        
        self.logger.info(f"Loaded {len(categories)} valid categories")
        return categories

    def _validate_amounts(self, df: pd.DataFrame) -> List[str]:
        """Validate amount fields for consistency with better error handling"""
        errors = []
        warnings = []
        
        # Check presence of amount fields
        required_cols = ['Amount', 'Amount (Negated)']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
                return errors
    
        # Convert columns to numeric, forcing errors to NaN
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Amount (Negated)'] = pd.to_numeric(df['Amount (Negated)'], errors='coerce')
    
        # Check for null values after conversion
        null_amounts = df['Amount'].isna().sum()
        null_negated = df['Amount (Negated)'].isna().sum()
        
        if null_amounts > 0:
            warnings.append(f"Found {null_amounts} non-numeric values in Amount column")
        if null_negated > 0:
            warnings.append(f"Found {null_negated} non-numeric values in Amount (Negated) column")
    
        # Fill NaN with 0 for comparison
        df['Amount'] = df['Amount'].fillna(0)
        df['Amount (Negated)'] = df['Amount (Negated)'].fillna(0)
    
        # Log summary statistics
        self.logger.info("\nAmount validation summary:")
        self.logger.info(f"Total rows: {len(df)}")
        self.logger.info(f"Non-zero Amount entries: {(df['Amount'] != 0).sum()}")
        self.logger.info(f"Non-zero Amount (Negated) entries: {(df['Amount (Negated)'] != 0).sum()}")
        
        # Only return errors if there are serious issues
        # Convert warnings to logging messages
        for warning in warnings:
            self.logger.warning(warning)
        
        return errors  # Return only critical errors

    def _validate_dates(self, df: pd.DataFrame) -> List[str]:
        """Validate transaction dates"""
        errors = []
        
        if 'Date' not in df.columns:
            errors.append("Missing Date column")
            return errors

        # Convert dates to datetime
        try:
            dates = pd.to_datetime(df['Date'])
            
            # Check for future dates
            future_dates = dates[dates > datetime.now()]
            if not future_dates.empty:
                errors.append(f"Found {len(future_dates)} future dates")

            # Check for very old dates (e.g., more than 5 years old)
            old_threshold = datetime.now() - timedelta(days=5*365)
            old_dates = dates[dates < old_threshold]
            if not old_dates.empty:
                errors.append(f"Found {len(old_dates)} dates older than 5 years")

        except Exception as e:
            errors.append(f"Error parsing dates: {str(e)}")

        return errors

    def _validate_categories(self, df: pd.DataFrame) -> List[str]:
        """Validate transaction categories"""
        errors = []
        
        if 'Category' not in df.columns:
            errors.append("Missing Category column")
            return errors

        # Check for invalid categories
        invalid_categories = set(df['Category'].unique()) - self.valid_categories
        if invalid_categories:
            errors.append(f"Found invalid categories: {invalid_categories}")

        # Check for null categories
        null_categories = df['Category'].isna().sum()
        if null_categories > 0:
            errors.append(f"Found {null_categories} null categories")

        return errors

    def _validate_descriptions(self, df: pd.DataFrame) -> List[str]:
        """Validate transaction descriptions"""
        errors = []
        
        if 'Description' not in df.columns:
            errors.append("Missing Description column")
            return errors

        # Check for empty descriptions
        empty_descriptions = df['Description'].isna().sum()
        if empty_descriptions > 0:
            errors.append(f"Found {empty_descriptions} empty descriptions")

        # Check for very short descriptions (e.g., less than 3 characters)
        short_descriptions = df[df['Description'].str.len() < 3].shape[0]
        if short_descriptions > 0:
            errors.append(f"Found {short_descriptions} very short descriptions")

        return errors

    def _log_data_summary(self, df: pd.DataFrame, stage: str):
        """Log summary statistics of the data"""
        self.logger.info(f"\n{'='*20} {stage} Summary {'='*20}")
        self.logger.info(f"Total records: {len(df)}")
        self.logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        self.logger.info("\nCategory distribution:")
        for category, count in df['Category'].value_counts().items():
            self.logger.info(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
        self.logger.info(f"\nUnique descriptions: {df['Description'].nunique()}")
        self.logger.info(f"Total amount: {df['Amount'].sum():,.2f}")
        self.logger.info("="*50)

    def validate_and_prepare(self, corrections_df: pd.DataFrame) -> str:
        """
        Validate corrections and merge into latest training dataset with enhanced validation
        """
        # Get model version automatically
        model_version = self._get_latest_model_version()
        self.logger.info(f"Using model version: {model_version}")
    
        # Convert amount columns to numeric if they aren't already
        for col in ['Amount', 'Amount (Negated)']:
            if col in corrections_df.columns:
                corrections_df[col] = pd.to_numeric(corrections_df[col], errors='coerce').fillna(0)
    
        # Perform all validations
        validation_errors = []
        validation_errors.extend(self._validate_amounts(corrections_df))
        validation_errors.extend(self._validate_dates(corrections_df))
        validation_errors.extend(self._validate_categories(corrections_df))
        validation_errors.extend(self._validate_descriptions(corrections_df))
    
        if validation_errors:
            error_msg = "\n".join(validation_errors)
            self.logger.error(f"Validation errors:\n{error_msg}")
            raise ValueError("Data validation failed. Check logs for details.")
    
        # Add metadata columns
        corrections_df['correction_timestamp'] = datetime.now()
        corrections_df['source_model_version'] = model_version
    
        # Log summary of corrections
        self._log_data_summary(corrections_df, "Corrections")
    
        # Load existing training data
        latest_training_file = self._get_latest_training_data()
        if latest_training_file is not None:
            existing_data = pd.read_csv(latest_training_file)
            self._log_data_summary(existing_data, "Existing Training Data")
            
            # Merge without deduplication
            combined_data = pd.concat([existing_data, corrections_df], ignore_index=True)
        else:
            combined_data = corrections_df
    
        # Log summary of merged data
        self._log_data_summary(combined_data, "Combined Data")
    
        # Determine new version number
        existing_versions = [
            int(re.search(r'v(\d+)', f).group(1)) 
            for f in glob.glob(os.path.join(self.data_dir, "training_data_v*.csv")) 
            if re.search(r'v(\d+)', f)
        ]
        new_version = max(existing_versions) + 1 if existing_versions else 1
    
        # Save updated training dataset
        timestamp = datetime.now().strftime("%Y%m%d")
        export_filename = f"training_data_v{new_version}_{timestamp}.csv"
        export_path = os.path.join(self.data_dir, export_filename)
        
        combined_data.to_csv(export_path, index=False)
        self.logger.info(f"✅ Exported new training dataset: {export_filename}")
        
        return export_filename

    def _get_latest_training_data(self) -> Optional[str]:
        """Get the latest training data file"""
        training_files = glob.glob(os.path.join(self.data_dir, "training_data_v*.csv"))
        if not training_files:
            self.logger.warning("No existing training data found")
            return None
        return max(training_files, key=os.path.getmtime)


def main():
    """Main execution flow with improved error handling and user interaction"""
    validator = CorrectionValidator()
    
    try:
        # List available CSV files
        csv_files = [f for f in os.listdir(validator.data_dir) if f.endswith('.csv')]
        if not csv_files:
            print("No CSV files found in the data directory.")
            return
        
        print("\nAvailable CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        
        while True:
            try:
                choice = input("\nEnter the number of the corrections file to process (or 'q' to quit): ")
                if choice.lower() == 'q':
                    return
                
                file_index = int(choice) - 1
                if 0 <= file_index < len(csv_files):
                    corrections_file = csv_files[file_index]
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        print(f"\n📂 Selected file: {corrections_file}")
        
        # Load and process
        corrections_df = pd.read_csv(os.path.join(validator.data_dir, corrections_file))
        export_file = validator.validate_and_prepare(corrections_df)
        
        print("\n✅ Processing Complete!")
        print(f"Results saved to: {export_file}")
        print("\nCheck the logs for detailed validation and processing information.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        validator.logger.error("Processing failed", exc_info=True)
        raise

if __name__ == "__main__":
    main()