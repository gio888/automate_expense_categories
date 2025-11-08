import os
import sys
from pathlib import Path

# Setup the path BEFORE any other imports
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import glob
import re
from typing import Optional, Dict, List, Tuple
from src.model_registry import ModelRegistry
from src.transaction_types import TransactionSource  # Import the TransactionSource enum
from src.utils.excel_processor import ExcelProcessor, is_excel_file

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

    def _get_latest_model_version(self, source: TransactionSource) -> str:
        """Get latest model version from registry for the specific source"""
        try:
            # Try getting version from source-specific lgbm model
            source_model_name = f"{source.value}_lgbm"
            version = self.model_registry.get_latest_version(source_model_name)
            if version is not None:
                return f"v{version}"
            
            self.logger.warning(f"Could not get version from model registry for {source.value}")
            # Fallback to scanning model files
            model_files = glob.glob(os.path.join(self.project_root, "models", f"{source.value}_lgbm_model_v*.pkl"))
            if model_files:
                versions = [int(re.search(r'v(\d+)', f).group(1)) for f in model_files]
                return f"v{max(versions)}"
            
            self.logger.error(f"Could not determine model version for {source.value}")
            return "unknown"
        except Exception as e:
            self.logger.error(f"Error getting model version for {source.value}: {str(e)}")
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
    
        # Handle empty strings and convert to numeric more carefully
        # First, replace empty strings with NaN, then convert to numeric
        df_copy = df.copy()  # Work with a copy to avoid modifying original
        
        for col in required_cols:
            # Replace empty strings with NaN before numeric conversion
            df_copy[col] = df_copy[col].replace('', pd.NA)
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
        # Check for null values after conversion
        null_amounts = df_copy['Amount'].isna().sum()
        null_negated = df_copy['Amount (Negated)'].isna().sum()
        
        if null_amounts > 0:
            self.logger.info(f"Found {null_amounts} empty/non-numeric values in Amount column")
        if null_negated > 0:
            self.logger.info(f"Found {null_negated} empty/non-numeric values in Amount (Negated) column")
    
        # Fill NaN with 0 for statistics
        df_copy['Amount'] = df_copy['Amount'].fillna(0)
        df_copy['Amount (Negated)'] = df_copy['Amount (Negated)'].fillna(0)
    
        # Check if all amounts are zero (which would be problematic)
        total_amount_entries = (df_copy['Amount'] != 0).sum()
        total_negated_entries = (df_copy['Amount (Negated)'] != 0).sum()
        
        if total_amount_entries == 0 and total_negated_entries == 0:
            errors.append("All amount fields are zero or empty - no valid transactions found")
    
        # Log summary statistics
        self.logger.info("\nAmount validation summary:")
        self.logger.info(f"Total rows: {len(df_copy)}")
        self.logger.info(f"Non-zero Amount entries: {total_amount_entries}")
        self.logger.info(f"Non-zero Amount (Negated) entries: {total_negated_entries}")
        
        return errors  # Return only critical errors

    def _validate_dates(self, df: pd.DataFrame) -> List[str]:
        """Validate transaction dates"""
        errors = []
    
        if 'Date' not in df.columns:
            errors.append("Missing Date column")
            return errors

        # Convert 'Date' column to datetime and handle errors
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            # Check for invalid dates that could not be converted
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"⚠️ Found {invalid_dates} invalid date(s), setting them to NaT.")

        except Exception as e:
            errors.append(f"Error parsing dates: {str(e)}")

        return errors

    def _validate_categories(self, df: pd.DataFrame) -> List[str]:
        """Validate transaction categories"""
        errors = []
        
        if 'Category' not in df.columns:
            # Check for alternative category column names
            if 'Corrected Category' in df.columns:
                self.logger.info("Found 'Corrected Category' column, will use it as 'Category'")
                df['Category'] = df['Corrected Category']
            elif 'predicted_category' in df.columns:
                self.logger.info("Found 'predicted_category' column, will use it as 'Category'")
                df['Category'] = df['predicted_category']
            else:
                errors.append("Missing Category column (tried: 'Category', 'Corrected Category', 'predicted_category')")
                return errors

        # Check for invalid categories - now convert to warning instead of error
        invalid_categories = set(df['Category'].unique()) - self.valid_categories
        if invalid_categories:
            self.logger.warning(f"Found categories not in valid list: {invalid_categories}")
            self.logger.warning("These categories will be used but may need review")
            # Don't add to errors - allow unknown categories to pass with warning

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

        # Check for very short descriptions (e.g., less than 3 characters) - now a warning instead of error
        short_descriptions = df[df['Description'].str.len() < 3].shape[0]
        if short_descriptions > 0:
            self.logger.warning(f"Found {short_descriptions} very short descriptions - these may be abbreviations")
            # Don't add to errors - allow short descriptions to pass

        return errors

    def _validate_transaction_source(self, df: pd.DataFrame) -> List[str]:
        """Validate transaction source column"""
        errors = []
        
        if 'transaction_source' not in df.columns:
            errors.append("Missing transaction_source column. Please add a 'transaction_source' column with values 'household' or 'credit_card'")
            return errors
            
        # Check that all values are valid
        valid_sources = [source.value for source in TransactionSource]
        invalid_sources = set(df['transaction_source'].unique()) - set(valid_sources)
        
        if invalid_sources:
            errors.append(f"Found invalid transaction sources: {invalid_sources}. Valid values are: {valid_sources}")
            
        # Check for null sources
        null_sources = df['transaction_source'].isna().sum()
        if null_sources > 0:
            errors.append(f"Found {null_sources} rows with missing transaction_source")
            
        return errors

    def _log_data_summary(self, df: pd.DataFrame, stage: str):
        """Log summary statistics of the data"""
        self.logger.info(f"\n{'='*20} {stage} Summary {'='*20}")
        self.logger.info(f"Total records: {len(df)}")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            self.logger.info(f"Date range: {df['Date'].dropna().min()} to {df['Date'].dropna().max()}")
            
        if 'Category' in df.columns:
            self.logger.info("\nCategory distribution:")
            for category, count in df['Category'].value_counts().items():
                self.logger.info(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
                
        if 'transaction_source' in df.columns:
            self.logger.info("\nTransaction source distribution:")
            for source, count in df['transaction_source'].value_counts().items():
                self.logger.info(f"  {source}: {count} ({count/len(df)*100:.1f}%)")
                
        if 'Description' in df.columns:
            self.logger.info(f"\nUnique descriptions: {df['Description'].nunique()}")
            
        if 'Amount' in df.columns:
            self.logger.info(f"Total amount: {df['Amount'].sum():,.2f}")
            
        self.logger.info("="*50)

    def _get_latest_training_data(self, source: TransactionSource) -> Optional[str]:
        """Get the latest training data file for a specific source"""
        # Look for source-specific training data files
        source_prefix = f"training_data_{source.value}_"
        training_files = glob.glob(os.path.join(self.data_dir, f"{source_prefix}*.csv"))
        
        if not training_files:
            self.logger.warning(f"No existing training data found for source: {source.value}")
            return None
            
        return max(training_files, key=os.path.getmtime)

    def _ensure_columns_match(self, corrections_df: pd.DataFrame, existing_data: pd.DataFrame, source: TransactionSource) -> pd.DataFrame:
        """
        Ensure the corrections dataframe has all the same columns as the existing training data
        """
        # Make a copy to avoid modifying the original
        formatted_df = corrections_df.copy()

        # First, map web interface column names to training data format
        column_mappings = {
            'DATE': 'Date',
            'DESCRIPTION': 'Description',
            'DEBIT': 'Amount (Negated)',
            'CREDIT': 'Amount'
        }

        for web_col, train_col in column_mappings.items():
            if web_col in formatted_df.columns and train_col not in formatted_df.columns:
                self.logger.info(f"Mapping '{web_col}' to '{train_col}'")
                formatted_df[train_col] = formatted_df[web_col]
                # Keep the original column for reference

        # Handle alternative category column names
        if 'Category' not in formatted_df.columns:
            if 'Corrected Category' in formatted_df.columns:
                self.logger.info("Using 'Corrected Category' as 'Category'")
                formatted_df['Category'] = formatted_df['Corrected Category']
                formatted_df.drop('Corrected Category', axis=1, inplace=True)
            elif 'predicted_category' in formatted_df.columns:
                self.logger.info("Using 'predicted_category' as 'Category'")
                formatted_df['Category'] = formatted_df['predicted_category']
                # Keep predicted_category for reference

        # Get missing columns
        existing_columns = set(existing_data.columns)
        corrections_columns = set(formatted_df.columns)
        missing_columns = existing_columns - corrections_columns

        self.logger.info(f"Adding {len(missing_columns)} missing columns to match training data structure")
        
        # Add each missing column with appropriate defaults
        for col in missing_columns:
            if col == 'Account':
                if source.value == 'credit_card':
                    formatted_df[col] = "Liabilities:Credit Card"
                else:
                    formatted_df[col] = "Assets:Current Assets:Cash Local:Cash for Groceries"
                self.logger.info(f"Added '{col}' with default value for {source.value}")
                
            elif col == 'Amount (Raw)':
                if 'Amount' in formatted_df.columns:
                    formatted_df[col] = formatted_df['Amount']
                else:
                    formatted_df[col] = 0
                self.logger.info(f"Added '{col}' based on 'Amount' column")
                
            elif col == 'Entered':
                formatted_df[col] = True
                self.logger.info(f"Added '{col}' with default True")

            elif col == 'Reconciled':
                formatted_df[col] = False
                self.logger.info(f"Added '{col}' with default False")
                
            elif col == 'correction_timestamp' or col == 'source_model_version':
                # These will be added later, skip for now
                pass
                
            else:
                # Generic handling for any other columns
                formatted_df[col] = None
                self.logger.info(f"Added '{col}' with default None")
        
        # Convert column types to match expected types
        for col in formatted_df.columns:
            if col in existing_data.columns:
                try:
                    # Try to convert the column to the same dtype as in existing data
                    orig_dtype = existing_data[col].dtype
                    if orig_dtype != formatted_df[col].dtype:
                        self.logger.info(f"Converting '{col}' from {formatted_df[col].dtype} to {orig_dtype}")
                        formatted_df[col] = formatted_df[col].astype(orig_dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert '{col}' to dtype {existing_data[col].dtype}: {str(e)}")
        
        return formatted_df

    def validate_and_prepare(self, corrections_df: pd.DataFrame) -> str:
        """
        Validate corrections and merge into latest training dataset with source-specific handling
        """
        # First validate transaction_source column
        source_errors = self._validate_transaction_source(corrections_df)
        if source_errors:
            error_msg = "\n".join(source_errors)
            self.logger.error(f"Transaction source validation errors:\n{error_msg}")
            raise ValueError("Transaction source validation failed. Check logs for details.")
        
        # Group corrections by transaction source and process each group separately
        result_files = []
        
        for source_value in corrections_df['transaction_source'].unique():
            try:
                source = TransactionSource(source_value)
                self.logger.info(f"\nProcessing corrections for source: {source.value}")
                
                # Filter corrections for this source
                source_corrections = corrections_df[corrections_df['transaction_source'] == source.value].copy()
                self.logger.info(f"Found {len(source_corrections)} corrections for {source.value}")
                
                # Get model version for this source
                model_version = self._get_latest_model_version(source)
                self.logger.info(f"Using model version for {source.value}: {model_version}")
            
                # Convert amount columns to numeric if they aren't already
                for col in ['Amount', 'Amount (Negated)']:
                    if col in source_corrections.columns:
                        source_corrections[col] = pd.to_numeric(source_corrections[col], errors='coerce').fillna(0)
            
                # Load existing training data for this source
                latest_training_file = self._get_latest_training_data(source)
                existing_data = None
                
                if latest_training_file is not None:
                    # Handle both CSV and Excel files
                    if is_excel_file(latest_training_file):
                        excel_processor = ExcelProcessor()
                        existing_data = excel_processor.process_excel_file(latest_training_file)
                    else:
                        existing_data = pd.read_csv(latest_training_file)
                    self._log_data_summary(existing_data, f"Existing {source.value.capitalize()} Training Data")
                    
                    # Check if transaction_source column exists in existing data
                    if 'transaction_source' not in existing_data.columns:
                        self.logger.warning(f"Adding missing transaction_source column to existing {source.value} data")
                        existing_data['transaction_source'] = source.value
                    
                    # Format corrections to match existing data structure
                    source_corrections = self._ensure_columns_match(source_corrections, existing_data, source)
                
                # Now perform validations on the corrected data
                validation_errors = []
                validation_errors.extend(self._validate_amounts(source_corrections))
                validation_errors.extend(self._validate_dates(source_corrections))
                validation_errors.extend(self._validate_categories(source_corrections))
                validation_errors.extend(self._validate_descriptions(source_corrections))
            
                if validation_errors:
                    error_msg = "\n".join(validation_errors)
                    self.logger.error(f"Validation errors for {source.value}:\n{error_msg}")
                    raise ValueError(f"Data validation for {source.value} failed. Check logs for details.")
            
                # Add metadata columns
                source_corrections['correction_timestamp'] = datetime.now()
                source_corrections['source_model_version'] = model_version
            
                # Log summary of corrections
                self._log_data_summary(source_corrections, f"{source.value.capitalize()} Corrections")
            
                # Merge with existing data or use corrections as new training data
                if existing_data is not None:
                    combined_data = pd.concat([existing_data, source_corrections], ignore_index=True)
                else:
                    combined_data = source_corrections
            
                # Log summary of merged data
                self._log_data_summary(combined_data, f"Combined {source.value.capitalize()} Data")
            
                # Determine new version number for this source
                source_pattern = f"training_data_{source.value}_v(\\d+)"
                existing_versions = [
                    int(re.search(source_pattern, f).group(1)) 
                    for f in glob.glob(os.path.join(self.data_dir, f"training_data_{source.value}_v*.csv")) 
                    if re.search(source_pattern, f)
                ]
                new_version = max(existing_versions) + 1 if existing_versions else 1
            
                # Save updated training dataset
                timestamp = datetime.now().strftime("%Y%m%d")
                export_filename = f"training_data_{source.value}_v{new_version}_{timestamp}.csv"
                export_path = os.path.join(self.data_dir, export_filename)
                
                combined_data.to_csv(export_path, index=False, encoding='utf-8')
                self.logger.info(f"✅ Exported new training dataset for {source.value}: {export_filename}")
                
                result_files.append(export_filename)
                
            except Exception as e:
                self.logger.error(f"Error processing {source_value} corrections: {str(e)}")
                raise
                
        return ", ".join(result_files)

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
        file_path = os.path.join(validator.data_dir, corrections_file)
        if is_excel_file(file_path):
            excel_processor = ExcelProcessor()
            corrections_df = excel_processor.process_excel_file(file_path)
        else:
            corrections_df = pd.read_csv(file_path)
        
        # Check if transaction_source column exists
        if 'transaction_source' not in corrections_df.columns:
            print("\nThe selected file doesn't have a 'transaction_source' column.")
            print("This column is required to identify whether transactions are from household or credit card sources.")
            print("\nPlease choose an option:")
            print("1. Add 'household' as transaction_source for all rows")
            print("2. Add 'credit_card' as transaction_source for all rows")
            print("3. Cancel processing")
            
            while True:
                try:
                    source_choice = input("\nEnter your choice (1-3): ")
                    if source_choice == '1':
                        corrections_df['transaction_source'] = 'household'
                        print("Added 'household' as transaction_source for all records.")
                        break
                    elif source_choice == '2':
                        corrections_df['transaction_source'] = 'credit_card'
                        print("Added 'credit_card' as transaction_source for all records.")
                        break
                    elif source_choice == '3':
                        print("Operation cancelled.")
                        return
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, or 3.")
                except ValueError:
                    print("❌ Please enter a valid number.")
        
        # Process the file
        try:
            export_files = validator.validate_and_prepare(corrections_df)
            
            print("\n✅ Processing Complete!")
            print(f"Results saved to: {export_files}")
            print("\nCheck the logs for detailed validation and processing information.")
            
        except ValueError as e:
            print(f"\n❌ Validation Error: {str(e)}")
            print("Please fix the issues and try again.")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        validator.logger.error("Processing failed", exc_info=True)
        raise

class CorrectionValidatorExtended(CorrectionValidator):
    """Extended CorrectionValidator with additional methods for unified workflow"""
    
    def process_corrections_file(self, file_path: str, transaction_source: str, auto_confirm: bool = False) -> bool:
        """
        Process corrections file for unified workflow
        
        Args:
            file_path: Path to corrections file
            transaction_source: 'household' or 'credit_card'
            auto_confirm: Skip interactive prompts if True
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Load corrections
            if is_excel_file(file_path):
                excel_processor = ExcelProcessor()
                corrections_df = excel_processor.process_excel_file(file_path)
            else:
                corrections_df = pd.read_csv(file_path)
            
            # Add transaction source if missing
            if 'transaction_source' not in corrections_df.columns:
                corrections_df['transaction_source'] = transaction_source
                if not auto_confirm:
                    print(f"Added '{transaction_source}' as transaction_source for all records.")
            
            # Validate and process
            export_files = self.validate_and_prepare(corrections_df)
            
            if not auto_confirm:
                print("\n✅ Training data updated successfully!")
                print(f"Results saved to: {export_files}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process corrections: {e}")
            if not auto_confirm:
                print(f"❌ Failed to process corrections: {str(e)}")
            return False

if __name__ == "__main__":
    main()