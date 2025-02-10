import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import glob
import re

class CorrectionValidator:
    def __init__(self):
        """Initialize with auto-detected project root"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        self.logs_dir = os.path.join(self.project_root, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.logs_dir, 'correction_validation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.valid_categories = self._load_valid_categories()
    
    def _load_valid_categories(self):
        """Load valid categories from valid_categories.txt"""
        categories_file = os.path.join(self.data_dir, "valid_categories.txt")
        if not os.path.exists(categories_file):
            self.logger.error(f"Categories file not found: {categories_file}")
            raise FileNotFoundError(f"Please create {categories_file} with all valid categories")
        
        with open(categories_file, 'r') as f:
            categories = set(line.strip() for line in f if line.strip())
        
        self.logger.info(f"Loaded {len(categories)} valid categories")
        return categories

    def validate_and_prepare(self, corrections_df: pd.DataFrame, model_version: str):
        """Validate corrections and merge into latest training dataset"""
        required_columns = ['Date', 'Description', 'Amount (Negated)', 'Amount', 
                            'Category', 'Amount (Raw)', 'Entered', 'Reconciled']
        
        # Handle missing columns
        for col in required_columns:
            if col not in corrections_df.columns:
                if col == 'Amount (Raw)':
                    corrections_df[col] = corrections_df.apply(
                        lambda x: (x['Amount'] - x['Amount (Negated)'])
                        if pd.notnull(x['Amount']) and pd.notnull(x['Amount (Negated)'])
                        else np.nan, axis=1
                    )
                elif col in ['Entered', 'Reconciled']:
                    corrections_df[col] = False  # Default to False
                else:
                    corrections_df[col] = None  # Default to None for any other missing columns
        
        # Extract validated data
        new_training_data = corrections_df[required_columns].copy()
        
        # Load existing training data
        latest_training_file = self._get_latest_training_data()
        if latest_training_file is not None:
            existing_data = pd.read_csv(latest_training_file)
            combined_data = pd.concat([existing_data, new_training_data]).drop_duplicates(subset=['Description'], keep='last')
        else:
            combined_data = new_training_data
        
        # Determine new version number
        existing_versions = [int(re.search(r'v(\d+)', f).group(1)) for f in glob.glob(os.path.join(self.data_dir, "training_data_v*.csv")) if re.search(r'v(\d+)', f)]
        new_version = max(existing_versions) + 1 if existing_versions else 1
        
        # Save updated training dataset
        timestamp = datetime.now().strftime("%Y%m%d")
        export_filename = f"training_data_v{new_version}_{timestamp}.csv"
        combined_data.to_csv(os.path.join(self.data_dir, export_filename), index=False)
        
        self.logger.info(f"✅ Merged training data exported to: {export_filename}")
        return export_filename
    
    def _get_latest_training_data(self):
        """Get the latest training data file"""
        training_files = glob.glob(os.path.join(self.data_dir, "training_data_v*.csv"))
        if not training_files:
            self.logger.warning("No existing training data found")
            return None
        return max(training_files, key=os.path.getmtime)


def main():
    """Main execution flow for the correction validator."""
    validator = CorrectionValidator()
    
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
    
    try:
        corrections_df = pd.read_csv(os.path.join(validator.data_dir, corrections_file))
        export_file = validator.validate_and_prepare(corrections_df, "v5")
        
        print("\n✅ Validation, Merging, and Processing Complete!")
        print(f"Final training data exported to: {export_file}")
    except Exception as e:
        print(f"\n❌ Error processing file: {str(e)}")
        raise
    

if __name__ == "__main__":
    main()
