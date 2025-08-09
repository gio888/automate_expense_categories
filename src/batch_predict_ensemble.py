import os
import sys
import argparse
from pathlib import Path

# Setup the path BEFORE any other imports
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
import glob
import json
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Any
from src.transaction_types import TransactionSource  # Import the enum
from src.model_registry import ModelRegistry
from src.model_storage import ModelStorage
from src.utils.filename_utils import extract_source_and_period, generate_filename
from src.utils.excel_processor import ExcelProcessor, is_excel_file
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'batch_predict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchPredictor:
    """Handles batch prediction with source-specific models and efficient processing"""
    
    def __init__(self, batch_size: int = 1000):
        credentials_path = os.getenv("GCP_CREDENTIALS_PATH")
        if not credentials_path:
            logger.error("GCP_CREDENTIALS_PATH environment variable is not set.")
            raise ValueError("GCP_CREDENTIALS_PATH environment variable is not set.")
        
        self.registry = ModelRegistry(registry_dir=os.path.join(MODELS_DIR, "registry"))
        self.storage = ModelStorage(
            bucket_name="expense-categorization-ml-models-backup",
            credentials_path=credentials_path,
            models_dir=MODELS_DIR,
            cache_dir="cache"
        )
        self.batch_size = batch_size
        # Dictionary to store models for each source
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[str, Dict[str, int]] = {}
        # Dictionary to store vectorizers for each source
        self.vectorizers: Dict[str, Any] = {}
        self.vectorizer_versions: Dict[str, int] = {}
        # Dictionary to store label encoders for each source
        self.label_encoders: Dict[str, Any] = {}
        self.label_encoder_versions: Dict[str, int] = {}
        self.load_latest_models()
        self.version = "1.0"  # Set the version attribute
    
    def load_latest_models(self) -> None:
        for source in [TransactionSource.HOUSEHOLD, TransactionSource.CREDIT_CARD]:
            try:
                logger.info(f"\nLoading models for {source.value}...")
            
                # Load vectorizer with version tracking
                model_info = self.registry.get_model_info(f"{source.value}_tfidf_vectorizer")
                if model_info:
                    version = model_info['version']
                    self.vectorizers[source.value] = self.storage.load_model(
                        f"{source.value}_tfidf_vectorizer", 
                        version
                    )
                    self.vectorizer_versions[source.value] = version
                    logger.info(f"✅ Loaded {source.value} TF-IDF vectorizer v{version}")
                
                    # Load label encoder with version tracking
                    encoder_info = self.registry.get_model_info(f"{source.value}_label_encoder")
                    if encoder_info:
                        version = encoder_info['version']
                        self.label_encoders[source.value] = self.storage.load_model(
                            f"{source.value}_label_encoder",
                            version
                        )
                        self.label_encoder_versions[source.value] = version
                        logger.info(f"✅ Successfully loaded {source.value} label encoder v{version}")
                        logger.info(f"Label encoder classes: {self.label_encoders[source.value].classes_}")
                    else:
                        logger.warning(f"⚠️ No label encoder found for {source.value}")
                
                    # Load models with version tracking
                    self.models[source.value] = {}
                    self.model_versions[source.value] = {}
                
                    for model_type in ['lgbm', 'xgboost', 'catboost']:
                        model_name = f"{source.value}_{model_type}"
                        model_info = self.registry.get_model_info(model_name)
                        if model_info:
                            version = model_info['version']
                            self.models[source.value][model_type] = self.storage.load_model(
                                model_name,
                                version
                            )
                            self.model_versions[source.value][model_type] = version
                            logger.info(f"Loaded {source.value} {model_type} model v{version}")
                        else:
                            logger.warning(f"No {model_type} model found for {source.value}")
                else:
                    logger.warning(f"No models found for {source.value}")
                
            except Exception as e:
                logger.error(f"Error loading {source.value} models: {str(e)}")

    def predict_batch(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """Process predictions in batches for better performance"""
        all_predictions = []
        all_confidences = []
        
        try:
            if 'transaction_source' not in df.columns:
                raise ValueError("DataFrame must contain 'transaction_source' column")
            
            # Handle case-insensitive column names for Description
            description_col = None
            for col in df.columns:
                if col.lower() == 'description':
                    description_col = col
                    break
            
            if description_col is None:
                raise ValueError("DataFrame must contain 'Description' or 'description' column")
            
            # Standardize column name to 'Description' if needed
            if description_col != 'Description':
                df = df.rename(columns={description_col: 'Description'})
                
            # Group data by transaction source
            for source in [TransactionSource.HOUSEHOLD, TransactionSource.CREDIT_CARD]:
                source_df = df[df['transaction_source'] == source.value]
                if len(source_df) == 0:
                    continue
                    
                logger.info(f"\nProcessing {len(source_df)} {source.value} transactions...")
                
                for i in range(0, len(source_df), self.batch_size):
                    batch = source_df.iloc[i:i + self.batch_size]
                    
                    # Transform batch using source-specific vectorizer
                    features = self.vectorizers[source.value].transform(batch['Description'])
                    
                    # Get predictions from all available models for this source
                    batch_predictions = []
                    batch_confidences = []
                    
                    for model_type, model in self.models[source.value].items():
                        try:
                            preds = model.predict(features)
                            probs = model.predict_proba(features)
                            batch_predictions.append(preds)
                            batch_confidences.append(np.max(probs, axis=1))
                            logger.debug(f"Got predictions from {source.value} {model_type} model")
                        except Exception as e:
                            logger.warning(f"Error getting predictions from {source.value} {model_type} model: {str(e)}")
                            continue
                    
                    if not batch_predictions:
                        raise ValueError(f"No valid predictions from any model for source {source.value}!")
                    
                    # Combine predictions using majority voting
                    pred_df = pd.DataFrame(batch_predictions)
                    final_predictions = pred_df.mode(axis=0).iloc[0]
                    
                    # Add debug logging for predictions
                    logger.info(f"Number of {source.value} predictions before label encoding: {len(final_predictions)}")
                    logger.info(f"Sample of numeric predictions: {final_predictions[:5]}")
                    logger.info(f"Predictions dtype: {final_predictions.dtype}")
                    
                    # Convert numeric predictions back to category names
                    if source.value in self.label_encoders:
                        logger.info(f"Converting {source.value} predictions using label encoder...")
                        try:
                            final_predictions = final_predictions.astype(int)
                            logger.info(f"Converted predictions to integers. Sample: {final_predictions[:5]}")
                            
                            final_predictions = self.label_encoders[source.value].inverse_transform(final_predictions)
                            logger.info(f"✅ Successfully converted {source.value} predictions to categories")
                            logger.info(f"Sample of category predictions: {final_predictions[:5]}")
                        except Exception as e:
                            logger.error(f"❌ Error converting {source.value} predictions: {str(e)}")
                            logger.error(f"Label encoder classes: {self.label_encoders[source.value].classes_}")
                    else:
                        logger.warning(f"⚠️ No label encoder available for {source.value} - predictions will remain numeric")
                    
                    # Average confidence scores
                    final_confidences = np.mean(batch_confidences, axis=0)
                    
                    all_predictions.extend(final_predictions)
                    all_confidences.extend(final_confidences)
                    
                    logger.debug(f"Processed batch of {len(batch)} {source.value} descriptions")
                
            if not all_predictions:
                raise ValueError("No predictions generated for any source!")
                
            return all_predictions, all_confidences
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise

def process_file(input_file: str) -> None:
    try:
        predictor = BatchPredictor()
        monitor = PredictionMonitor(LOGS_DIR)
        
        logger.info(f"Loading transaction data from: {input_file}")
        
        # Handle both CSV and Excel files
        if is_excel_file(input_file):
            excel_processor = ExcelProcessor()
            df = excel_processor.process_excel_file(input_file)
        else:
            df = pd.read_csv(input_file)
        
        if 'transaction_source' not in df.columns:
            df['transaction_source'] = 'credit_card'
            
        logger.info(f"Loaded {len(df)} transactions")
        
        predictions, confidences = predictor.predict_batch(df)
        
        df["Category"] = predictions
        df["Prediction_Confidence"] = confidences
        
        # Use the correct column names from your dataset
        if 'Amount (Negated)' in df.columns:
            df["Amount (Negated)"] = df["Amount (Negated)"]
        else:
            logger.warning("Column 'Amount (Negated)' not found in the DataFrame.")
            df["Amount (Negated)"] = np.nan
        
        if 'Amount' in df.columns:
            df["Amount"] = df["Amount"]
        else:
            logger.warning("Column 'Amount' not found in the DataFrame.")
            df["Amount"] = np.nan
        
        df = df[["Date", "Description", "Category", "Prediction_Confidence", 
                 "Amount (Negated)", "Amount"]]
        
        # Generate standardized output filename
        original_filename = os.path.basename(input_file)
        source, period = extract_source_and_period(original_filename)
        timestamp = datetime.now()
        output_filename = generate_filename(source, period, "predictions", timestamp)
        output_file = os.path.join(DATA_DIR, output_filename)
        
        logging.info(f"Generated standardized filename: {output_filename} (source: {source}, period: {period})")
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        monitor.log_metrics(df)
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Total transactions processed: {len(df)}")
        logger.info(f"Average prediction confidence: {df['Prediction_Confidence'].mean():.2%}")
        logger.info(f"Low confidence predictions (<50%): {(df['Prediction_Confidence'] < 0.5).sum()}")
        
        print(f"\n✅ Processing complete! Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise

class PredictionMonitor:
    """Monitors prediction performance and logs metrics"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, 'prediction_metrics.jsonl')
        
    def log_metrics(self, predictions_df: pd.DataFrame) -> None:
        """Log prediction metrics for monitoring"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(predictions_df),
                'confidence_stats': {
                    'mean': float(predictions_df['Prediction_Confidence'].mean()),
                    'median': float(predictions_df['Prediction_Confidence'].median()),
                    'std': float(predictions_df['Prediction_Confidence'].std()),
                    'low_confidence_rate': float((predictions_df['Prediction_Confidence'] < 0.5).mean())
                },
                'category_distribution': predictions_df['Category'].value_counts().to_dict()
            }
            
            # Log metrics
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Check for concerning patterns
            self.check_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def check_metrics(self, metrics: Dict[str, Any]) -> None:
        """Check metrics for concerning patterns"""
        alerts = []
        
        # Check confidence levels
        if metrics['confidence_stats']['low_confidence_rate'] > 0.2:
            alerts.append(f"High rate of low confidence predictions: {metrics['confidence_stats']['low_confidence_rate']:.2%}")
        
        if metrics['confidence_stats']['mean'] < 0.6:
            alerts.append(f"Low average confidence score: {metrics['confidence_stats']['mean']:.2%}")
        
        # Log alerts if any
        if alerts:
            logger.warning("⚠️ Prediction Quality Alerts:")
            for alert in alerts:
                logger.warning(f"- {alert}")

def interactive_file_selection():
    """Interactive file selection with improved UX"""
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    if not csv_files:
        print("❌ No CSV files found in data/ directory")
        return None
    
    print(f"\nAvailable CSV files in {DATA_DIR}:")
    for i, file in enumerate(csv_files, 1):
        file_path = os.path.join(DATA_DIR, file)
        size = os.path.getsize(file_path)
        print(f"  {i}. 📄 {file} ({size:,} bytes)")
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(csv_files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(csv_files):
                return csv_files[file_index]
            else:
                print(f"❌ Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")

def main():
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(
        description="Predict expense categories for financial transactions using ensemble ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --file transactions.csv                    # Process specific file
  python %(prog)s --list                                     # List available files
  python %(prog)s --interactive                              # Interactive file selection
  python %(prog)s --file data.csv --source household        # Specify transaction source

Files are processed from the data/ directory and results saved using standardized naming:
{source}_{period}_{stage}_{timestamp}.csv (e.g., unionbank_visa_2025-07_predictions_20250808_194140.csv)
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        help='CSV file to process (relative to data/ directory)'
    )
    
    parser.add_argument(
        '--source', '-s',
        choices=['household', 'credit_card'],
        help='Transaction source type (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available CSV files in data directory'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive file selection mode'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # List files mode
        if args.list:
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx', '.xls'))]
            print(f"\nAvailable CSV files in {DATA_DIR}:")
            for file in csv_files:
                file_path = os.path.join(DATA_DIR, file)
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size:,} bytes)")
            return
        
        # Determine input file
        input_file = None
        
        if args.file:
            # Direct file specification
            # Support both CSV and Excel files
            if not (args.file.endswith('.csv') or args.file.endswith('.xlsx') or args.file.endswith('.xls')):
                # Only add .csv if no extension is provided
                if '.' not in os.path.basename(args.file):
                    args.file += '.csv'
            input_file = args.file
            
        elif args.interactive:
            # Interactive mode
            input_file = interactive_file_selection()
            
        else:
            # Default: try interactive or show help
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx', '.xls'))]
            if len(csv_files) == 0:
                print("❌ No CSV files found in data/ directory")
                print("💡 Add your transaction files to the data/ folder and try again")
                return
            elif len(csv_files) == 1:
                input_file = csv_files[0]
                print(f"📄 Auto-selected only file: {input_file}")
            else:
                print("Multiple files found. Use --interactive or specify --file")
                print("Run with --list to see available files")
                return
        
        if not input_file:
            return
            
        # Validate file exists
        input_path = os.path.join(DATA_DIR, input_file)
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            print("💡 Run with --list to see available files")
            return
        
        print(f"\n🚀 Processing: {input_file}")
        if args.source:
            print(f"📊 Transaction source: {args.source}")
        
        # Process the file
        process_file(input_path)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print("\n❌ An error occurred. Check the logs for details.")

if __name__ == "__main__":
    main()