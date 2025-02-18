import os
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

# Define paths and setup logging
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        self.registry = ModelRegistry(registry_dir=os.path.join(MODELS_DIR, "registry"))
        self.storage = ModelStorage(
            bucket_name="expense-categorization-ml-models-backup",
            credentials_path=os.getenv("GCP_CREDENTIALS_PATH"),
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
            
            if 'Description' not in df.columns:
                raise ValueError("DataFrame must contain 'Description' column")
                
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
        
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DATA_DIR, f"processed_{input_basename}_v{predictor.version}_{timestamp}.csv")
        
        df.to_csv(output_file, index=False)
        
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
        raise

def main():
    """Main entry point with improved error handling"""
    try:
        # Show available CSV files
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        print("\nAvailable CSV files in data directory:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        
        # Get user input with improved error handling
        while True:
            try:
                choice = input("\nEnter the number of the file to process (or 'q' to quit): ")
                if choice.lower() == 'q':
                    return
                
                file_index = int(choice) - 1
                if 0 <= file_index < len(csv_files):
                    input_file = csv_files[file_index]
                    break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Process the selected file
        input_path = os.path.join(DATA_DIR, input_file)
        print(f"\n📂 Selected file: {input_file}")
        
        confirm = input("Proceed with processing? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
        process_file(input_path)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print("\n❌ An error occurred. Check the logs for details.")

if __name__ == "__main__":
    main()