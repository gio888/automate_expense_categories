import os
import pandas as pd
import numpy as np
import joblib
import glob
import json
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Any

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
    """Handles batch prediction with efficient processing and monitoring"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.models = {}
        self.model_versions = {}  # Track versions for each model
        self.vectorizer = None
        self.vectorizer_version = None
        self.label_encoder = None
        self.load_latest_models()
        
    def get_latest_version(self, pattern: str) -> int:
        """Find latest version for a specific model pattern"""
        files = glob.glob(os.path.join(MODELS_DIR, pattern))
        if not files:
            return None
            
        versions = []
        for f in files:
            try:
                version = int(f.split('_v')[-1].split('.pkl')[0])
                versions.append(version)
            except (ValueError, IndexError):
                continue
                
        return max(versions) if versions else None
        
    def load_latest_models(self) -> None:
        """Load the latest versions of all models and vectorizer"""
        try:
            # Load latest vectorizer version first
            self.vectorizer_version = self.get_latest_version("tfidf_vectorizer_model_v*.pkl")
            if self.vectorizer_version is None:
                raise ValueError("No vectorizer versions found!")
                
            vectorizer_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer_model_v{self.vectorizer_version}.pkl")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Latest vectorizer file not found: {vectorizer_path}")
            
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Loaded TF-IDF vectorizer v{self.vectorizer_version}")
            
            # Load label encoder with correct model_storage.py pattern
            label_encoder_path = os.path.join(MODELS_DIR, f"label_encoder_model_v{self.vectorizer_version}.pkl")
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info(f"Loaded label encoder v{self.vectorizer_version}")
            else:
                logger.warning("Label encoder not found! Predictions will be numeric.")
            
            # Load latest version of each model type independently
            model_types = ['lgbm', 'xgboost', 'catboost']
            for model_type in model_types:
                pattern = f"{model_type}_model_v*.pkl"
                latest_version = self.get_latest_version(pattern)
                
                if latest_version is not None:
                    model_path = os.path.join(MODELS_DIR, f"{model_type}_model_v{latest_version}.pkl")
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        self.models[model_type] = model
                        self.model_versions[model_type] = latest_version
                        logger.info(f"Loaded {model_type} model v{latest_version}")
                    else:
                        logger.warning(f"Model not found: {model_path}")
                else:
                    logger.warning(f"No versions found for {model_type}")
            
            if not self.models:
                raise ValueError("No models could be loaded!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def predict_batch(self, descriptions: List[str]) -> Tuple[List[str], List[float]]:
        """Process predictions in batches for better performance"""
        all_predictions = []
        all_confidences = []
        
        try:
            # Process in batches
            for i in range(0, len(descriptions), self.batch_size):
                batch = descriptions[i:i + self.batch_size]
                
                # Transform batch using vectorizer
                features = self.vectorizer.transform(batch)
                
                # Get predictions from all available models
                batch_predictions = []
                batch_confidences = []
                
                for model_name, model in self.models.items():
                    try:
                        preds = model.predict(features)
                        probs = model.predict_proba(features)
                        batch_predictions.append(preds)
                        batch_confidences.append(np.max(probs, axis=1))
                    except Exception as e:
                        logger.warning(f"Error getting predictions from {model_name} model: {str(e)}")
                        continue
                
                if not batch_predictions:
                    raise ValueError("No valid predictions from any model!")
                
                # Combine predictions using majority voting
                pred_df = pd.DataFrame(batch_predictions)
                final_predictions = pred_df.mode(axis=0).iloc[0]
                
                # Convert numeric predictions back to category names if possible
                if self.label_encoder is not None:
                    final_predictions = self.label_encoder.inverse_transform(final_predictions)
                
                # Average confidence scores
                final_confidences = np.mean(batch_confidences, axis=0)
                
                all_predictions.extend(final_predictions)
                all_confidences.extend(final_confidences)
                
                logger.debug(f"Processed batch of {len(batch)} descriptions")
            
            if not all_predictions:
                raise ValueError("No predictions generated!")
                
            return all_predictions, all_confidences
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
            
    @property
    def version(self) -> str:
        """Generate a version string combining all model versions"""
        versions = [f"{model}v{ver}" for model, ver in self.model_versions.items()]
        return '_'.join(versions)

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

def process_file(input_file: str) -> None:
    """Process input file with improved batch processing and monitoring"""
    try:
        # Initialize predictor and monitor
        predictor = BatchPredictor()
        monitor = PredictionMonitor(LOGS_DIR)
        
        # Load input file
        logger.info(f"Loading transaction data from: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} transactions")
        
        # Get predictions in batches
        predictions, confidences = predictor.predict_batch(df["Description"].tolist())
        
        # Add predictions to dataframe
        df["Category"] = predictions
        df["Prediction_Confidence"] = confidences
        
        # Map In/Out columns to Amount columns
        df["Amount (Negated)"] = df["Out"]
        df["Amount"] = df["In"]
        
        # Keep required columns
        df = df[["Date", "Description", "Category", "Prediction_Confidence", 
                "Amount (Negated)", "Amount"]]
        
        # Generate output filename
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DATA_DIR, f"processed_{input_basename}_v{predictor.version}_{timestamp}.csv")
        
        # Save processed data
        df.to_csv(output_file, index=False)
        
        # Log metrics
        monitor.log_metrics(df)
        
        # Log summary statistics
        logger.info("\nProcessing Summary:")
        logger.info(f"Total transactions processed: {len(df)}")
        logger.info(f"Average prediction confidence: {df['Prediction_Confidence'].mean():.2%}")
        logger.info(f"Low confidence predictions (<50%): {(df['Prediction_Confidence'] < 0.5).sum()}")
        
        print(f"\n✅ Processing complete! Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
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