import sys
import os

# Ensure the src directory is in the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import joblib
import glob
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
from model_registry import ModelRegistry
from model_storage import ModelStorage

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
REGISTRY_DIR = os.path.join(MODELS_DIR, "registry")

# Initialize model registry and storage
model_registry = ModelRegistry(REGISTRY_DIR)
model_storage = ModelStorage(
    bucket_name="expense-categorization-ml-models-backup",
    credentials_path="***CREDENTIALS-REMOVED***",
    models_dir=MODELS_DIR
)

# Logging setup
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
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.models = {}
        self.vectorizer = None
        self.version = None
        self.load_latest_models()

    def load_latest_models(self) -> None:
        """Load the latest stable model and vectorizer, with cloud fallback."""
        try:
            latest_version = model_registry.get_latest_version("lgbm")
            if latest_version is None:
                raise ValueError("No registered model versions found!")
            
            self.version = latest_version
            logger.info(f"Using model version v{self.version}")
            
            model_types = ['lgbm', 'xgboost', 'catboost']
            for model_type in model_types:
                model_path = os.path.join(MODELS_DIR, f"{model_type}_model_v{self.version}.pkl")
                if not os.path.exists(model_path):
                    logger.warning(f"{model_type} model missing locally. Attempting cloud retrieval.")
                    model_path = model_storage.load_model(model_type, self.version)
                
                self.models[model_type] = joblib.load(model_path)
                logger.info(f"Loaded {model_type} model.")
            
            # Load vectorizer with validation
            vectorizer_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer_v{self.version}.pkl")
            if not os.path.exists(vectorizer_path):
                logger.warning("Vectorizer missing locally. Attempting cloud retrieval.")
                vectorizer_path = model_storage.load_model("tfidf_vectorizer", self.version)
            
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("Loaded TF-IDF vectorizer.")
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def predict_batch(self, descriptions: List[str]) -> Tuple[List[str], List[float]]:
        """Process predictions in batches and handle low-confidence cases."""
        all_predictions = []
        all_confidences = []
        
        try:
            for i in range(0, len(descriptions), self.batch_size):
                batch = descriptions[i:i + self.batch_size]
                features = self.vectorizer.transform(batch)
                
                batch_predictions = []
                batch_confidences = []
                
                for model_name, model in self.models.items():
                    preds = model.predict(features)
                    probs = model.predict_proba(features)
                    batch_predictions.append(preds)
                    batch_confidences.append(np.max(probs, axis=1))
                
                final_predictions = pd.DataFrame(batch_predictions).mode(axis=0).iloc[0]
                final_confidences = np.mean(batch_confidences, axis=0)
                
                # Handle low-confidence cases
                for idx, confidence in enumerate(final_confidences):
                    if confidence < 0.5:
                        final_predictions.iloc[idx] = "Uncategorized"  # Fallback category
                
                all_predictions.extend(final_predictions)
                all_confidences.extend(final_confidences)
                
                logger.debug(f"Processed batch of {len(batch)} descriptions")
            
            return all_predictions, all_confidences
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise


def process_file(input_file: str) -> None:
    try:
        predictor = BatchPredictor()
        logger.info(f"Loading transaction data from: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} transactions")
        
        predictions, confidences = predictor.predict_batch(df["Description"].tolist())
        df["Category"] = predictions
        df["Prediction_Confidence"] = confidences
        
        df = df[["Date", "Description", "Category", "Prediction_Confidence"]]
        output_file = os.path.join(DATA_DIR, f"processed_{os.path.basename(input_file)}")
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Processing complete! Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print("\nAvailable CSV files in data directory:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    while True:
        choice = input("\nEnter the number of the file to process (or 'q' to quit): ")
        if choice.lower() == 'q':
            exit()
        
        try:
            file_index = int(choice) - 1
            if 0 <= file_index < len(csv_files):
                process_file(os.path.join(DATA_DIR, csv_files[file_index]))
                break
            else:
                print("❌ Invalid selection. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")
