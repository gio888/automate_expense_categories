import os
import pandas as pd
import numpy as np
import joblib
import glob
from datetime import datetime
import logging

# ✅ Define paths and setup logging
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # File handler for debugging/troubleshooting
        logging.FileHandler(os.path.join(LOGS_DIR, f'batch_predict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        # Console handler for immediate feedback
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_latest_model_version():
    """Get the latest version number from model files"""
    model_files = glob.glob(os.path.join(MODELS_DIR, "lgbm_model_v*.pkl"))
    versions = [int(f.split('_v')[-1].split('.pkl')[0]) for f in model_files]
    return max(versions) if versions else None

def load_latest_models():
    """Load the latest versions of all models and vectorizer"""
    version = get_latest_model_version()
    if not version:
        raise ValueError("No versioned models found!")
    
    logger.info(f"Loading model version v{version}")
    
    models = {
        'lgbm': joblib.load(os.path.join(MODELS_DIR, f"lgbm_model_v{version}.pkl")),
        'xgboost': joblib.load(os.path.join(MODELS_DIR, f"xgboost_model_v{version}.pkl")),
        'catboost': joblib.load(os.path.join(MODELS_DIR, f"catboost_model_v{version}.pkl")),
    }
    
    vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_vectorizer_v{version}.pkl"))
    
    return models, vectorizer, version

def predict_category_ensemble(description, models, vectorizer):
    """
    Predict categories using an ensemble of models with confidence scores.
    Returns predicted category and confidence score.
    """
    try:
        # Transform text description to TF-IDF features
        features = vectorizer.transform([description])
        
        predictions = []
        probabilities = []
        
        # Get predictions and probabilities from all models
        for model_name, model in models.items():
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)
            predictions.append(pred)
            probabilities.append(np.max(prob))  # Get highest probability
            
            # Log detailed prediction info for debugging
            logger.debug(f"{model_name} prediction: {pred}, confidence: {np.max(prob):.3f}")
        
        # Majority voting using pandas
        final_category = pd.Series(predictions).mode()[0]
        
        # Calculate confidence as average of probabilities
        confidence = np.mean(probabilities)
        
        return final_category, confidence
        
    except Exception as e:
        logger.error(f"Error predicting category for description '{description}': {str(e)}")
        raise

def process_file(input_file):
    """Process input file with latest models"""
    try:
        # Load latest models
        models, vectorizer, version = load_latest_models()
        
        # Load input file
        logger.info(f"Loading transaction data from: {input_file}")
        df = pd.read_csv(input_file)
        
        # Log initial data statistics
        logger.info(f"Loaded {len(df)} transactions")
        
        # Predict categories and confidence scores
        logger.info("Starting category predictions")
        predictions = [predict_category_ensemble(desc, models, vectorizer) 
                      for desc in df["Description"]]
        
        # Split predictions and confidence scores
        df["Category"], df["Prediction_Confidence"] = zip(*predictions)
        
        # Map In/Out columns to Amount columns
        df["Amount (Negated)"] = df["Out"]  # Out amounts (negative/decreasing)
        df["Amount"] = df["In"]    # In amounts (positive/increasing)
        
        # Keep required columns
        df = df[["Date", "Description", "Category", "Prediction_Confidence", 
                "Amount (Negated)", "Amount"]]
        
        # Generate output filename
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DATA_DIR, f"processed_{input_basename}_v{version}_{timestamp}.csv")
        
        # Save processed data
        df.to_csv(output_file, index=False)
        
        # Log performance metrics
        logger.info("\nPerformance Metrics:")
        logger.info(f"Total transactions processed: {len(df)}")
        logger.info(f"Average prediction confidence: {df['Prediction_Confidence'].mean():.2%}")
        logger.info(f"Confidence distribution:")
        logger.info(f"- High confidence (>70%): {(df['Prediction_Confidence'] > 0.7).sum()}")
        logger.info(f"- Medium confidence (50-70%): {((df['Prediction_Confidence'] > 0.5) & (df['Prediction_Confidence'] <= 0.7)).sum()}")
        logger.info(f"- Low confidence (<50%): {(df['Prediction_Confidence'] < 0.5).sum()}")
        
        # Log category distribution
        logger.info("\nCategory Distribution:")
        category_dist = df["Category"].value_counts()
        for category, count in category_dist.items():
            logger.info(f"{category}: {count}")
        
        # Log low confidence predictions for review
        low_conf_threshold = 0.5
        low_conf_preds = df[df["Prediction_Confidence"] < low_conf_threshold]
        if not low_conf_preds.empty:
            logger.warning("\nLow Confidence Predictions (Require Review):")
            for _, row in low_conf_preds.iterrows():
                logger.warning(f"Description: {row['Description']}")
                logger.warning(f"Predicted Category: {row['Category']}")
                logger.warning(f"Confidence: {row['Prediction_Confidence']:.2%}\n")
        
        print(f"\n✅ Processing complete! Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise

def main():
    # Show available CSV files in data directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    print("\nAvailable CSV files in data directory:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    # Ask user to select a file
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
    
    # Construct full path to input file
    input_path = os.path.join(DATA_DIR, input_file)
    print(f"\n📂 Selected file: {input_file}")
    
    # Confirm with user
    confirm = input("Proceed with processing? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    process_file(input_path)

if __name__ == "__main__":
    main()