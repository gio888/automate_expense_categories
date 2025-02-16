# src/improvements/create_baseline.py
import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.batch_predict_ensemble import BatchPredictor, PredictionMonitor

def create_baseline():
    # Initialize existing predictor
    predictor = BatchPredictor()
    monitor = PredictionMonitor(os.path.join("logs"))
    
    # Load latest processed data for testing
    data_dir = Path("data")
    processed_files = list(data_dir.glob("processed_*.csv"))
    if not processed_files:
        raise FileNotFoundError("No processed data files found")
    
    # Use latest processed file as test data
    test_data = pd.read_csv(max(processed_files, key=os.path.getctime))
    print(f"Using test data: {test_data.shape[0]} records")
    
    # Get predictions using existing pipeline
    predictions, confidences = predictor.predict_batch(test_data["Description"].tolist())
    
    # Create metrics using existing monitor
    df_predictions = pd.DataFrame({
        'Description': test_data["Description"],
        'Category': predictions,
        'Prediction_Confidence': confidences
    })
    
    # Log metrics using existing monitor
    monitor.log_metrics(df_predictions)
    
    # Save this as our baseline
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    baseline = {
        'timestamp': datetime.now().isoformat(),
        'test_file': str(max(processed_files, key=os.path.getctime)),
        'model_versions': predictor.model_versions,
        'vectorizer_version': predictor.vectorizer_version,
        'predictions_shape': df_predictions.shape,
        'confidence_mean': float(df_predictions['Prediction_Confidence'].mean()),
        'confidence_std': float(df_predictions['Prediction_Confidence'].std())
    }
    
    baseline_file = metrics_dir / "baseline_v1.json"
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\nBaseline created: {baseline_file}")
    print(f"Using models: {predictor.model_versions}")
    print(f"Mean confidence: {baseline['confidence_mean']:.4f}")

if __name__ == "__main__":
    create_baseline()