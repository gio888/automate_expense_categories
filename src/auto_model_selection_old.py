import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import os
import time

def auto_find_best_model(X_train, y_train, X_test, y_test, time_budget=300):
    """
    Automatically find the best model using FLAML
    """
    print(f"Starting automated model search (time budget: {time_budget} seconds)...")
    
    # Initialize AutoML
    automl = AutoML()
    
    # Start timer
    start_time = time.time()
    
    # Fit AutoML - corrected estimator_list
    automl.fit(
        X_train, y_train,
        task='classification',
        metric='macro_f1',
        time_budget=time_budget,
        estimator_list=['lgbm', 'rf', 'xgboost', 'catboost', 'lrl1'],  # corrected list
        verbose=2
    )
    
    # Print results
    print("\nAutoML Results:")
    print(f"Best ML leaner: {automl.best_estimator}")
    print(f"Best hyperparmeter config: {automl.best_config}")
    print(f"Best macro F1 score: {automl.best_loss}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Evaluate on test set
    y_pred = automl.predict(X_test)
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred))
    
    return automl

if __name__ == "__main__":
    # Import load_and_split_data from model_evaluation
    from model_evaluation import load_and_split_data
    
    # Set up paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')
    
    # Load and split data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, min_samples=5)
    
    # Convert text to TF-IDF features first
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    
    # Find best model
    best_model = auto_find_best_model(
        X_train_tfidf, y_train, 
        X_test_tfidf, y_test,
        time_budget=300  # 5 minutes
    )