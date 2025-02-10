import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os

def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Create and train a simple TF-IDF + Logistic Regression model
    """
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1000,  # Limit features to top 1000 terms
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

if __name__ == "__main__":
    # Import load_and_split_data from model_evaluation
    from model_evaluation import load_and_split_data
    
    # Set up paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, min_samples=5)
    
    # Train and evaluate model
    model = create_and_train_model(X_train, y_train, X_test, y_test)