import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import os

def load_and_split_data(data_path, min_samples=5, test_size=0.2, random_state=42):
    """
    Load data and split into train/test sets
    
    Args:
        data_path: Path to CSV file
        min_samples: Minimum number of samples per category to include
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Remove rows with missing values
    df = df.dropna(subset=['Description', 'Transfer Account'])
    
    # Clean descriptions
    df['Description_Clean'] = df['Description'].str.lower()
    df['Description_Clean'] = df['Description_Clean'].str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    df['Description_Clean'] = df['Description_Clean'].str.replace(r'\s+', ' ', regex=True)
    df['Description_Clean'] = df['Description_Clean'].str.strip()
    
    # Remove categories with too few samples
    category_counts = df['Transfer Account'].value_counts()
    valid_categories = category_counts[category_counts >= min_samples].index
    df_filtered = df[df['Transfer Account'].isin(valid_categories)]
    
    # Print category distribution
    print("\nCategory distribution after filtering:")
    print(f"Original number of categories: {len(category_counts)}")
    print(f"Categories with >= {min_samples} samples: {len(valid_categories)}")
    print(f"Remaining data points: {len(df_filtered)} of {len(df)}")
    
    print("\nTop 10 categories:")
    print(df_filtered['Transfer Account'].value_counts().head(10))
    
    # Split into features (X) and target (y)
    X = df_filtered['Description_Clean']
    y = df_filtered['Transfer Account']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def create_baseline_model():
    """
    Create a simple baseline model that always predicts the most frequent class
    """
    return DummyClassifier(strategy='most_frequent')

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance
    """
    print(f"\n{model_name} Performance:")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return classification_report(y_true, y_pred, output_dict=True)

if __name__ == "__main__":
    # Set up paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, min_samples=5)
    
    # Create and train baseline model
    baseline = create_baseline_model()
    baseline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = baseline.predict(X_test)
    
    # Evaluate baseline model
    baseline_scores = evaluate_model(y_test, y_pred, "Baseline Model")