import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

def create_category_hierarchy(df):
    """
    Create hierarchical categories from the full category path
    """
    # Split categories into levels
    df = df.copy()
    df['category_levels'] = df['Transfer Account'].str.split(':')
    
    # Create columns for each level
    df['level_1'] = df['category_levels'].apply(lambda x: x[0] if len(x) > 0 else 'Other')
    df['level_2'] = df['category_levels'].apply(lambda x: x[1] if len(x) > 1 else 'Other')
    df['level_3'] = df['category_levels'].apply(lambda x: x[2] if len(x) > 2 else 'Other')
    
    # Print hierarchy statistics
    print("\nCategory Hierarchy Statistics:")
    print(f"Level 1 categories: {df['level_1'].nunique()}")
    print(f"Level 2 categories: {df['level_2'].nunique()}")
    print(f"Level 3 categories: {df['level_3'].nunique()}")
    
    # Print sample counts for each level
    print("\nLevel 1 distribution:")
    print(df['level_1'].value_counts().head())
    print("\nTop Level 2 categories:")
    print(df['level_2'].value_counts().head())
    
    return df

def train_hierarchical_model(X_train, y_train, X_test, y_test, level):
    """
    Train a model for a specific hierarchy level
    """
    # Create and train pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced'
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Print performance
    print(f"\nLevel {level} Performance:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

if __name__ == "__main__":
    # Import load_and_split_data from model_evaluation
    from model_evaluation import load_and_split_data
    
    # Set up paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')
    
    # Load data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, min_samples=5)
    
    # Create hierarchical categories
    print("Creating category hierarchy...")
    df_train = pd.DataFrame({'Description': X_train, 'Transfer Account': y_train})
    df_test = pd.DataFrame({'Description': X_test, 'Transfer Account': y_test})
    
    df_train = create_category_hierarchy(df_train)
    df_test = create_category_hierarchy(df_test)
    
    # Train models for each level
    print("\nTraining hierarchical models...")
    for level in [1, 2, 3]:
        print(f"\nTraining Level {level} model...")
        X_train_level = df_train['Description']
        y_train_level = df_train[f'level_{level}']
        X_test_level = df_test['Description']
        y_test_level = df_test[f'level_{level}']
        
        model = train_hierarchical_model(
            X_train_level, y_train_level,
            X_test_level, y_test_level,
            level
        )
        
        # Show some example predictions
        print(f"\nExample Level {level} predictions:")
        sample_idx = np.random.randint(0, len(X_test), 5)
        for idx in sample_idx:
            desc = X_test.iloc[idx]
            pred = model.predict([desc])[0]
            true = y_test_level.iloc[idx]
            print(f"\nDescription: {desc}")
            print(f"Predicted: {pred}")
            print(f"Actual: {true}")