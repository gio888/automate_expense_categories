# Module 1: Data Loading and Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# Create plots directory if it doesn't exist
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def load_and_analyze_data(file_path):
    """
    Load the transaction data and perform initial analysis
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("Dataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Category distribution
    print("\nTop 10 categories by frequency:")
    category_counts = df['Transfer Account'].value_counts()
    print(category_counts.head(10))
    
    # Plot category distribution and save
    plt.figure(figsize=(15, 6))
    category_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Categories Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'category_distribution.png'))
    plt.close()
    
    return df

def clean_data(df):
    """
    Clean and preprocess the dataframe
    """
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Remove rows with missing descriptions or categories
    df_clean = df_clean.dropna(subset=['Description', 'Transfer Account'])
    
    # Clean descriptions
    df_clean['Description_Clean'] = df_clean['Description'].str.lower()
    df_clean['Description_Clean'] = df_clean['Description_Clean'].str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    df_clean['Description_Clean'] = df_clean['Description_Clean'].str.replace(r'\s+', ' ', regex=True)
    df_clean['Description_Clean'] = df_clean['Description_Clean'].str.strip()
    
    # Clean amounts if present
    if 'Amount' in df_clean.columns:
        df_clean['Amount'] = pd.to_numeric(df_clean['Amount'].str.replace(',', ''), errors='coerce')
    
    # Print cleaning results
    print("\nCleaning Results:")
    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(df_clean)}")
    
    # Show sample of cleaned data
    print("\nSample of cleaned descriptions:")
    sample_data = pd.DataFrame({
        'Original': df_clean['Description'].head(),
        'Cleaned': df_clean['Description_Clean'].head()
    })
    print(sample_data)
    
    return df_clean

def analyze_descriptions(df):
    """
    Analyze description patterns in the data
    """
    # Description length statistics
    df['desc_length'] = df['Description_Clean'].str.len()
    
    print("\nDescription Length Statistics:")
    print(df['desc_length'].describe())
    
    # Plot description length distribution and save
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='desc_length', bins=50)
    plt.title('Description Length Distribution')
    plt.xlabel('Description Length')
    plt.savefig(os.path.join(PLOTS_DIR, 'description_length_distribution.png'))
    plt.close()
    
    # Word count per category
    df['word_count'] = df['Description_Clean'].str.split().str.len()
    
    print("\nWord Count Statistics by Top Categories:")
    top_categories = df['Transfer Account'].value_counts().head(5).index
    print(df[df['Transfer Account'].isin(top_categories)].groupby('Transfer Account')['word_count'].describe())

if __name__ == "__main__":
    # Load and analyze
    df = load_and_analyze_data(DATA_PATH)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Analyze descriptions
    analyze_descriptions(df_clean)
    
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.")