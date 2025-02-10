from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import pandas as pd
import numpy as np

def evaluate_model(automl, X_test, y_test):
    """
    Evaluate the trained model and print various performance metrics including AUC scores.
    """
    y_pred = automl.predict(X_test)
    y_pred_proba = automl.predict_proba(X_test)  # Get probability estimates

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred))

    # Compute Micro and Macro F1 Scores
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Convert labels to One-vs-Rest (OvR) format for AUC computation
    y_test_bin = pd.get_dummies(y_test)

    # Compute AUC Scores
    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
    pr_auc = average_precision_score(y_test_bin, y_pred_proba, average='macro')

    print(f"\nPerformance Metrics:")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"PR AUC Score: {pr_auc:.4f}")

    return micro_f1, macro_f1, roc_auc, pr_auc

if __name__ == "__main__":
    from model_evaluation import load_and_split_data
    from flaml import AutoML
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os

    print("Loading and preparing data...")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, min_samples=5)

    # Convert text to TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()

    # Train FLAML model
    print("Starting automated model search...")
    automl = AutoML()
    automl.fit(
        X_train_tfidf, y_train,
        task='classification',
        metric='macro_f1',
        time_budget=300,
        estimator_list=['lgbm', 'rf', 'xgboost', 'catboost', 'lrl1'],
        verbose=2
    )

    # Evaluate FLAML model
    evaluate_model(automl, X_test_tfidf, y_test)
