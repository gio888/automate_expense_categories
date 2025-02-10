import numpy as np
import joblib
import os
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import pandas as pd
from scipy.stats import mode
from flaml import AutoML
from model_evaluation import load_and_split_data
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Set up correct paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Kitty Transactions for automl 2019-2023 - Sheet1.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)  # ✅ Ensure models/ directory exists

# ✅ Load and split dataset
print("📂 Loading and preparing data...")
X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, min_samples=5)

# ✅ Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# ✅ Save TF-IDF vectorizer for later use
tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
joblib.dump(tfidf, tfidf_path)
print(f"✅ Saved TF-IDF vectorizer: {tfidf_path}")

# ✅ Train and save each model
models = []
for estimator in ['lgbm', 'xgboost', 'catboost']:
    print(f"🚀 Training {estimator}...")
    automl = AutoML()
    automl.fit(
        X_train_tfidf, y_train,
        task='classification',
        metric='macro_f1',
        time_budget=100,
        estimator_list=[estimator],
        verbose=2
    )
    models.append(automl)

    # ✅ Save each trained model properly
    model_filename = os.path.join(MODEL_DIR, f"{estimator}_model.pkl")
    joblib.dump(automl, model_filename)
    print(f"✅ Model saved: {model_filename}")

print("\n🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY! 🎉\n")

# ✅ Evaluate Ensemble Model
def evaluate_ensemble(models, X_test, y_test):
    """
    Evaluate an ensemble model using multiple learners and compute AUC scores.
    """
    y_preds = np.column_stack([model.predict(X_test) for model in models])
    y_probas = np.mean([model.predict_proba(X_test) for model in models], axis=0)  # Average probabilities
    y_pred = mode(y_preds, axis=1).mode.flatten()

    print("\n📊 Ensemble Model Performance:")
    print(classification_report(y_test, y_pred))

    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    y_test_bin = pd.get_dummies(y_test)
    roc_auc = roc_auc_score(y_test_bin, y_probas, average='macro', multi_class='ovr')
    pr_auc = average_precision_score(y_test_bin, y_probas, average='macro')

    print(f"\n📈 Ensemble Performance Metrics:")
    print(f"✅ Micro F1 Score: {micro_f1:.4f}")
    print(f"✅ Macro F1 Score: {macro_f1:.4f}")
    print(f"✅ ROC AUC Score: {roc_auc:.4f}")
    print(f"✅ PR AUC Score: {pr_auc:.4f}")

evaluate_ensemble(models, X_test_tfidf, y_test)
