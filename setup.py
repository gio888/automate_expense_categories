from setuptools import setup, find_packages

setup(
    name="automate_expense_categories",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'google-cloud-storage',
        'flaml',
        'joblib',
        'shap',
        'lightgbm',
        'xgboost',
        'catboost'
    ]
)