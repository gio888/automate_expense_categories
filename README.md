# Expense Categorization ML Pipeline

An automated machine learning pipeline for categorizing financial transactions using ensemble models and active learning. The system employs FLAML AutoML for model training and supports both local and cloud-based model storage with Google Cloud Storage integration.

## 🌟 Features

- Automated model training using FLAML AutoML
- Ensemble approach combining LightGBM, XGBoost, and CatBoost models
- Model versioning and registry system
- Google Cloud Storage integration for model backup
- Batch prediction capabilities with confidence scoring
- Performance monitoring and alerting system
- SHAP-based model interpretability
- Active learning workflow with human feedback incorporation

## 🏗️ Project Structure

```
├── data/                      # Data storage
├── logs/                      # Training and prediction logs
├── models/                    # Model storage
│   └── registry/             # Model registry files
├── plots/                    # Visualization outputs
├── src/                      # Source code
└── venv/                     # Virtual environment
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Google Cloud Storage account (optional)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Google Cloud credentials (optional):
- Place your GCP service account JSON file in a secure location
- Update the `CREDENTIALS_PATH` in relevant configuration files

## 🛠️ Usage

### Training Models

Run the automated model training pipeline:

```bash
python src/auto_model_ensemble.py
```

This will:
- Load and preprocess training data
- Train ensemble models using FLAML AutoML
- Generate SHAP explanations
- Save models locally and to GCS
- Update the model registry

### Batch Predictions

Process new transactions:

```bash
python src/batch_predict_ensemble.py
```

Features:
- Efficient batch processing
- Confidence scoring
- Performance monitoring
- Automated alerting for low-confidence predictions

## 🔄 ML Pipeline Components

### 1. Data Management
- Training data versioning
- Data validation checks
- Category distribution analysis

### 2. Model Training
- Text vectorization using TF-IDF
- Ensemble model training (LightGBM, XGBoost, CatBoost)
- Cross-validation
- Performance metric optimization

### 3. Model Registry
- Version tracking
- Model metadata storage
- Performance metrics logging
- Cloud backup integration

### 4. Prediction System
- Batch processing
- Confidence scoring
- Performance monitoring
- Alert system for low-confidence predictions

### 5. Feedback Loop
- Human review interface
- Correction tracking
- Training data updates
- Model retraining triggers

## 📊 Performance Monitoring

The system tracks:
- Overall accuracy
- Per-category F1 scores
- Prediction confidence distribution
- Category distribution drift
- Correction rates

## 🔒 Best Practices

- Comprehensive version control
- Model and data versioning
- Extensive logging
- Error handling
- Performance monitoring
- Cloud backup
- Security considerations

## 🚧 Future Improvements

1. Feature Engineering
   - Amount-based features
   - Temporal features
   - Merchant name extraction

2. Model Enhancements
   - Neural network integration
   - Active learning implementation
   - Enhanced uncertainty estimation

3. Pipeline Optimization
   - Automated retraining triggers
   - Enhanced data validation
   - Improved confidence estimation

## 📝 License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 🤝 Contributing

We welcome contributions to improve the expense categorization ML pipeline! Here's how you can help:

### Contributing Guidelines

1. **Fork the Repository**
   - Create your own fork of the code
   - Make the changes in your fork

2. **Make Your Changes**
   - Create a new branch for your feature or fix
   - Write clear, concise commit messages
   - Include tests if applicable
   - Update documentation as needed

3. **Code Style**
   - Follow PEP 8 guidelines for Python code
   - Use meaningful variable and function names
   - Add comments for complex logic
   - Maintain existing code formatting

4. **Testing**
   - Ensure all tests pass
   - Add new tests for new features
   - Test your changes with different data scenarios

5. **Submit a Pull Request**
   - Provide a clear description of the changes
   - Link any relevant issues
   - Update the README if needed
   - Fill out the pull request template

6. **Code Review**
   - Be responsive to feedback
   - Make requested changes promptly
   - Keep discussions constructive

### Reporting Issues

- Use the issue tracker to report bugs
- Include detailed steps to reproduce the issue
- Provide system information and stack traces if applicable
- Tag issues appropriately

### Feature Requests

- Use the issue tracker for feature requests
- Clearly describe the proposed functionality
- Explain the use case and benefits
- Be open to discussion and feedback

## 📫 Contact

gbacareza@gmail.com
