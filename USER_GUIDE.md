# Complete User Guide

## Getting Started

### 1. Install Dependencies

**If you're starting fresh:**
```bash
cd automate_expense_categories
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**If you already have a virtual environment:**
```bash
cd automate_expense_categories
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Make sure all dependencies are installed
```

**Quick test:**
```bash
python -c "import pandas, numpy, sklearn; print('✅ Core dependencies working')"
```

### 2. Configure Your Personal Settings
```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your Google Cloud Storage details:
```yaml
gcp:
  credentials_path: "~/.config/gcp/service-account-key.json"
  bucket_name: "your-expense-ml-models-backup"
```

**Set up your credentials securely:**
```bash
# Create secure directory
mkdir -p ~/.config/gcp/

# Move your downloaded key file (change filename as needed)
mv ~/Downloads/your-key.json ~/.config/gcp/service-account-key.json

# Set secure permissions (owner read/write only)
chmod 600 ~/.config/gcp/service-account-key.json
```

### 3. Validate Setup
```bash
python setup_validator.py
```

**If validation fails with missing packages:**
- The validator will show exactly what to install
- Usually: `pip install -r requirements.txt` fixes everything
- Make sure your virtual environment is activated first

## Daily Usage

### For Credit Card Transactions (Most Common)

**Step 1: Get your bank data ready**
- Download your credit card statement as CSV
- Save it as: `"For Automl Statement UNIONBANK Visa YYYY-MM.csv"`
- Put it in the `data/` folder

**Step 2: Run prediction**
```bash
python src/batch_predict_ensemble.py
```
- Select your CSV file when prompted
- Output: `processed_[filename]_v[version]_[timestamp].csv`

**Step 3: Review and correct predictions**
- Open the output file
- Fix any wrong categories (look for low confidence scores <70%)
- Save corrected file

**Step 4: Update training data**
```bash
python src/merge_training_data.py
```
- Select your corrected file
- Choose 'credit_card' when asked for source

**Step 5: Retrain models (monthly)**
```bash
python src/auto_model_ensemble.py --source credit_card
```

### For Household Transactions

**Step 1: Transform data**
```bash
python src/transform_monthly_household_transactions.py
```

**Step 2-5: Same as credit card steps above**
- Use `--source household` in the final step

## Key Files You'll Work With

- **Input**: Your bank CSV files in `data/` folder
- **Output**: `processed_*.csv` files with predictions
- **Categories**: `data/valid_categories.txt` (your approved categories)

## Quick Commands

```bash
# Check if everything is working
python setup_validator.py

# Predict categories for any CSV file
python src/batch_predict_ensemble.py

# After corrections, update training data
python src/merge_training_data.py

# Retrain models (do monthly)
python src/auto_model_ensemble.py --source [credit_card|household]
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Models not found" | Run `python src/verify_models.py` |
| Import errors | Make sure you're in the project folder and virtual environment is active |
| Low accuracy | More manual corrections needed, retrain models |
| File not found | Check file is in `data/` folder with correct naming |

## Time Expectations

- **Setup**: 15 minutes (one time)
- **Daily prediction**: 2-3 minutes
- **Manual corrections**: 15-25 minutes
- **Model retraining**: 15-30 minutes (monthly)

---

That's it! Start with the "Getting Started" section, then use "Daily Usage" for regular work.