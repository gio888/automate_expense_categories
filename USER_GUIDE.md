# Complete User Guide

## Choose Your Interface

### 🌐 Web Interface (Recommended)
The easiest way to use the expense categorization system is through the web interface:

```bash
python start_web_server.py
```

Then open **http://localhost:8000** in your browser.

**Features:**
- 📁 Drag & drop file upload
- 🤖 Automatic transaction type detection
- 📊 Real-time processing with progress bars
- ✏️ Interactive correction interface with searchable categories
- 📥 Download results in multiple formats
- 🔄 Automatic model retraining with corrections
- 💻 Works on any device with a web browser

### 💻 Command Line Interface
For advanced users or automation, use the CLI commands described below.

---

## Getting Started

### 1. Install Dependencies

**If you're starting fresh:**
```bash
cd automate_expense_categories
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**If you already have a virtual environment:**
```bash
cd automate_expense_categories
source .venv/bin/activate  
pip install -r requirements.txt  # Make sure all dependencies are installed
```

**Quick test:**
```bash
python -c "import pandas, numpy, sklearn; print('✅ Core dependencies working')"
```

### 2. Set Up Personal Configuration

**Copy personal configuration templates:**
```bash
cp -r personal.example/ personal/
```

**Configure your settings:**
```bash
# Edit your GCP credentials and settings
nano personal/config.yaml

# Define your bank accounts and staff (if applicable)
nano personal/accounts.yaml

# Customize your expense categories
nano personal/categories.yaml
```

**Generate your category definitions:**
```bash
python generate_categories.py
```

**Set up your credentials securely:**
```bash
# Create secure directory
mkdir -p ~/.config/gcp/

# Move your downloaded GCP key file
mv ~/Downloads/your-key.json ~/.config/gcp/service-account-key.json

# Set secure permissions
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

### 🌐 Using the Web Interface (Recommended)

**Step 1: Start the web server**
```bash
python start_web_server.py
```

**Step 2: Open your browser**
- Go to http://localhost:8000
- The interface will guide you through the process

**Step 3: Upload and process**
1. Drag & drop your CSV file (or click "Choose File")
2. The system automatically detects if it's household or credit card data
3. Click "Process" to run ML predictions
4. Review predictions with confidence scores
5. Make corrections using the searchable dropdown menus
6. Download results in your preferred format
7. Optional: Retrain models with your corrections

**That's it!** The web interface handles the entire workflow seamlessly.

---

### 💻 Command Line Interface

#### For Credit Card Transactions (Most Common)

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

#### For Household Transactions

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

### Web Interface
```bash
# Start the web interface
python start_web_server.py
# Then open http://localhost:8000
```

### Command Line
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
| Web interface won't start | Check `pip install fastapi uvicorn` and virtual environment is active |
| "Models not found" | Run `python src/verify_models.py` |
| Import errors | Make sure you're in the project folder and virtual environment is active |
| Low accuracy | More manual corrections needed, retrain models |
| File not found | Check file is in `data/` folder with correct naming |
| Browser shows "Connection refused" | Ensure web server is running with `python start_web_server.py` |
| File upload fails | Check file is CSV format and less than 50MB |

## Time Expectations

### Web Interface
- **Setup**: 15 minutes (one time)
- **Daily prediction & corrections**: 5-10 minutes
- **Model retraining**: 1-2 minutes (automatic)

### Command Line Interface
- **Setup**: 15 minutes (one time)
- **Daily prediction**: 2-3 minutes
- **Manual corrections**: 15-25 minutes
- **Model retraining**: 15-30 minutes (monthly)

---

That's it! Start with the "Getting Started" section, then use "Daily Usage" for regular work.