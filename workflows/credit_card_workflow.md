# Credit Card Expense Complete Workflow

## Prerequisites

- Ensure `data/valid_categories.txt` exists with approved categories
- Credit Card data folder: `$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/UnionBank`
- Code location: `$PROJECT_ROOT` (this repository)

## Complete Monthly Workflow

### Step 1: Download Monthly Data (⏱️ 2-3 minutes) - SIMPLIFIED!

**✨ Direct Excel Processing - No conversion needed!**

1. Go to online.unionbankph.com
2. Click on the credit card on the dashboard → Statements → Select statement date → Download statement → Download Excel
3. Save as: "Statement UNIONBANK Visa YYYY-MM.xlsx" to default folder
4. You will receive SMS password to open the Excel file
5. **That's it!** Use the Excel file directly in the ML pipeline

**Supported Formats:**
- ✅ Excel files (.xlsx, .xls) - Direct processing from UnionBank
- ✅ CSV files - Legacy format still supported

### Step 2: Process Transactions (⏱️ 1-2 minutes)

**🌐 Option A: Web Interface (Recommended)**
```bash
cd $PROJECT_ROOT
python start_web_server.py
```
Then open http://localhost:8000 and drag & drop your Excel or CSV file. The system will automatically:
- Detect file format (Excel/CSV) and transaction type
- Remove PHP currency symbols and clean data
- Run ensemble ML predictions with confidence scores
- Show results in an interactive correction interface

**💻 Option B: Command Line Interface**
```bash
cd $PROJECT_ROOT
python src/batch_predict_ensemble.py --file "path/to/Statement UNIONBANK Visa 2025-08.xlsx"
```

**What it does**: Uses trained credit card models to predict merchant categories

- Script will list available Excel and CSV files in the data folder
- Select your Excel statement file or CSV file from Step 1
- Automatically removes PHP currency symbols and processes amounts
- **Output**: `unionbank_visa_2025-08_predictions_YYYYMMDD_HHMMSS.csv` with predicted categories
- **Recovery**: If "models not found" error, run `python src/verify_models.py` first

### Step 3: Manual Review & Corrections (⏱️ 15-25 minutes)

**Manual Step - Outside ML Pipeline**

1. Go to: https://docs.google.com/spreadsheets/d/183Bekh9eeOJsDFPnG3DW0rarBzRqgHz_7TolIWf-n7w/edit?gid=920038515#gid=920038515
2. Replace content with the processed file from Step 2
3. Review predicted categories - correct any mistakes
4. **for_import_gnucash** tab will auto-populate
5. Download CSV: "For Import GnuCash Statement UNIONBANK Visa YYYY-MM.csv"

**Common credit card category corrections**:

- "AMAZON.COM" → "Shopping" (not "Online Services")
- "SHELL STATION" → "Transportation" (not "Gas")
- "MCDONALDS" → "Dining Out" (not "Fast Food")

### Step 4: Import to GnuCash (⏱️ 10-15 minutes)

**Manual Step - Outside ML Pipeline**

- Import "For Import GnuCash Statement UNIONBANK Visa YYYY-MM.csv" into GnuCash
- Check for unrecognized transactions that need manual correction
- Use Union Visa Template

### Step 5: Prepare Training Data (⏱️ 1 minute)

**Manual Step - Outside ML Pipeline**

1. Go back to the Google Sheets correction tab
2. **for_merge_automl** tab will auto-populate with your corrections
3. Download CSV: "For Merge Automl Statement UNIONBANK Visa YYYY-MM.csv"
4. Save to: `$PROJECT_ROOT/data`

### Step 6: Merge Corrections with Training Data (⏱️ 1 minute)

```bash
python merge_training_data.py

```

**What it does**: Adds your corrections to credit card training dataset

- Script lists available CSV files in data folder
- Select your "For Merge Automl Statement UNIONBANK Visa" file
- If missing `transaction_source`, choose 'credit_card' when prompted
- **Output**: `training_data_credit_card_v{N}_{timestamp}.csv`
- **Recovery**: If validation fails, check categories match `valid_categories.txt`

### Step 7: Retrain Credit Card Models (⏱️ 30 minutes)

```bash
python auto_model_ensemble.py --source credit_card

```

**What it does**: Creates new credit card models with your corrections included

- Trains 3 models: LightGBM, XGBoost, CatBoost (~5 min each)
- **Output**: New credit card model versions saved and registered
- **Recovery**: If fails, check `logs/training.log` and ensure >50 credit card records exist

## Time Summary

**Complete credit card cycle**: 45-65 minutes

- Step 1 (Download & setup): ~10-15 minutes
- Step 2 (ML prediction): ~3-5 minutes
- Steps 3-5 (Manual corrections & prep): ~27-40 minutes
- Steps 6-7 (ML training): ~17-23 minutes

## Monthly Schedule

Process credit card expenses when:

- ✅ Monthly UnionBank statement is available
- ✅ You have 45-65 minutes for complete workflow
- ✅ Previous month's processing is complete

## Quick Reference Commands

```bash
# Navigate to project
cd $PROJECT_ROOT

# ML Pipeline commands
python batch_predict_ensemble.py
python merge_training_data.py
python auto_model_ensemble.py --source credit_card

```

## Key URLs

- **UnionBank Online**: online.unionbankph.com
- **Data Processing**: https://docs.google.com/spreadsheets/d/183Bekh9eeOJsDFPnG3DW0rarBzRqgHz_7TolIWf-n7w/edit?gid=40583327#gid=40583327
- **Manual Corrections**: https://docs.google.com/spreadsheets/d/183Bekh9eeOJsDFPnG3DW0rarBzRqgHz_7TolIWf-n7w/edit?gid=920038515#gid=920038515