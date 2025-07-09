# Household Expense Complete Workflow

## Prerequisites

- Ensure `data/valid_categories.txt` exists with approved categories
- Household data folder: `$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/House Expenses`
- Code location: `/Users/gio/Code/automate_expense_categories`

## Complete Monthly Workflow

### Step 1: Balance Reconciliation (⏱️ 5-10 minutes)

**Manual Step - Outside ML Pipeline**

1. Go to https://docs.google.com/spreadsheets/d/1uzkF9mCnvLsgnvJiRfIhHPLrGDmWxymWWmAGpl7zzJc
2. Check calculated balance against manually encoded balance
    - **Minor discrepancy**: Add adjustment entry to balance it out
    - **Major discrepancy**: Ask Michelle and Ara, resolve it, add adjustment entry to balance it out
    - **All good**: Proceed to next step

### Step 2: Download Monthly Data (⏱️ 2 minutes)

**Manual Step - Outside ML Pipeline**

- Download CSV from Google Sheets with naming: "House Kitty Transactions - Cash YYYY-MM.csv"
- Save to: `$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/House Expenses`

### Step 3: Transform Raw Household Data (⏱️ 2-3 minutes)

```bash
cd /Users/gio/Code/automate_expense_categories
python src/transform_monthly_household_transactions.py --source-dir "$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/House Expenses"

```

**What it does**: Converts monthly Google Sheets export to ML-ready format

- Script lists available "House Kitty Transactions - Cash" files
- Select the month you want to process
- **Output**: "House Kitty Transactions - Cash - Corrected YYYY-MM.csv"
- **Recovery**: If fails, check file exists and has columns: Date, Description, Out, In

### Step 4: Predict Categories (⏱️ 3-5 minutes)

```bash
python src/batch_predict_ensemble.py --input-dir "$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/House Expenses"

```

**What it does**: Uses trained household models to predict expense categories

- Select the transformed file from Step 3
- **Output**: `processed_[filename]_v[version]_[timestamp].csv` with predicted categories
- **Recovery**: If "models not found" error, run `python src/verify_models.py` first

### Step 5: Manual Review & Corrections (⏱️ 15-30 minutes)

**Manual Step - Outside ML Pipeline**

1. Open the processed file from Step 4
2. Go to https://docs.google.com/spreadsheets/d/1S-14MB4LxeHGJfP6BB_uWjedZNuAOxRWPt7ZjN4cM9o/
3. Navigate to appropriate month tab
4. Review predicted categories - correct any mistakes
5. Create tab named "**for merge YYYY-MM**"
6. Save corrected data as: "House Kitty AutoML Import To Gnucash - for merge YYYY-MM.csv"

**Common household category corrections**:

- "milk, bread, eggs" → "Groceries" (not "Shopping")
- "electric bill" → "Utilities" (not "Bills")
- "gas for car" → "Transportation" (not "Fuel")

### Step 6: Import to GnuCash (⏱️ 10-15 minutes)

**Manual Step - Outside ML Pipeline**

- Import corrected data to GnuCash using "XX" template
- Check for unrecognized transactions that need manual correction
- Multi-select capability available for bulk operations

### Step 7: Merge Corrections with Training Data (⏱️ 2-3 minutes)

```bash
python src/merge_training_data.py

```

**What it does**: Adds your corrections to household training dataset

- Script lists available CSV files in data folder
- Select your "House Kitty AutoML Import To Gnucash - for merge YYYY-MM.csv" file
- If missing `transaction_source`, choose 'household' when prompted
- **Output**: `training_data_household_v{N}_{timestamp}.csv`
- **Recovery**: If validation fails, check categories match `valid_categories.txt`

### Step 8: Retrain Household Models (⏱️ 15-20 minutes)

```bash
python src/auto_model_ensemble.py --source household

```

**What it does**: Creates new household models with your corrections included

- Trains 3 models: LightGBM, XGBoost, CatBoost (~5 min each)
- **Output**: New household model versions saved and registered
- **Recovery**: If fails, check `logs/training.log` and ensure >50 household records exist

## Time Summary

**Complete household cycle**: 50-75 minutes

- Steps 1-2 (Manual setup): ~7-12 minutes
- Steps 3-4 (ML prediction): ~5-8 minutes
- Steps 5-6 (Manual corrections & GnuCash): ~25-45 minutes
- Steps 7-8 (ML training): ~17-23 minutes

## Monthly Schedule

Process household expenses when:

- ✅ Monthly Google Sheets data is ready for download
- ✅ Michelle and Ara are available if balance issues arise
- ✅ You have 50-75 minutes for complete workflow
- ✅ Previous month's processing is complete

## Quick Reference Commands

```bash
# Navigate to project
cd /Users/gio/Code/automate_expense_categories

# ML Pipeline commands
python src/transform_monthly_household_transactions.py --source-dir "$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/House Expenses"
python src/batch_predict_ensemble.py --input-dir "$HOME/Library/CloudStorage/GoogleDrive-[your-email]/My Drive/Money/House Expenses"
python src/merge_training_data.py
python src/auto_model_ensemble.py --source household

```

## Key URLs

- **Balance Check**: https://docs.google.com/spreadsheets/d/1uzkF9mCnvLsgnvJiRfIhHPLrGDmWxymWWmAGpl7zzJc
- **Manual Corrections**: https://docs.google.com/spreadsheets/d/1S-14MB4LxeHGJfP6BB_uWjedZNuAOxRWPt7ZjN4cM9o/