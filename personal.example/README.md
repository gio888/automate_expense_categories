# Personal Configuration Template

This folder contains template files for setting up your personal configuration for the Expense Categorization ML Pipeline.

## Quick Setup

1. **Copy this entire folder to `personal/`:**
   ```bash
   cp -r personal.example/ personal/
   ```

2. **Edit your personal configuration files:**
   - `personal/config.yaml` - Main configuration (GCP settings, paths)
   - `personal/accounts.yaml` - Your personal accounts and staff definitions  
   - `personal/categories.yaml` - Your expense category structure

3. **Generate your category definitions:**
   ```bash
   python generate_categories.py
   ```

4. **Test your configuration:**
   ```bash
   python src/config.py
   ```

## File Descriptions

### `config.yaml`
Main configuration file containing:
- **GCP settings** - Cloud storage for model backups
- **File paths** - Where to find your transaction files
- **ML parameters** - Model training settings
- **Logging configuration**

### `accounts.yaml`  
Personal account definitions including:
- **Household staff** (if applicable) - Names and roles
- **Personal loans** - People you lend to/borrow from
- **Bank accounts** - Your actual bank account names
- **Investments** - Your investment account names

### `categories.yaml`
Expense category structure defining:
- **Category generation rules** - Which personal categories to create
- **Generic categories** - Standard expense categories
- **Enable/disable options** - Turn on/off category types you don't need

## Security Notes

✅ **Safe:** The `personal/` folder is automatically ignored by git  
✅ **Private:** Your personal data never gets committed to the repository  
✅ **Flexible:** Easy to customize for your specific needs  

## Customization Examples

### If you don't have household staff:
In `personal/categories.yaml`:
```yaml
household_staff_expenses:
  enabled: false  # Disable household staff categories
```

### If you only use one bank:
In `personal/accounts.yaml`:
```yaml
bank_accounts:
  - name: "My Bank Checking"
    type: "local"
```

### If you have different expense needs:
Add your categories to `generic_categories` in `personal/categories.yaml`

## Getting Help

- **Configuration issues:** Run `python src/config.py` to test
- **Category problems:** Run `python generate_categories.py` to regenerate
- **Setup validation:** Run `python setup_validator.py`

## Next Steps

After setup:
1. Start the web interface: `python start_web_server.py`
2. Upload your transaction files
3. Review and correct predictions
4. Retrain models for better accuracy