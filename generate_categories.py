#!/usr/bin/env python3
"""
Generate valid_categories.txt from personal configuration files.

This script reads personal/accounts.yaml and personal/categories.yaml
and generates a comprehensive valid_categories.txt file.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Set, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_personal_accounts() -> Dict[str, Any]:
    """Load personal accounts configuration"""
    accounts_file = PROJECT_ROOT / "personal" / "accounts.yaml"
    if not accounts_file.exists():
        print(f"❌ Personal accounts file not found: {accounts_file}")
        print("💡 Create personal/accounts.yaml with your account definitions")
        return {}
    
    try:
        with open(accounts_file, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ Error loading personal accounts: {e}")
        return {}

def load_personal_categories() -> Dict[str, Any]:
    """Load personal categories configuration"""
    categories_file = PROJECT_ROOT / "personal" / "categories.yaml"
    if not categories_file.exists():
        print(f"❌ Personal categories file not found: {categories_file}")
        print("💡 Create personal/categories.yaml with your category structure")
        return {}
    
    try:
        with open(categories_file, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ Error loading personal categories: {e}")
        return {}

def generate_household_staff_categories(accounts: Dict[str, Any]) -> List[str]:
    """Generate household staff expense categories"""
    categories = []
    household_staff = accounts.get('household_staff', [])
    
    for staff in household_staff:
        name = staff.get('name')
        roles = staff.get('roles', [])
        
        # Base staff category
        categories.append(f"Expenses:Household Staff:{name}")
        
        # Staff role categories
        for role in roles:
            categories.append(f"Expenses:Household Staff:{name}:{role.title()}")
    
    return categories

def generate_loan_categories(accounts: Dict[str, Any]) -> List[str]:
    """Generate loan and asset categories"""
    categories = []
    personal_loans = accounts.get('personal_loans', [])
    
    for loan in personal_loans:
        name = loan.get('name')
        loan_types = loan.get('types', [])
        
        # Base loan categories
        categories.append(f"Assets:Loans to:{name}")
        categories.append(f"Liabilities:Other:{name}")
        
        # Loan type categories
        for loan_type in loan_types:
            categories.append(f"Liabilities:Other:{name}:{loan_type.title()}")
    
    return categories

def generate_bank_account_categories(accounts: Dict[str, Any]) -> List[str]:
    """Generate bank account categories"""
    categories = []
    bank_accounts = accounts.get('bank_accounts', [])
    
    for account in bank_accounts:
        name = account.get('name')
        account_type = account.get('type', 'local')
        
        if account_type == 'local':
            categories.append(f"Assets:Current Assets:Banks Local:{name}")
        else:  # foreign
            categories.append(f"Assets:Current Assets:Banks Foreign:{name}")
    
    return categories

def generate_investment_categories(accounts: Dict[str, Any]) -> List[str]:
    """Generate investment categories"""
    categories = []
    investments = accounts.get('investments', [])
    
    for investment in investments:
        name = investment.get('name')
        categories.append(f"Assets:Investments:{name}")
    
    return categories

def generate_other_account_categories(accounts: Dict[str, Any]) -> List[str]:
    """Generate other account categories"""
    categories = []
    other_accounts = accounts.get('other_accounts', [])
    
    for account in other_accounts:
        name = account.get('name')
        categories.append(f"Liabilities:Other:{name}")
    
    return categories

def generate_all_categories() -> List[str]:
    """Generate complete list of valid categories"""
    print("📋 Loading personal configuration...")
    
    # Load personal configurations
    accounts = load_personal_accounts()
    categories_config = load_personal_categories()
    
    if not accounts or not categories_config:
        print("❌ Could not load personal configurations")
        return []
    
    all_categories = set()
    
    # Add generic categories (no personal data)
    print("📝 Adding generic categories...")
    generic_categories = categories_config.get('generic_categories', [])
    all_categories.update(generic_categories)
    print(f"   Added {len(generic_categories)} generic categories")
    
    # Generate personal categories based on configuration
    category_structure = categories_config.get('category_structure', {})
    
    # Household staff expenses
    if category_structure.get('household_staff_expenses', {}).get('enabled', False):
        print("👥 Generating household staff categories...")
        staff_categories = generate_household_staff_categories(accounts)
        all_categories.update(staff_categories)
        print(f"   Added {len(staff_categories)} staff categories")
    
    # Personal loans
    if category_structure.get('personal_loans', {}).get('enabled', False):
        print("💰 Generating loan categories...")
        loan_categories = generate_loan_categories(accounts)
        all_categories.update(loan_categories)
        print(f"   Added {len(loan_categories)} loan categories")
    
    # Bank accounts
    if category_structure.get('bank_accounts', {}).get('enabled', False):
        print("🏦 Generating bank account categories...")
        bank_categories = generate_bank_account_categories(accounts)
        all_categories.update(bank_categories)
        print(f"   Added {len(bank_categories)} bank categories")
    
    # Investments
    if category_structure.get('investments', {}).get('enabled', False):
        print("📈 Generating investment categories...")
        investment_categories = generate_investment_categories(accounts)
        all_categories.update(investment_categories)
        print(f"   Added {len(investment_categories)} investment categories")
    
    # Other accounts
    other_categories = generate_other_account_categories(accounts)
    all_categories.update(other_categories)
    print(f"🔧 Added {len(other_categories)} other account categories")
    
    # Convert to sorted list
    return sorted(list(all_categories))

def write_categories_file(categories: List[str], output_file: Path):
    """Write categories to file"""
    try:
        with open(output_file, 'w') as f:
            for category in categories:
                f.write(f"{category}\n")
        print(f"✅ Generated {output_file} with {len(categories)} categories")
    except Exception as e:
        print(f"❌ Error writing categories file: {e}")

def main():
    """Main function"""
    print("🎯 Generating valid_categories.txt from personal configuration")
    print("=" * 60)
    
    # Generate categories
    categories = generate_all_categories()
    
    if not categories:
        print("\n❌ No categories generated!")
        print("💡 Make sure personal/accounts.yaml and personal/categories.yaml exist")
        sys.exit(1)
    
    # Write to data/valid_categories.txt
    output_file = PROJECT_ROOT / "data" / "valid_categories.txt" 
    write_categories_file(categories, output_file)
    
    print(f"\n✅ Successfully generated {len(categories)} categories!")
    print(f"📁 Output: {output_file}")
    print("\n💡 This file is now automatically generated from your personal configuration.")
    print("   To update categories, edit personal/accounts.yaml or personal/categories.yaml")
    print("   and run this script again.")

if __name__ == "__main__":
    main()