"""
Intelligent file detection and validation for transaction CSV files.
Eliminates exact naming requirements and auto-detects transaction sources.
"""

import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Import from parent directory
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.transaction_types import TransactionSource
from src.utils.excel_processor import ExcelProcessor, is_excel_file

logger = logging.getLogger(__name__)

@dataclass
class FileValidationResult:
    """Result of file validation with detailed feedback"""
    is_valid: bool
    transaction_source: Optional[TransactionSource]
    confidence: float
    issues: List[str]
    suggestions: List[str]
    detected_columns: Dict[str, str]
    row_count: int
    file_size: int

class FileDetectionError(Exception):
    """Custom exception for file detection issues"""
    pass

class FileDetector:
    """Intelligent CSV file detector for transaction data"""
    
    # Expected column patterns for different transaction sources
    HOUSEHOLD_PATTERNS = {
        'date': ['date', 'transaction_date', 'trans_date', 'timestamp'],
        'description': ['description', 'item', 'expense', 'details', 'memo'],
        'amount': ['amount', 'cost', 'price', 'value', 'expense_amount', 'out', 'in', 'debit', 'credit'],
        'category': ['category', 'type', 'classification', 'expense_type']
    }
    
    CREDIT_CARD_PATTERNS = {
        'date': ['date', 'transaction_date', 'trans_date', 'posting_date', 'purchase_date'],
        'description': ['description', 'merchant', 'vendor', 'payee', 'details'],
        'amount': ['amount', 'transaction_amount', 'charge', 'debit', 'credit'],
        'category': ['category', 'type', 'classification', 'merchant_category']
    }
    
    # Keywords that suggest transaction source
    HOUSEHOLD_KEYWORDS = [
        'house', 'kitty', 'household', 'cash', 'expenses', 'grocery',
        'utilities', 'rent', 'mortgage', 'home'
    ]
    
    CREDIT_CARD_KEYWORDS = [
        'visa', 'mastercard', 'amex', 'discover', 'card', 'credit',
        'bank', 'statement', 'purchase', 'merchant'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_and_validate(self, file_path: str) -> FileValidationResult:
        """
        Main method to detect transaction source and validate file format
        
        Args:
            file_path: Path to CSV file to analyze
            
        Returns:
            FileValidationResult with detection results and validation feedback
        """
        try:
            file_path = Path(file_path)
            
            # Basic file checks
            if not file_path.exists():
                return FileValidationResult(
                    is_valid=False,
                    transaction_source=None,
                    confidence=0.0,
                    issues=[f"File not found: {file_path}"],
                    suggestions=["Check the file path and ensure the file exists"],
                    detected_columns={},
                    row_count=0,
                    file_size=0
                )
            
            file_size = file_path.stat().st_size
            
            # Check file size
            if file_size == 0:
                return FileValidationResult(
                    is_valid=False,
                    transaction_source=None,
                    confidence=0.0,
                    issues=["File is empty"],
                    suggestions=["Ensure the CSV file contains transaction data"],
                    detected_columns={},
                    row_count=0,
                    file_size=file_size
                )
            
            # Try to read file (CSV or Excel)
            try:
                if is_excel_file(file_path):
                    # Process Excel file
                    excel_processor = ExcelProcessor()
                    df = excel_processor.process_excel_file(file_path)
                else:
                    # Process CSV file
                    df = pd.read_csv(file_path)
            except Exception as e:
                file_type = "Excel" if is_excel_file(file_path) else "CSV"
                return FileValidationResult(
                    is_valid=False,
                    transaction_source=None,
                    confidence=0.0,
                    issues=[f"Cannot read {file_type} file: {str(e)}"],
                    suggestions=[
                        f"Ensure file is in valid {file_type} format",
                        "Check for special characters or encoding issues",
                        "Try opening in Excel to verify format"
                    ],
                    detected_columns={},
                    row_count=0,
                    file_size=file_size
                )
            
            # Check if empty dataframe
            if df.empty:
                return FileValidationResult(
                    is_valid=False,
                    transaction_source=None,
                    confidence=0.0,
                    issues=["File contains no data rows"],
                    suggestions=["Ensure file has both headers and data rows"],
                    detected_columns={},
                    row_count=0,
                    file_size=file_size
                )
            
            # Detect transaction source and validate columns
            source, confidence = self._detect_transaction_source(file_path.name, df)
            column_mapping = self._detect_columns(df, source)
            issues, suggestions = self._validate_data_quality(df, column_mapping)
            
            is_valid = len(issues) == 0 and confidence > 0.5
            
            return FileValidationResult(
                is_valid=is_valid,
                transaction_source=source,
                confidence=confidence,
                issues=issues,
                suggestions=suggestions,
                detected_columns=column_mapping,
                row_count=len(df),
                file_size=file_size
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error during file detection: {e}")
            return FileValidationResult(
                is_valid=False,
                transaction_source=None,
                confidence=0.0,
                issues=[f"Unexpected error: {str(e)}"],
                suggestions=["Contact support if this error persists"],
                detected_columns={},
                row_count=0,
                file_size=0
            )
    
    def _detect_transaction_source(self, filename: str, df: pd.DataFrame) -> Tuple[Optional[TransactionSource], float]:
        """
        Detect transaction source based on filename and content analysis
        
        Returns:
            Tuple of (TransactionSource, confidence_score)
        """
        household_score = 0.0
        credit_card_score = 0.0
        
        # Analyze filename
        filename_lower = filename.lower()
        
        for keyword in self.HOUSEHOLD_KEYWORDS:
            if keyword in filename_lower:
                household_score += 0.3
        
        for keyword in self.CREDIT_CARD_KEYWORDS:
            if keyword in filename_lower:
                credit_card_score += 0.3
        
        # Analyze column names
        columns_lower = [col.lower() for col in df.columns]
        
        # Check for household-specific patterns
        if any('item' in col or 'expense' in col or 'grocery' in col for col in columns_lower):
            household_score += 0.4
        
        # Check for credit card specific patterns
        if any('merchant' in col or 'vendor' in col or 'payee' in col for col in columns_lower):
            credit_card_score += 0.4
        
        if any('card' in col or 'credit' in col for col in columns_lower):
            credit_card_score += 0.2
        
        # Analyze data content (sample first 100 rows)
        sample_df = df.head(100)
        
        # Look for typical household expense patterns
        if 'description' in columns_lower or 'item' in columns_lower:
            desc_col = next((col for col in df.columns if col.lower() in ['description', 'item', 'details']), None)
            if desc_col:
                desc_text = ' '.join(sample_df[desc_col].astype(str).str.lower())
                household_indicators = ['grocery', 'supermarket', 'gas', 'utilities', 'rent', 'insurance']
                credit_indicators = ['visa', 'mastercard', 'payment', 'purchase', 'pos']
                
                for indicator in household_indicators:
                    if indicator in desc_text:
                        household_score += 0.1
                
                for indicator in credit_indicators:
                    if indicator in desc_text:
                        credit_card_score += 0.1
        
        # Determine source based on highest score
        if household_score > credit_card_score and household_score > 0.3:
            return TransactionSource.HOUSEHOLD, min(household_score, 0.95)
        elif credit_card_score > household_score and credit_card_score > 0.3:
            return TransactionSource.CREDIT_CARD, min(credit_card_score, 0.95)
        else:
            # Default to credit card if ambiguous (more common case)
            return TransactionSource.CREDIT_CARD, 0.5
    
    def _detect_columns(self, df: pd.DataFrame, source: Optional[TransactionSource]) -> Dict[str, str]:
        """
        Detect which columns correspond to required fields

        Returns:
            Dictionary mapping field_name -> column_name
        """
        column_mapping = {}
        columns_lower = {col.lower(): col for col in df.columns}

        # Choose patterns based on detected source
        if source == TransactionSource.HOUSEHOLD:
            patterns = self.HOUSEHOLD_PATTERNS
        else:
            patterns = self.CREDIT_CARD_PATTERNS

        # Special handling for Out/In or Debit/Credit column pairs (common in household files)
        has_out = 'out' in columns_lower
        has_in = 'in' in columns_lower
        has_debit = 'debit' in columns_lower
        has_credit = 'credit' in columns_lower

        if (has_out and has_in) or (has_debit and has_credit):
            # Mark that we have valid amount columns as a pair
            column_mapping['amount'] = 'out_in_pair' if (has_out and has_in) else 'debit_credit_pair'
            column_mapping['amount_out'] = columns_lower.get('out') or columns_lower.get('debit')
            column_mapping['amount_in'] = columns_lower.get('in') or columns_lower.get('credit')

        # Find best match for each required field
        for field, possible_names in patterns.items():
            # Skip amount if we already detected a pair
            if field == 'amount' and 'amount' in column_mapping:
                continue

            best_match = None

            # Look for exact matches first
            for pattern in possible_names:
                if pattern in columns_lower:
                    best_match = columns_lower[pattern]
                    break

            # If no exact match, look for partial matches
            if not best_match:
                for pattern in possible_names:
                    for col_lower, col_original in columns_lower.items():
                        if pattern in col_lower:
                            best_match = col_original
                            break
                    if best_match:
                        break

            if best_match:
                column_mapping[field] = best_match

        return column_mapping
    
    def _validate_data_quality(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Validate data quality and provide feedback

        Returns:
            Tuple of (issues, suggestions)
        """
        issues = []
        suggestions = []

        # Check for required columns
        required_fields = ['date', 'description', 'amount']
        missing_fields = [field for field in required_fields if field not in column_mapping]

        if missing_fields:
            issues.append(f"Missing required columns: {', '.join(missing_fields)}")
            suggestions.append("Ensure your CSV has columns for date, description, and amount (or Out/In for household expenses)")
        
        # Validate data in detected columns
        for field, column in column_mapping.items():
            # Skip special marker fields (out_in_pair, debit_credit_pair, amount_out, amount_in)
            if field in ['amount_out', 'amount_in'] or column in ['out_in_pair', 'debit_credit_pair']:
                continue

            if column not in df.columns:
                continue

            # Check for mostly empty columns
            non_null_ratio = df[column].notna().sum() / len(df)
            if non_null_ratio < 0.5:
                issues.append(f"Column '{column}' is mostly empty ({non_null_ratio:.1%} filled)")
                suggestions.append(f"Ensure '{column}' column contains valid data")

            # Field-specific validations
            if field == 'date':
                self._validate_date_column(df, column, issues, suggestions)
            elif field == 'amount':
                # For amount field, validate the actual columns (could be single or pair)
                if column == 'out_in_pair' or column == 'debit_credit_pair':
                    # Validate both out/in columns
                    if 'amount_out' in column_mapping:
                        self._validate_amount_column(df, column_mapping['amount_out'], issues, suggestions)
                    if 'amount_in' in column_mapping:
                        self._validate_amount_column(df, column_mapping['amount_in'], issues, suggestions)
                else:
                    # Validate single amount column
                    self._validate_amount_column(df, column, issues, suggestions)
        
        # Check minimum row count
        if len(df) < 10:
            issues.append(f"File contains only {len(df)} rows - may be too small for ML processing")
            suggestions.append("Ensure file contains at least 10 transaction records")
        
        return issues, suggestions
    
    def _validate_date_column(self, df: pd.DataFrame, column: str, issues: List[str], suggestions: List[str]):
        """Validate date column format"""
        try:
            # Try to parse dates
            sample_dates = df[column].dropna().head(10)
            parsed_count = 0
            
            for date_val in sample_dates:
                try:
                    pd.to_datetime(date_val)
                    parsed_count += 1
                except:
                    pass
            
            if parsed_count < len(sample_dates) * 0.8:
                issues.append(f"Date column '{column}' contains unparseable dates")
                suggestions.append("Ensure dates are in recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.)")
        except:
            issues.append(f"Cannot validate date column '{column}'")
    
    def _validate_amount_column(self, df: pd.DataFrame, column: str, issues: List[str], suggestions: List[str]):
        """Validate amount column format"""
        try:
            # Try to convert to numeric
            numeric_values = pd.to_numeric(df[column], errors='coerce').dropna()
            conversion_ratio = len(numeric_values) / len(df[column].dropna())
            
            if conversion_ratio < 0.8:
                issues.append(f"Amount column '{column}' contains non-numeric values")
                suggestions.append("Ensure amounts are numeric (remove currency symbols if present)")
            
            # Check for suspicious values
            if len(numeric_values) > 0:
                # Removed zero values check - normal in double-entry accounting
                # Removed large amounts suggestion - normal in bank statements
                pass
        except:
            issues.append(f"Cannot validate amount column '{column}'")

def get_csv_files_with_metadata(directories: Optional[List[str]] = None, patterns: Optional[List[str]] = None, max_files: int = 20, sort_by: str = "modified_date") -> List[Dict[str, Any]]:
    """
    Get list of CSV files from multiple directories with metadata and validation results
    
    Args:
        directories: List of directories to search (uses config if None)
        patterns: File patterns to match (uses config if None)
        max_files: Maximum number of files to return
        sort_by: Sort method - "modified_date", "name", or "size"
    
    Returns:
        List of dictionaries with file information and validation results
    """
    from src.config import get_config
    
    # Use config if parameters not provided
    if directories is None or patterns is None:
        config = get_config()
        file_config = config.get_file_handling_config()
        if directories is None:
            directories = file_config['default_directories']
        if patterns is None:
            patterns = file_config['file_patterns']
    
    detector = FileDetector()
    files_info = []
    
    for directory in directories:
        try:
            data_dir = Path(directory)
            if not data_dir.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
            
            # Find files matching patterns
            matched_files = []
            for pattern in patterns:
                matched_files.extend(data_dir.glob(pattern))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for file_path in matched_files:
                if file_path not in seen and file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                    seen.add(file_path)
                    unique_files.append(file_path)
            
            for file_path in unique_files:
                try:
                    stat = file_path.stat()
                    validation_result = detector.detect_and_validate(str(file_path))
                    
                    # Format file size
                    size_mb = stat.st_size / (1024 * 1024)
                    if size_mb < 1:
                        size_human = f"{stat.st_size / 1024:.1f} KB"
                    else:
                        size_human = f"{size_mb:.1f} MB"
                    
                    files_info.append({
                        'name': file_path.name,
                        'path': str(file_path),
                        'directory': str(data_dir),
                        'size': stat.st_size,
                        'size_human': size_human,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                        'modified_timestamp': stat.st_mtime,
                        'validation': validation_result,
                        'is_valid': validation_result.is_valid,
                        'source': validation_result.transaction_source.value if validation_result.transaction_source else 'unknown',
                        'confidence': f"{validation_result.confidence:.1%}",
                        'row_count': validation_result.row_count
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analyzing file {file_path}: {e}")
                    files_info.append({
                        'name': file_path.name,
                        'path': str(file_path),
                        'directory': str(data_dir),
                        'size': 0,
                        'size_human': 'Unknown',
                        'modified': 'Unknown',
                        'modified_timestamp': 0,
                        'validation': None,
                        'is_valid': False,
                        'source': 'error',
                        'confidence': '0%',
                        'row_count': 0,
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
    
    # Sort files
    if sort_by == "modified_date":
        files_info.sort(key=lambda x: x.get('modified_timestamp', 0), reverse=True)
    elif sort_by == "name":
        files_info.sort(key=lambda x: x['name'].lower())
    elif sort_by == "size":
        files_info.sort(key=lambda x: x['size'], reverse=True)
    
    # Limit results
    if max_files > 0:
        files_info = files_info[:max_files]
    
    return files_info

def smart_file_selection(directories: Optional[List[str]] = None, show_all: bool = False) -> Optional[str]:
    """
    Enhanced file selection interface with validation and guidance
    
    Args:
        directories: List of directories to search (uses config if None)
        show_all: If True, show all CSV files regardless of patterns
        
    Returns:
        Path to selected file or None if cancelled
    """
    from src.config import get_config
    
    # Use config if directories not provided
    if directories is None:
        config = get_config()
        file_config = config.get_file_handling_config()
        directories = file_config['default_directories']
        max_files = file_config['max_files_shown']
        patterns = file_config['file_patterns'] if not show_all else ['*.csv', '*.xlsx', '*.xls']
    else:
        max_files = 20
        patterns = ['*.csv', '*.xlsx', '*.xls']
    
    files_info = get_csv_files_with_metadata(directories, patterns, max_files)
    
    if not files_info:
        print("❌ No matching transaction files found")
        print("📍 Searched directories:")
        for directory in directories:
            exists = "✅" if Path(directory).exists() else "❌"
            print(f"   {exists} {directory}")
        print()
        print("💡 Solutions:")
        print("   • Add transaction files to one of the directories above")
        print("   • Update file_handling.default_directories in config.yaml")
        print("   • Use --directory option to specify different location")
        return None
    
    # Group files by directory for cleaner display
    files_by_dir = {}
    for file_info in files_info:
        dir_name = file_info['directory']
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append(file_info)
    
    print(f"\n📂 Found {len(files_info)} matching transaction file(s):")
    print()
    
    valid_files = []
    file_counter = 1
    
    for directory, files in files_by_dir.items():
        # Show directory header
        dir_display = directory.replace(str(Path.home()), "~")
        print(f"📁 {dir_display} ({len(files)} files)")
        
        for file_info in files:
            status_icon = "✅" if file_info['is_valid'] else "⚠️" if file_info.get('validation') else "❌"
            source_text = file_info['source'].replace('_', ' ').title()
            
            print(f"  {file_counter}. {status_icon} {file_info['name']}")
            print(f"      📊 {file_info['row_count']} rows, {file_info['size_human']}")
            print(f"      🎯 Detected: {source_text} ({file_info['confidence']} confidence)")
            print(f"      📅 Modified: {file_info['modified']}")
            
            if file_info['is_valid']:
                valid_files.append(file_counter)
            elif file_info.get('validation'):
                validation = file_info['validation']
                if validation.issues:
                    print(f"      ⚠️  Issues: {'; '.join(validation.issues[:2])}")
            
            # Store the file counter for selection
            file_info['selection_index'] = file_counter
            file_counter += 1
            print()
    
    if not valid_files:
        print("❌ No valid transaction files found")
        print("💡 Check file format and ensure proper CSV structure")
        if not show_all:
            print("💡 Try 'all' to see all transaction files (ignoring patterns)")
        return None
    
    print("💡 Options:")
    print(f"  • Enter number (1-{len(files_info)}) to select file")
    print("  • 'r' or 'refresh' to rescan files")
    if not show_all:
        print("  • 'all' to show all transaction files (ignore patterns)")
    print("  • 'q' or 'quit' to exit")
    print()
    
    while True:
        try:
            choice = input(f"Select file (1-{len(files_info)}): ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                return None
            
            if choice in ['r', 'refresh']:
                return smart_file_selection(directories, show_all)
            
            if choice == 'all' and not show_all:
                return smart_file_selection(directories, show_all=True)
            
            try:
                selection_index = int(choice)
                
                # Find file by selection index
                selected_file = None
                for file_info in files_info:
                    if file_info['selection_index'] == selection_index:
                        selected_file = file_info
                        break
                
                if selected_file:
                    if not selected_file['is_valid']:
                        print(f"⚠️  Warning: Selected file has validation issues:")
                        if selected_file.get('validation'):
                            for issue in selected_file['validation'].issues:
                                print(f"   • {issue}")
                        
                        confirm = input("Continue anyway? (y/N): ").strip().lower()
                        if confirm not in ['y', 'yes']:
                            continue
                    
                    return selected_file['path']
                else:
                    print(f"❌ Please enter a number between 1 and {len(files_info)}")
            except ValueError:
                print("❌ Please enter a valid number, 'r' to refresh, 'all' for all files, or 'q' to quit")
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            return None

if __name__ == "__main__":
    # Test the file detector
    import argparse
    
    parser = argparse.ArgumentParser(description="Test file detection capabilities")
    parser.add_argument("--file", help="Test specific file")
    parser.add_argument("--directory", default="data", help="Directory to scan")
    
    args = parser.parse_args()
    
    if args.file:
        detector = FileDetector()
        result = detector.detect_and_validate(args.file)
        
        print(f"\n🔍 File Analysis: {args.file}")
        print(f"Valid: {result.is_valid}")
        print(f"Source: {result.transaction_source}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Rows: {result.row_count}")
        print(f"Detected columns: {result.detected_columns}")
        
        if result.issues:
            print(f"\n⚠️  Issues:")
            for issue in result.issues:
                print(f"  • {issue}")
        
        if result.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in result.suggestions:
                print(f"  • {suggestion}")
    else:
        selected = smart_file_selection(args.directory)
        if selected:
            print(f"\n✅ Selected: {selected}")