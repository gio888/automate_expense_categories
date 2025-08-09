"""
Excel file processor for bank statements, specifically UnionBank credit card statements.
Handles Excel format conversion to standardized DataFrame for the ML pipeline.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExcelProcessorError(Exception):
    """Custom exception for Excel processing issues"""
    pass

class ExcelProcessor:
    """
    Processes Excel files from banks, specifically UnionBank credit card statements.
    Converts to standardized format for the expense categorization pipeline.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_excel_file(self, file_path: str) -> pd.DataFrame:
        """
        Process Excel file and return standardized DataFrame
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            DataFrame with standardized columns: date, description, amount
            
        Raises:
            ExcelProcessorError: If processing fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise ExcelProcessorError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in ['.xlsx', '.xls']:
                raise ExcelProcessorError(f"File is not an Excel file: {file_path}")
            
            # Detect the bank format and process accordingly
            if self._is_unionbank_format(file_path):
                return self._process_unionbank_excel(file_path)
            else:
                # Try generic Excel processing
                return self._process_generic_excel(file_path)
                
        except Exception as e:
            if isinstance(e, ExcelProcessorError):
                raise
            raise ExcelProcessorError(f"Error processing Excel file {file_path}: {str(e)}")
    
    def _is_unionbank_format(self, file_path: Path) -> bool:
        """
        Detect if this is a UnionBank credit card statement format
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            True if UnionBank format detected
        """
        try:
            # Read first few rows to check format
            df_header = pd.read_excel(file_path, nrows=15)
            
            # Check for UnionBank specific patterns
            unionbank_indicators = [
                'ACCOUNT NAME',
                'CARD NUMBER', 
                'STATEMENT BALANCE',
                'PAYMENT DUE DATE',
                'TOTAL AMOUNT DUE'
            ]
            
            # Convert to string and check if indicators are present
            header_text = df_header.to_string().upper()
            
            matches = sum(1 for indicator in unionbank_indicators if indicator in header_text)
            return matches >= 3  # At least 3 indicators should match
            
        except Exception as e:
            self.logger.warning(f"Error detecting UnionBank format: {e}")
            return False
    
    def _process_unionbank_excel(self, file_path: Path) -> pd.DataFrame:
        """
        Process UnionBank credit card statement Excel file
        
        Expected format:
        - Rows 1-11: Account information and summary
        - Row 13: Headers (DATE, DESCRIPTION, AMOUNT)
        - Row 14+: Transaction data
        
        Args:
            file_path: Path to UnionBank Excel file
            
        Returns:
            Standardized DataFrame with cleaned data
        """
        try:
            # Read the Excel file, skipping the header rows
            # UnionBank transactions start around row 14 (0-indexed = 13)
            # Use header=None since there are no column headers in the transaction section
            df = pd.read_excel(file_path, skiprows=13, header=None)
            
            # Manually assign column names based on UnionBank format
            # The format should be: DATE, DESCRIPTION, AMOUNT
            if df.shape[1] >= 3:
                df.columns = ['date', 'description', 'amount'] + [f'col_{i}' for i in range(3, df.shape[1])]
                # Keep only the first 3 columns we care about
                df = df[['date', 'description', 'amount']]
            else:
                raise ExcelProcessorError(
                    f"Expected at least 3 columns for DATE, DESCRIPTION, AMOUNT. "
                    f"Found {df.shape[1]} columns."
                )
            
            # Clean the data
            result_df = self._clean_unionbank_data(df)
            
            # Remove empty rows
            result_df = result_df.dropna(subset=['date', 'description', 'amount'])
            
            if result_df.empty:
                raise ExcelProcessorError("No valid transaction data found after processing")
            
            self.logger.info(f"Successfully processed {len(result_df)} transactions from UnionBank Excel file")
            return result_df
            
        except Exception as e:
            if isinstance(e, ExcelProcessorError):
                raise
            raise ExcelProcessorError(f"Error processing UnionBank Excel file: {str(e)}")
    
    def _clean_unionbank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean UnionBank specific data formats
        
        Args:
            df: DataFrame with raw UnionBank data
            
        Returns:
            DataFrame with cleaned data
        """
        df = df.copy()
        
        # Clean amount column - remove PHP currency symbol and commas
        if 'amount' in df.columns:
            df['amount'] = df['amount'].astype(str)
            
            # Remove PHP prefix and clean formatting
            df['amount'] = df['amount'].str.replace('PHP ', '', regex=False)
            df['amount'] = df['amount'].str.replace(',', '', regex=False)
            df['amount'] = df['amount'].str.strip()
            
            # Convert to numeric, handling any remaining non-numeric values
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Clean date column
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except Exception as e:
                self.logger.warning(f"Error parsing dates: {e}")
        
        # Clean description column
        if 'description' in df.columns:
            df['description'] = df['description'].astype(str).str.strip()
        
        return df
    
    def _process_generic_excel(self, file_path: Path) -> pd.DataFrame:
        """
        Process generic Excel file format
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Try to read the Excel file
            df = pd.read_excel(file_path)
            
            # Try to detect columns automatically
            detected_columns = self._detect_columns_generic(df)
            
            if not detected_columns:
                raise ExcelProcessorError("Could not detect required columns (date, description, amount)")
            
            # Map to standard format
            result_df = pd.DataFrame()
            result_df['date'] = df[detected_columns.get('date', df.columns[0])]
            result_df['description'] = df[detected_columns.get('description', df.columns[1])]  
            result_df['amount'] = df[detected_columns.get('amount', df.columns[-1])]
            
            # Clean generic data
            result_df = self._clean_generic_data(result_df)
            
            # Remove empty rows
            result_df = result_df.dropna(subset=['date', 'description', 'amount'])
            
            if result_df.empty:
                raise ExcelProcessorError("No valid transaction data found after processing")
            
            self.logger.info(f"Successfully processed {len(result_df)} transactions from generic Excel file")
            return result_df
            
        except Exception as e:
            if isinstance(e, ExcelProcessorError):
                raise
            raise ExcelProcessorError(f"Error processing generic Excel file: {str(e)}")
    
    def _detect_columns_generic(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect columns in generic Excel format
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping field_name -> column_name
        """
        column_mapping = {}
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Date patterns
        date_patterns = ['date', 'transaction_date', 'trans_date', 'posting_date']
        for pattern in date_patterns:
            if pattern in columns_lower:
                column_mapping['date'] = columns_lower[pattern]
                break
        
        # Description patterns  
        desc_patterns = ['description', 'merchant', 'vendor', 'payee', 'details', 'item']
        for pattern in desc_patterns:
            if pattern in columns_lower:
                column_mapping['description'] = columns_lower[pattern]
                break
        
        # Amount patterns
        amount_patterns = ['amount', 'transaction_amount', 'charge', 'debit', 'credit', 'value']
        for pattern in amount_patterns:
            if pattern in columns_lower:
                column_mapping['amount'] = columns_lower[pattern]
                break
        
        return column_mapping
    
    def _clean_generic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean generic Excel data
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with cleaned data
        """
        df = df.copy()
        
        # Clean amount column
        if 'amount' in df.columns:
            df['amount'] = df['amount'].astype(str)
            
            # Remove common currency symbols and formatting
            currency_symbols = ['$', '€', '£', '¥', '₱', 'PHP', 'USD', 'EUR']
            for symbol in currency_symbols:
                df['amount'] = df['amount'].str.replace(symbol, '', regex=False)
            
            df['amount'] = df['amount'].str.replace(',', '', regex=False)
            df['amount'] = df['amount'].str.replace('(', '-', regex=False)  # Handle negative amounts in parentheses
            df['amount'] = df['amount'].str.replace(')', '', regex=False)
            df['amount'] = df['amount'].str.strip()
            
            # Convert to numeric
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Clean date column
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except Exception as e:
                self.logger.warning(f"Error parsing dates: {e}")
        
        # Clean description column
        if 'description' in df.columns:
            df['description'] = df['description'].astype(str).str.strip()
        
        return df

def is_excel_file(file_path: str) -> bool:
    """
    Check if file is an Excel file based on extension
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file has Excel extension
    """
    return Path(file_path).suffix.lower() in ['.xlsx', '.xls']

def convert_excel_to_csv_format(file_path: str) -> pd.DataFrame:
    """
    Convert Excel file to CSV-like DataFrame format for compatibility
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        DataFrame in CSV format
    """
    processor = ExcelProcessor()
    return processor.process_excel_file(file_path)

if __name__ == "__main__":
    # Test the Excel processor
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Excel processing capabilities")
    parser.add_argument("file", help="Excel file to process")
    parser.add_argument("--output", help="Output CSV file (optional)")
    
    args = parser.parse_args()
    
    try:
        processor = ExcelProcessor()
        df = processor.process_excel_file(args.file)
        
        print(f"\n✅ Successfully processed Excel file: {args.file}")
        print(f"📊 Rows: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"\n📝 First 5 rows:")
        print(df.head())
        
        if args.output:
            df.to_csv(args.output, index=False, encoding='utf-8')
            print(f"\n💾 Saved to: {args.output}")
            
    except ExcelProcessorError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")