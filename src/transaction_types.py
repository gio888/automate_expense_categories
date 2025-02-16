from enum import Enum
from typing import Dict, Optional
import logging
from datetime import datetime

class TransactionSource(Enum):
    """Enum representing different transaction sources in the system"""
    HOUSEHOLD = 'household'
    CREDIT_CARD = 'credit_card'
    
    @classmethod
    def get_source_display_name(cls, source: 'TransactionSource') -> str:
        """Get human-readable display name for a transaction source"""
        display_names = {
            cls.HOUSEHOLD: 'Household Expenses',
            cls.CREDIT_CARD: 'Credit Card Transactions'
        }
        return display_names.get(source, source.value)
    
    @classmethod
    def from_string(cls, source_str: str) -> Optional['TransactionSource']:
        """
        Convert string to TransactionSource enum, with validation
        
        Args:
            source_str: String representation of transaction source
            
        Returns:
            TransactionSource enum value or None if invalid
        """
        try:
            return cls(source_str.lower())
        except ValueError:
            logging.warning(f"Invalid transaction source: {source_str}")
            return None
    
    def get_source_config(self) -> Dict:
        """
        Get default configuration for this source type
        
        Returns:
            Dictionary containing source-specific settings
        """
        base_config = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        source_configs = {
            self.HOUSEHOLD: {
                **base_config,
                'min_samples_per_category': 5,
                'confidence_threshold': 0.6,
                'description_type': 'item_level',
                'vectorizer_config': {
                    'ngram_range': (1, 2),
                    'analyzer': 'word',
                    'max_features': 5000,
                    'min_df': 2,
                    'max_df': 0.95
                }
            },
            self.CREDIT_CARD: {
                **base_config,
                'min_samples_per_category': 3,
                'confidence_threshold': 0.7,
                'description_type': 'merchant_name',
                'vectorizer_config': {
                    'ngram_range': (1, 3),
                    'analyzer': 'char_wb',
                    'max_features': 5000,
                    'min_df': 2,
                    'max_df': 0.95,
                    'strip_accents': 'unicode'
                }
            }
        }
        
        return source_configs.get(self, source_configs[self.HOUSEHOLD])

# Example usage
if __name__ == "__main__":
    # Create from enum
    source = TransactionSource.HOUSEHOLD
    print(f"Source: {source}")
    print(f"Display name: {TransactionSource.get_source_display_name(source)}")
    
    # Create from string
    source_str = "credit_card"
    source = TransactionSource.from_string(source_str)
    if source:
        print(f"\nSource from string: {source}")
        print(f"Config: {source.get_source_config()}")
    
    # Test invalid source
    invalid_source = TransactionSource.from_string("invalid_source")
    print(f"\nInvalid source result: {invalid_source}")