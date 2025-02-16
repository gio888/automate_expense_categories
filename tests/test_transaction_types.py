import unittest
from datetime import datetime
from src.transaction_types import TransactionSource

class TestTransactionSource(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.household = TransactionSource.HOUSEHOLD
        self.credit_card = TransactionSource.CREDIT_CARD

    def test_enum_values(self):
        """Test basic enum value consistency"""
        self.assertEqual(self.household.value, 'household')
        self.assertEqual(self.credit_card.value, 'credit_card')
        self.assertEqual(len(TransactionSource), 2)  # Ensure only 2 sources exist

    def test_display_names(self):
        """Test display name generation"""
        self.assertEqual(
            TransactionSource.get_source_display_name(self.household),
            'Household Expenses'
        )
        self.assertEqual(
            TransactionSource.get_source_display_name(self.credit_card),
            'Credit Card Transactions'
        )
        
        # Test invalid source
        invalid_source = 'invalid'
        self.assertEqual(
            TransactionSource.get_source_display_name(invalid_source),
            'invalid'
        )

    def test_from_string(self):
        """Test string to enum conversion"""
        # Test valid sources
        self.assertEqual(
            TransactionSource.from_string('household'),
            TransactionSource.HOUSEHOLD
        )
        self.assertEqual(
            TransactionSource.from_string('credit_card'),
            TransactionSource.CREDIT_CARD
        )
        
        # Test case insensitivity
        self.assertEqual(
            TransactionSource.from_string('HOUSEHOLD'),
            TransactionSource.HOUSEHOLD
        )
        self.assertEqual(
            TransactionSource.from_string('Credit_Card'),
            TransactionSource.CREDIT_CARD
        )
        
        # Test invalid source
        self.assertIsNone(TransactionSource.from_string('invalid_source'))

    def test_source_config(self):
        """Test source configuration generation"""
        household_config = self.household.get_source_config()
        credit_card_config = self.credit_card.get_source_config()
        
        # Test household config
        self.assertEqual(household_config['min_samples_per_category'], 5)
        self.assertEqual(household_config['confidence_threshold'], 0.6)
        self.assertEqual(household_config['description_type'], 'item_level')
        self.assertEqual(household_config['vectorizer_config']['ngram_range'], (1, 2))
        self.assertEqual(household_config['vectorizer_config']['analyzer'], 'word')
        
        # Test credit card config
        self.assertEqual(credit_card_config['min_samples_per_category'], 3)
        self.assertEqual(credit_card_config['confidence_threshold'], 0.7)
        self.assertEqual(credit_card_config['description_type'], 'merchant_name')
        self.assertEqual(credit_card_config['vectorizer_config']['ngram_range'], (1, 3))
        self.assertEqual(credit_card_config['vectorizer_config']['analyzer'], 'char_wb')
        
        # Test common config elements
        for config in [household_config, credit_card_config]:
            self.assertIn('timestamp', config)
            self.assertIn('version', config)
            self.assertEqual(config['version'], '1.0')
            
            # Validate vectorizer config
            vec_config = config['vectorizer_config']
            self.assertEqual(vec_config['max_features'], 5000)
            self.assertEqual(vec_config['min_df'], 2)
            self.assertEqual(vec_config['max_df'], 0.95)

    def test_timestamp_generation(self):
        """Test timestamp generation in config"""
        # Get configs with small time delay
        config1 = self.household.get_source_config()
        config2 = self.household.get_source_config()
        
        # Convert timestamps to datetime objects
        timestamp1 = datetime.fromisoformat(config1['timestamp'])
        timestamp2 = datetime.fromisoformat(config2['timestamp'])
        
        # Ensure timestamps are different (real-time generation)
        self.assertNotEqual(timestamp1, timestamp2)
        
        # Ensure timestamps are recent
        now = datetime.now()
        self.assertLess((now - timestamp1).total_seconds(), 1)
        self.assertLess((now - timestamp2).total_seconds(), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)