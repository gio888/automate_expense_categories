from enum import Enum

class TransactionSource(Enum):
    HOUSEHOLD = 'household'
    CREDIT_CARD = 'credit_card'
    
    @classmethod
    def get_source_display_name(cls, source):
        return {
            cls.HOUSEHOLD: 'Household Expenses',
            cls.CREDIT_CARD: 'Credit Card Transactions'
        }.get(source, source.value)
