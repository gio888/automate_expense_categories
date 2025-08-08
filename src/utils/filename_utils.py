"""
Filename utilities for standardized naming conventions across the ML pipeline.

Provides consistent naming for all generated files with format:
{source}_{period}_{stage}_{timestamp}.csv

Where:
- source: unionbank_visa, household_cash, bpi_mastercard, etc.
- period: YYYY-MM (statement period)
- stage: input, predictions, corrected, accounting, training
- timestamp: YYYYMMDD_HHMMSS (processing time)
"""

import re
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Source name mapping for standardization
SOURCE_MAPPINGS = {
    # Credit card patterns
    'unionbank': 'unionbank_visa',
    'union bank': 'unionbank_visa', 
    'unionbank visa': 'unionbank_visa',
    'bpi': 'bpi_mastercard',
    'bpi mastercard': 'bpi_mastercard',
    'metrobank': 'metrobank_visa',
    
    # Household patterns
    'house': 'household_cash',
    'household': 'household_cash',
    'kitty': 'household_cash',
    'cash': 'household_cash',
}

def extract_source_and_period(filename: str) -> Tuple[str, Optional[str]]:
    """
    Extract source and period from various filename formats.
    
    Args:
        filename: Original filename (e.g., "For Automl Statement UNIONBANK Visa 2025-07.csv")
        
    Returns:
        Tuple of (standardized_source, period) where period is YYYY-MM format
        
    Examples:
        "For Automl Statement UNIONBANK Visa 2025-07.csv" -> ("unionbank_visa", "2025-07")
        "House Kitty Transactions - Cash 2025-08.csv" -> ("household_cash", "2025-08")
    """
    
    # Remove file extension
    name = Path(filename).stem.lower()
    
    # Extract period (YYYY-MM format)
    period_match = re.search(r'(\d{4}-\d{2})', name)
    period = period_match.group(1) if period_match else None
    
    # Extract source from filename
    source = "unknown"
    
    # Try to match known patterns
    for pattern, standard_name in SOURCE_MAPPINGS.items():
        if pattern in name:
            source = standard_name
            break
    
    # If no match, try to extract from common filename patterns
    if source == "unknown":
        # Pattern: "automl statement [BANK] [CARD] YYYY-MM"
        automl_match = re.search(r'automl.*statement.*?([a-z]+(?:\s+[a-z]+)?)', name)
        if automl_match:
            bank_text = automl_match.group(1).strip()
            source = standardize_source_name(bank_text)
        
        # Pattern: "house kitty" or variations
        elif any(word in name for word in ['house', 'kitty', 'household']):
            source = "household_cash"
    
    return source, period

def standardize_source_name(source_text: str) -> str:
    """
    Convert source text to standardized lowercase_underscore format.
    
    Args:
        source_text: Raw source text from filename
        
    Returns:
        Standardized source name
    """
    
    # Clean and normalize
    clean_text = re.sub(r'[^\w\s]', '', source_text.lower().strip())
    
    # Check mappings first
    if clean_text in SOURCE_MAPPINGS:
        return SOURCE_MAPPINGS[clean_text]
    
    # Default transformations
    # Convert spaces to underscores
    standardized = re.sub(r'\s+', '_', clean_text)
    
    # Handle common bank name patterns
    if 'union' in standardized and 'bank' in standardized:
        return 'unionbank_visa'
    elif 'bpi' in standardized:
        return 'bpi_mastercard'
    elif any(word in standardized for word in ['house', 'kitty', 'household', 'cash']):
        return 'household_cash'
    
    return standardized if standardized else 'unknown'

def generate_filename(
    source: str, 
    period: Optional[str], 
    stage: str, 
    timestamp: Optional[datetime] = None,
    extension: str = "csv"
) -> str:
    """
    Generate standardized filename using the new naming convention.
    
    Args:
        source: Standardized source name (e.g., "unionbank_visa")
        period: Statement period in YYYY-MM format
        stage: Processing stage (input, predictions, corrected, accounting, training)
        timestamp: Processing timestamp (defaults to now)
        extension: File extension (defaults to "csv")
        
    Returns:
        Standardized filename
        
    Example:
        generate_filename("unionbank_visa", "2025-07", "predictions")
        -> "unionbank_visa_2025-07_predictions_20250808_194140.csv"
    """
    
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Build filename components
    components = [source]
    
    if period:
        components.append(period)
    
    components.extend([stage, timestamp_str])
    
    filename = "_".join(components)
    
    return f"{filename}.{extension}"

def parse_standardized_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse a standardized filename back into its components.
    
    Args:
        filename: Standardized filename
        
    Returns:
        Dictionary with keys: source, period, stage, timestamp
        
    Example:
        parse_standardized_filename("unionbank_visa_2025-07_predictions_20250808_194140.csv")
        -> {"source": "unionbank_visa", "period": "2025-07", "stage": "predictions", "timestamp": "20250808_194140"}
    """
    
    # Remove extension
    name = Path(filename).stem
    
    # Split by underscores
    parts = name.split('_')
    
    result = {
        "source": None,
        "period": None, 
        "stage": None,
        "timestamp": None
    }
    
    if len(parts) >= 4:
        # Expected format: source_period_stage_YYYYMMDD_HHMMSS
        # Timestamp is actually the last TWO parts joined
        
        # Last two parts form the timestamp (YYYYMMDD_HHMMSS)
        if len(parts) >= 2:
            result["timestamp"] = f"{parts[-2]}_{parts[-1]}"
        
        # Third to last is the stage
        if len(parts) >= 3:
            result["stage"] = parts[-3]
        
        # Everything before stage could be source and period
        remaining_parts = parts[:-3] if len(parts) >= 3 else []
        
        # Look for period pattern (YYYY-MM) in remaining parts
        period_index = None
        for i, part in enumerate(remaining_parts):
            if re.match(r'\d{4}-\d{2}', part):
                result["period"] = part
                period_index = i
                break
        
        # Source is everything before the period (or everything if no period)
        if period_index is not None:
            source_parts = remaining_parts[:period_index]
        else:
            source_parts = remaining_parts
            
        if source_parts:
            result["source"] = "_".join(source_parts)
    
    return result

def get_file_stage_sequence() -> list:
    """Return the standard sequence of file processing stages."""
    return ["input", "predictions", "corrected", "accounting", "training"]

def handle_legacy_filename(legacy_filename: str, target_stage: str) -> str:
    """
    Handle legacy format filenames that can't be parsed by standard utilities.
    
    Args:
        legacy_filename: Old format filename like 'corrected_job_file_1_20250809_005040_005042.csv'
        target_stage: The stage we want to convert to (e.g., 'accounting')
        
    Returns:
        A fallback filename in standardized format
    """
    
    # Remove extension
    name = Path(legacy_filename).stem
    
    # Try to extract some useful info from legacy format
    parts = name.split('_')
    
    # Look for date pattern (YYYYMMDD)
    date_part = None
    time_parts = []
    
    for i, part in enumerate(parts):
        # Look for YYYYMMDD pattern
        if re.match(r'\d{8}', part):
            date_part = part
            # Collect any following HHMMSS parts
            for j in range(i+1, len(parts)):
                if re.match(r'\d{6}', parts[j]):
                    time_parts.append(parts[j])
                else:
                    break
            break
    
    # Generate fallback filename
    source = "legacy_file"  # Generic source for legacy files
    period = None  # No period info available in legacy format
    
    # Use current timestamp if we can't parse the legacy timestamp
    if date_part and time_parts:
        try:
            timestamp = datetime.strptime(f"{date_part}_{time_parts[0]}", "%Y%m%d_%H%M%S")
        except:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()
    
    return generate_filename(source, period, target_stage, timestamp)

def validate_filename_format(filename: str) -> bool:
    """
    Validate if filename follows the standardized convention.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if filename follows convention, False otherwise
    """
    
    parsed = parse_standardized_filename(filename)
    
    # Must have source and stage
    if not parsed["source"] or not parsed["stage"]:
        return False
    
    # Stage must be valid
    valid_stages = get_file_stage_sequence()
    if parsed["stage"] not in valid_stages:
        return False
    
    # Timestamp must be valid format if present
    if parsed["timestamp"]:
        try:
            datetime.strptime(parsed["timestamp"], "%Y%m%d_%H%M%S")
        except ValueError:
            return False
    
    # Period must be YYYY-MM format if present
    if parsed["period"]:
        if not re.match(r'\d{4}-\d{2}', parsed["period"]):
            return False
    
    return True

# Convenience functions for common use cases

def create_predictions_filename(original_filename: str, timestamp: Optional[datetime] = None) -> str:
    """Create predictions filename from original upload filename."""
    source, period = extract_source_and_period(original_filename)
    return generate_filename(source, period, "predictions", timestamp)

def create_corrected_filename(predictions_filename: str) -> str:
    """Create corrected filename from predictions filename."""
    parsed = parse_standardized_filename(predictions_filename)
    return generate_filename(
        parsed["source"], 
        parsed["period"], 
        "corrected", 
        datetime.strptime(parsed["timestamp"], "%Y%m%d_%H%M%S") if parsed["timestamp"] else None
    )

def create_accounting_filename(input_filename: str) -> str:
    """Create accounting filename from corrected or predictions filename.""" 
    parsed = parse_standardized_filename(input_filename)
    
    # Handle legacy format files that can't be parsed properly
    if not parsed["source"] or not parsed["stage"]:
        return handle_legacy_filename(input_filename, "accounting")
    
    # Try to parse timestamp, use fallback if it fails
    timestamp = None
    if parsed["timestamp"]:
        try:
            timestamp = datetime.strptime(parsed["timestamp"], "%Y%m%d_%H%M%S")
        except ValueError:
            # Timestamp parsing failed - this is likely a legacy file
            logger.warning(f"Could not parse timestamp '{parsed['timestamp']}' in filename '{input_filename}', using fallback")
            return handle_legacy_filename(input_filename, "accounting")
    
    return generate_filename(
        parsed["source"],
        parsed["period"], 
        "accounting",
        timestamp
    )

def create_accounting_from_predictions_filename(predictions_filename: str) -> str:
    """Create accounting filename from predictions filename."""
    # This is now the same as create_accounting_filename since we made it more flexible
    return create_accounting_filename(predictions_filename)

def create_training_filename(corrected_filename: str) -> str:
    """Create training data filename from corrected filename."""
    parsed = parse_standardized_filename(corrected_filename)
    return generate_filename(
        parsed["source"],
        parsed["period"],
        "training", 
        datetime.strptime(parsed["timestamp"], "%Y%m%d_%H%M%S") if parsed["timestamp"] else None
    )

if __name__ == "__main__":
    # Test the filename utilities
    test_files = [
        "For Automl Statement UNIONBANK Visa 2025-07.csv",
        "House Kitty Transactions - Cash 2025-08.csv", 
        "BPI Mastercard Statement 2025-06.csv"
    ]
    
    print("Testing filename extraction and generation:")
    print("=" * 50)
    
    for filename in test_files:
        source, period = extract_source_and_period(filename)
        predictions_file = create_predictions_filename(filename)
        corrected_file = create_corrected_filename(predictions_file)
        accounting_file = create_accounting_filename(corrected_file)
        
        print(f"Original: {filename}")
        print(f"Source: {source}, Period: {period}")
        print(f"Predictions: {predictions_file}")
        print(f"Corrected: {corrected_file}")
        print(f"Accounting: {accounting_file}")
        print(f"Valid format: {validate_filename_format(predictions_file)}")
        print("-" * 30)