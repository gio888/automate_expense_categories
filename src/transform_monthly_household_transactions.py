import pandas as pd
import os
import glob
import re
from datetime import datetime
import argparse

# Get project root directory
def get_project_root():
    """Find the project root directory"""
    script_path = os.path.abspath(__file__)
    
    # If running from src directory
    if os.path.basename(os.path.dirname(script_path)) == 'src':
        return os.path.dirname(os.path.dirname(script_path))
    
    # Otherwise, assume current directory might be project root
    return os.path.dirname(script_path)

# Setup paths based on project structure
PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def list_monthly_files(data_dir):
    """List only files with 'House Kitty Transactions - Cash' pattern"""
    # Search for all CSV files and filter for only House Kitty files
    all_csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Filter to include ONLY "House Kitty Transactions - Cash" files
    valid_files = []
    for file in all_csv_files:
        basename = os.path.basename(file)
        # Only include "House Kitty Transactions - Cash" files
        if "House Kitty Transactions - Cash" in basename:
            valid_files.append(file)
    
    return valid_files

def transform_monthly_log(input_file, output_dir=None):
    """
    Transform monthly household expense logs into the format compatible with the ML pipeline.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str, optional): Directory to save the output file
    
    Returns:
        str: Path to the transformed file
    """
    print(f"Processing file: {input_file}")
    
    try:
        # Load the monthly log
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records")
        
        # Verify required columns exist
        required_columns = ['Date', 'Description', 'Out', 'In']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Create a new DataFrame with the required structure
        transformed_df = pd.DataFrame()
        
        # Map the columns
        transformed_df['Date'] = df['Date']
        transformed_df['Description'] = df['Description']
        
        # Handle Amount and Amount (Negated)
        transformed_df['Amount (Negated)'] = df['Out']
        transformed_df['Amount'] = df['In']
        
        # Clean up Amount fields - replace NaN with 0
        transformed_df['Amount (Negated)'] = pd.to_numeric(transformed_df['Amount (Negated)'], errors='coerce').fillna(0)
        transformed_df['Amount'] = pd.to_numeric(transformed_df['Amount'], errors='coerce').fillna(0)
    
        # Add transaction_source column
        transformed_df['transaction_source'] = 'household'
        
        # Determine output path
        if output_dir is None:
            output_dir = DATA_DIR
        
        # Create the output filename based on the input file pattern
        basename = os.path.basename(input_file)
        
        # Extract date part from the House Kitty format
        match = re.search(r"House Kitty Transactions - Cash (\d{4}-\d{2})\.csv", basename)
        if match:
            date_part = match.group(1)
            # Use exactly the requested filename format
            output_file = os.path.join(output_dir, f"House Kitty Transactions - Cash - Corrected {date_part}.csv")
        else:
            # Fallback
            timestamp = datetime.now().strftime("%Y-%m")
            output_file = os.path.join(output_dir, f"House Kitty Transactions - Cash - Corrected {timestamp}.csv")
        
        # Save the transformed DataFrame
        transformed_df.to_csv(output_file, index=False)
        print(f"Transformed data saved to: {output_file}")
        print(f"Transformation complete! {len(transformed_df)} records processed")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Transformation failed. Please check the input file format.")
        return None

def main():
    """Main function to handle file selection and transformation"""
    parser = argparse.ArgumentParser(description='Transform monthly logs to ML pipeline format')
    parser.add_argument('--data-dir', default=DATA_DIR, help='Directory containing transaction files')
    parser.add_argument('--output-dir', help='Directory to save the output file')
    
    args = parser.parse_args()
    
    # Use the data directory from the project structure
    data_dir = args.data_dir if os.path.exists(args.data_dir) else DATA_DIR
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        return
    
    if args.output_dir and not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)
    
    print(f"Searching for House Kitty Transaction files in: {data_dir}")
    
    # List available files
    files = list_monthly_files(data_dir)
    if not files:
        print(f"No House Kitty Transaction files found in {data_dir}")
        return
    
    print("\nAvailable files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nEnter the number of the file to process (or 'q' to quit): ")
            if choice.lower() == 'q':
                return
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(files):
                selected_file = files[file_index]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Process the selected file
    transform_monthly_log(selected_file, args.output_dir)

if __name__ == "__main__":
    main()