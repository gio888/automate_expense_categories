#!/usr/bin/env python3
"""
Unified Transaction Processing Workflow

Single command interface for the entire expense categorization pipeline.
Eliminates the need for multiple separate commands and provides guided workflow.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import time
from datetime import datetime

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our modules
from src.utils.file_detector import smart_file_selection, FileDetector
from src.utils.progress import TaskProgress, safe_execute, show_spinner
from src.transaction_types import TransactionSource
from src.batch_predict_ensemble import BatchPredictor
from src.merge_training_data import CorrectionValidatorExtended
from src.auto_model_ensemble import main as train_models
from src.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedWorkflow:
    """Unified workflow orchestrator for transaction processing"""
    
    def __init__(self, custom_directory: Optional[str] = None):
        self.config = get_config()
        self.data_dir = Path(self.config.get_paths_config()['data_dir'])
        self.file_detector = FileDetector()
        self.custom_directory = custom_directory
        
    def run_interactive_workflow(self) -> bool:
        """
        Run the complete interactive workflow
        
        Returns:
            True if workflow completed successfully, False otherwise
        """
        print("🎯 Welcome to the Expense Categorization Pipeline!")
        print("This will guide you through the complete process of categorizing your transactions.")
        print()
        
        # Initialize progress tracking
        progress = TaskProgress("Transaction Processing Workflow", 4)
        
        # Step 1: File selection and validation
        progress.start_step("File Selection & Validation")
        
        selected_file = self._select_and_validate_file()
        if not selected_file:
            print("❌ No file selected. Workflow cancelled.")
            return False
            
        # Detect transaction source
        validation_result = self.file_detector.detect_and_validate(selected_file)
        transaction_source = validation_result.transaction_source
        
        print(f"🎯 Detected transaction source: {transaction_source.value.replace('_', ' ').title()}")
        
        # Step 2: Process predictions
        progress.start_step("Generate ML Predictions")
        
        success, output_file = safe_execute(
            self._run_predictions, 
            selected_file, 
            transaction_source,
            task_name="Running ensemble ML models",
            show_progress=False
        )
        
        if not success:
            print(f"❌ Prediction failed: {output_file}")
            return False
            
        # Step 3: Manual corrections
        progress.start_step("Review & Correct Predictions")
        
        success, corrected_file = safe_execute(
            self._handle_corrections,
            output_file,
            task_name="Processing manual corrections",
            show_progress=False
        )
        
        if not success or not corrected_file:
            print("❌ Correction step cancelled. Workflow incomplete.")
            return False
            
        # Step 4: Update training data and retrain
        progress.start_step("Update Models & Retrain")
        
        success, _ = safe_execute(
            self._update_training_and_retrain,
            corrected_file,
            transaction_source,
            task_name="Updating training data and retraining models",
            show_progress=False
        )
        
        if not success:
            print("❌ Model update failed. Manual corrections saved but models not updated.")
            return False
            
        progress.complete_task()
        
        print(f"\n🎉 Success! Your transactions have been categorized and models updated.")
        print(f"📁 Results saved in: {Path(output_file).parent}")
        
        return True
    
    def _select_and_validate_file(self) -> Optional[str]:
        """Select and validate input file"""
        print("📂 Scanning for transaction files...")
        
        # Use custom directory if specified
        directories = None
        if self.custom_directory:
            directories = [self.custom_directory]
            print(f"🎯 Using custom directory: {self.custom_directory}")
        
        selected_file = smart_file_selection(directories)
        
        if selected_file:
            # Validate selected file
            validation_result = self.file_detector.detect_and_validate(selected_file)
            
            if validation_result.is_valid:
                print(f"✅ File validation passed")
                return selected_file
            else:
                print(f"⚠️  File has validation issues:")
                for issue in validation_result.issues:
                    print(f"   • {issue}")
                    
                if validation_result.suggestions:
                    print(f"\n💡 Suggestions:")
                    for suggestion in validation_result.suggestions:
                        print(f"   • {suggestion}")
                
                # Ask if user wants to continue anyway
                proceed = input(f"\nContinue with this file anyway? (y/N): ").strip().lower()
                if proceed in ['y', 'yes']:
                    return selected_file
                else:
                    return None
        
        return None
    
    def _run_predictions(self, input_file: str, transaction_source: TransactionSource) -> str:
        """Run prediction pipeline with progress tracking"""
        print("🤖 Loading ML models...")
        
        # Initialize predictor
        predictor = BatchPredictor()
        
        # Load data
        print("📊 Loading transaction data...")
        import pandas as pd
        df = pd.read_csv(input_file)
        
        # Add transaction source column if not present
        if 'transaction_source' not in df.columns:
            df['transaction_source'] = transaction_source.value
        
        print(f"⚡ Running ensemble predictions on {len(df)} transactions...")
        show_spinner(2.0, "Processing with ML models")
        
        # Run predictions
        predictions, probabilities = predictor.predict_batch(df)
        
        # Generate output filename
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{input_path.stem}_v1_{timestamp}.csv"
        output_path = input_path.parent / output_filename
        
        # Add predictions to dataframe
        df['predicted_category'] = predictions
        df['confidence'] = probabilities
        
        print("💾 Saving results...")
        df.to_csv(output_path, index=False)
        
        # Show prediction summary
        self._show_prediction_summary(df)
        
        return str(output_path)
    
    def _show_prediction_summary(self, df):
        """Show summary of prediction results"""
        total_predictions = len(df)
        high_confidence = (df['confidence'] > 0.7).sum()
        medium_confidence = ((df['confidence'] >= 0.5) & (df['confidence'] <= 0.7)).sum()
        low_confidence = (df['confidence'] < 0.5).sum()
        
        print(f"\n📊 Prediction Summary:")
        print(f"   Total transactions: {total_predictions}")
        print(f"   🟢 High confidence (>70%): {high_confidence} ({high_confidence/total_predictions:.1%})")
        print(f"   🟡 Medium confidence (50-70%): {medium_confidence} ({medium_confidence/total_predictions:.1%})")
        print(f"   🔴 Low confidence (<50%): {low_confidence} ({low_confidence/total_predictions:.1%})")
        
        if low_confidence > 0:
            print(f"\n💡 Tip: Focus on reviewing the {low_confidence} low-confidence predictions")
        
        # Show category distribution
        print(f"\n📈 Predicted Categories:")
        category_counts = df['predicted_category'].value_counts().head(5)
        for category, count in category_counts.items():
            print(f"   • {category}: {count} transactions")
        if len(df['predicted_category'].unique()) > 5:
            print(f"   • ... and {len(df['predicted_category'].unique()) - 5} other categories")
    
    def _handle_corrections(self, predictions_file: str) -> str:
        """Handle manual correction workflow"""
        print(f"\n📝 Time to review and correct predictions!")
        print(f"📁 File to review: {Path(predictions_file).name}")
        print()
        print("👀 Review Process:")
        print("   1. Focus on transactions with low confidence (<70%)")
        print("   2. Check if predicted categories make sense")
        print("   3. Edit the 'predicted_category' column as needed")
        print("   4. Save the file when done")
        print()
        
        # Offer options for correction
        print("🛠️  Correction Options:")
        print("   1. Open file in spreadsheet application (Excel, LibreOffice, etc.)")
        print("   2. Quick review mode (command-line)")
        print("   3. Skip corrections (use predictions as-is)")
        
        while True:
            choice = input("\nHow would you like to review predictions? (1-3): ").strip()
            
            if choice == '1':
                return self._open_in_external_editor(predictions_file)
            elif choice == '2':
                return self._quick_review_mode(predictions_file)
            elif choice == '3':
                confirm = input("⚠️  Skip corrections? Models won't improve without feedback (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return predictions_file
            else:
                print("❌ Please enter 1, 2, or 3")
    
    def _open_in_external_editor(self, predictions_file: str) -> Optional[str]:
        """Open file in external editor for corrections"""
        import subprocess
        import platform
        
        file_path = Path(predictions_file)
        
        print(f"📂 Opening {file_path.name} in default spreadsheet application...")
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.call(["open", str(file_path)])
            elif platform.system() == "Windows":  # Windows
                subprocess.call(["start", str(file_path)], shell=True)
            else:  # Linux
                subprocess.call(["xdg-open", str(file_path)])
            
            print("✏️  Make your corrections in the spreadsheet:")
            print("   • Edit the 'predicted_category' column")
            print("   • Focus on low confidence predictions")
            print("   • Save the file when finished")
            
            input("\n⏳ Press Enter when you've finished making corrections...")
            
            # Verify file was modified
            if self._verify_corrections_made(predictions_file):
                return predictions_file
            else:
                print("⚠️  No modifications detected. Using original predictions.")
                return predictions_file
                
        except Exception as e:
            print(f"❌ Could not open file: {e}")
            print("💡 You can manually open this file: {file_path}")
            
            input("Press Enter when you've finished manual corrections...")
            return predictions_file
    
    def _quick_review_mode(self, predictions_file: str) -> Optional[str]:
        """Quick command-line review mode"""
        import pandas as pd
        
        try:
            df = pd.read_csv(predictions_file)
            
            # Show only low confidence predictions
            low_confidence = df[df['confidence'] < 0.7].copy()
            
            if len(low_confidence) == 0:
                print("✅ All predictions have high confidence! No manual review needed.")
                return predictions_file
            
            print(f"\n🔍 Reviewing {len(low_confidence)} low-confidence predictions:")
            print("   Enter new category or press Enter to keep current prediction")
            print("   Type 'q' to finish reviewing")
            print()
            
            corrections_made = False
            
            for idx, row in low_confidence.iterrows():
                print(f"Transaction: {row.get('description', 'N/A')}")
                print(f"Amount: ${row.get('amount', 'N/A')}")
                print(f"Current prediction: {row['predicted_category']} (confidence: {row['confidence']:.1%})")
                
                new_category = input("New category (or Enter to keep): ").strip()
                
                if new_category.lower() == 'q':
                    break
                elif new_category:
                    df.loc[idx, 'predicted_category'] = new_category
                    df.loc[idx, 'confidence'] = 1.0  # User correction gets full confidence
                    corrections_made = True
                    print("✅ Updated")
                
                print("-" * 50)
            
            if corrections_made:
                # Save corrections
                corrected_file = predictions_file.replace('.csv', '_corrected.csv')
                df.to_csv(corrected_file, index=False)
                print(f"💾 Corrections saved to: {Path(corrected_file).name}")
                return corrected_file
            else:
                print("ℹ️  No corrections made")
                return predictions_file
                
        except Exception as e:
            print(f"❌ Error in quick review: {e}")
            return predictions_file
    
    def _verify_corrections_made(self, file_path: str) -> bool:
        """Check if file was modified recently"""
        try:
            file_stat = Path(file_path).stat()
            # Check if modified in last 5 minutes
            return time.time() - file_stat.st_mtime < 300
        except:
            return False
    
    def _update_training_and_retrain(self, corrected_file: str, transaction_source: TransactionSource) -> None:
        """Update training data and retrain models"""
        print("📚 Integrating corrections into training data...")
        
        # Use existing merge_training_data functionality
        validator = CorrectionValidatorExtended()
        
        # Process the corrections
        success = validator.process_corrections_file(
            corrected_file, 
            transaction_source.value,
            auto_confirm=True  # Skip interactive prompts
        )
        
        if not success:
            raise Exception("Failed to integrate corrections into training data")
        
        print("✅ Training data updated")
        
        # Retrain models
        print("🎓 Retraining models with new data...")
        show_spinner(3.0, "Training ensemble models")
        
        # Call model training
        original_argv = sys.argv
        sys.argv = ['auto_model_ensemble.py', '--source', transaction_source.value]
        
        try:
            train_models()
            print("✅ Models retrained successfully")
        except Exception as e:
            print(f"⚠️  Model retraining encountered issues: {e}")
            print("💡 Your corrections are saved - models can be retrained later")
            raise Exception(f"Model retraining failed: {e}")
        finally:
            sys.argv = original_argv

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Transaction Processing Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_transactions.py                          # Interactive workflow
  python process_transactions.py --quick                  # Skip detailed prompts
  python process_transactions.py --file transactions.csv  # Process specific file

This tool replaces the need to run multiple separate commands for:
1. File transformation (household only)
2. Batch prediction 
3. Manual correction
4. Training data merge
5. Model retraining
        """
    )
    
    parser.add_argument("--file", help="Process specific CSV file")
    parser.add_argument("--directory", help="Search for files in specific directory")
    parser.add_argument("--interactive", action="store_true", default=True, 
                       help="Run interactive workflow (default)")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick mode with minimal prompts")
    
    args = parser.parse_args()
    
    workflow = UnifiedWorkflow(args.directory)
    
    if args.file:
        print(f"🎯 Processing file: {args.file}")
        # Direct file processing mode
        # TODO: Implement direct file mode
        print("❌ Direct file processing not yet implemented. Use interactive mode.")
        return 1
    else:
        # Interactive workflow
        success = workflow.run_interactive_workflow()
        return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Workflow cancelled by user. Goodbye!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error occurred: {str(e)}")
        print("💡 Please report this issue if it persists")
        sys.exit(1)