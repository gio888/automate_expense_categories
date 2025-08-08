"""
FastAPI web application for the expense categorization pipeline.
Provides REST API endpoints for file upload, processing, and correction.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import asyncio
from io import StringIO

# FastAPI and dependencies
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our modules
from src.utils.file_detector import FileDetector, FileValidationResult
from src.transaction_types import TransactionSource
from src.batch_predict_ensemble import BatchPredictor
from src.merge_training_data import CorrectionValidatorExtended
from src.auto_model_ensemble import main as train_models
from src.config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Expense Categorization Pipeline",
    description="Web API for automated expense categorization using ML",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking processing jobs
processing_jobs = {}
job_counter = 0

# Load valid categories for corrections
def load_valid_categories():
    """Load valid categories from file for correction dropdowns"""
    try:
        categories_file = PROJECT_ROOT / "data" / "valid_categories.txt"
        if categories_file.exists():
            with open(categories_file, 'r') as f:
                categories = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(categories)} valid categories")
            return categories
        else:
            logger.warning(f"Categories file not found: {categories_file}")
            return []
    except Exception as e:
        logger.error(f"Error loading categories: {e}")
        return []

# Cache categories on startup
VALID_CATEGORIES = load_valid_categories()

# Pydantic models for API requests/responses
class FileUploadResponse(BaseModel):
    success: bool
    filename: str
    file_id: str
    validation_result: Optional[Dict[str, Any]]
    message: str

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_step: str
    message: str
    result_file: Optional[str] = None
    error: Optional[str] = None

class CorrectionRequest(BaseModel):
    file_id: str
    corrections: List[Dict[str, Any]]  # List of row corrections

class CorrectionResponse(BaseModel):
    success: bool
    message: str
    updated_rows: int

# Initialize components
file_detector = FileDetector()
config = get_config()
upload_dir = Path(config.get_paths_config()['data_dir']) / "uploads"
results_dir = Path(config.get_paths_config()['data_dir']) / "results"

# Create necessary directories
upload_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Expense Categorization Pipeline</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; 
                        border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .upload-zone { border: 2px dashed #ddd; border-radius: 8px; padding: 40px; 
                          text-align: center; margin: 20px 0; transition: border-color 0.3s; }
            .upload-zone.dragover { border-color: #007bff; background: #f8f9ff; }
            .upload-zone input { display: none; }
            .upload-button { background: #007bff; color: white; padding: 12px 24px; 
                           border-radius: 6px; border: none; cursor: pointer; font-size: 16px; }
            .upload-button:hover { background: #0056b3; }
            .file-info { background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 10px 0; }
            .progress-bar { width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }
            .progress-fill { height: 100%; background: #28a745; transition: width 0.3s; }
            .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .predictions-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .predictions-table th, .predictions-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .predictions-table th { background: #f8f9fa; }
            .confidence-high { background-color: #d4edda; }
            .confidence-medium { background-color: #fff3cd; }
            .confidence-low { background-color: #f8d7da; }
            .searchable-dropdown { position: relative; display: inline-block; width: 100%; }
            .dropdown-list { position: absolute; top: 100%; left: 0; right: 0; background: white; 
                           border: 1px solid #ddd; max-height: 200px; overflow-y: auto; z-index: 1000; display: none; }
            .dropdown-item { padding: 8px; cursor: pointer; border-bottom: 1px solid #f0f0f0; }
            .dropdown-item:hover { background-color: #f5f5f5; }
            .dropdown-item:last-child { border-bottom: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Expense Categorization Pipeline</h1>
                <p>Upload your transaction CSV files for automatic categorization using machine learning</p>
            </div>
            
            <div class="upload-zone" id="uploadZone">
                <input type="file" id="fileInput" accept=".csv" />
                <p>📁 Drag and drop your CSV file here</p>
                <button class="upload-button" onclick="document.getElementById('fileInput').click()">
                    Choose File
                </button>
                <p style="font-size: 14px; color: #666; margin-top: 15px;">
                    Supports transaction files from banks, credit cards, or household expense tracking
                </p>
            </div>
            
            <div id="fileInfo" style="display: none;"></div>
            <div id="progress" style="display: none;"></div>
            <div id="results" style="display: none;"></div>
        </div>
        
        <script src="/static/app.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and validate a CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Generate unique file ID
        global job_counter
        job_counter += 1
        file_id = f"file_{job_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save uploaded file
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Validate file
        validation_result = file_detector.detect_and_validate(str(file_path))
        
        return FileUploadResponse(
            success=True,
            filename=file.filename,
            file_id=file_id,
            validation_result={
                "is_valid": validation_result.is_valid,
                "transaction_source": validation_result.transaction_source.value if validation_result.transaction_source else None,
                "confidence": validation_result.confidence,
                "row_count": validation_result.row_count,
                "file_size": validation_result.file_size,
                "issues": validation_result.issues,
                "suggestions": validation_result.suggestions,
                "detected_columns": validation_result.detected_columns
            },
            message="File uploaded and validated successfully"
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process/{file_id}")
async def start_processing(file_id: str, background_tasks: BackgroundTasks):
    """Start processing a file in the background"""
    try:
        # Find the uploaded file
        file_path = None
        for f in upload_dir.glob(f"{file_id}_*"):
            file_path = f
            break
            
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Generate job ID
        job_id = f"job_{file_id}_{datetime.now().strftime('%H%M%S')}"
        
        # Initialize job status
        processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="queued",
            progress=0.0,
            current_step="Initializing",
            message="Processing job queued"
        )
        
        # Start background processing
        background_tasks.add_task(process_file_background, job_id, str(file_path))
        
        return {"job_id": job_id, "message": "Processing started"}
        
    except Exception as e:
        logger.error(f"Processing start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get processing results as JSON with enhanced correction support"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job.status != "completed" or not job.result_file:
        raise HTTPException(status_code=404, detail="Results not ready")
    
    try:
        # Read the results file
        df = pd.read_csv(job.result_file)
        
        # Convert to JSON-friendly format with correction support
        results = []
        for _, row in df.iterrows():
            result_row = {
                "id": len(results),
                "Date": row.get('Date', ''),
                "Description": row.get('Description', ''),
                "Amount (Negated)": row.get('Amount (Negated)', 0),
                "Amount": row.get('Amount', 0),
                "predicted_category": row.get('predicted_category', ''),
                "corrected_category": row.get('predicted_category', ''),  # Pre-populate with prediction
                "confidence": row.get('confidence', 0),
                "transaction_source": row.get('transaction_source', '')
            }
            results.append(result_row)
        
        return {
            "job_id": job_id,
            "total_rows": len(results),
            "high_confidence": len([r for r in results if r['confidence'] > 0.7]),
            "medium_confidence": len([r for r in results if 0.5 <= r['confidence'] <= 0.7]),
            "low_confidence": len([r for r in results if r['confidence'] < 0.5]),
            "predictions": results,
            "valid_categories": VALID_CATEGORIES  # Include categories for dropdown
        }
        
    except Exception as e:
        logger.error(f"Results retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@app.post("/correct/{job_id}", response_model=CorrectionResponse)
async def submit_corrections(job_id: str, corrections: CorrectionRequest):
    """Submit manual corrections for predictions"""
    try:
        if job_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = processing_jobs[job_id]
        
        if not job.result_file:
            raise HTTPException(status_code=400, detail="No results file to correct")
        
        # Load the results file
        df = pd.read_csv(job.result_file)
        
        # Apply corrections
        updated_rows = 0
        for correction in corrections.corrections:
            row_id = correction.get('id')
            new_category = correction.get('predicted_category')
            
            if row_id is not None and row_id < len(df) and new_category:
                df.loc[row_id, 'predicted_category'] = new_category
                df.loc[row_id, 'confidence'] = 1.0  # User corrections get full confidence
                updated_rows += 1
        
        # Save corrected file
        corrected_path = results_dir / f"corrected_{job_id}.csv"
        df.to_csv(corrected_path, index=False)
        
        # Update job with corrected file path
        job.result_file = str(corrected_path)
        processing_jobs[job_id] = job
        
        return CorrectionResponse(
            success=True,
            message=f"Applied {updated_rows} corrections successfully",
            updated_rows=updated_rows
        )
        
    except Exception as e:
        logger.error(f"Corrections failed: {e}")
        raise HTTPException(status_code=500, detail=f"Corrections failed: {str(e)}")

@app.post("/retrain/{job_id}")
async def retrain_models(job_id: str, background_tasks: BackgroundTasks):
    """Retrain models with corrected data"""
    try:
        if job_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = processing_jobs[job_id]
        
        if not job.result_file:
            raise HTTPException(status_code=400, detail="No corrected data available")
        
        # Start background retraining
        background_tasks.add_task(retrain_models_background, job_id)
        
        return {"message": "Model retraining started", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/download/{job_id}")
async def download_results(job_id: str, format: str = "predictions"):
    """Download results as CSV file in specified format"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if not job.result_file or not Path(job.result_file).exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    if format == "accounting":
        # Generate accounting system format: Date,Description,Amount (Negated),Amount,Classification
        try:
            df = pd.read_csv(job.result_file)
            
            # Create accounting format with corrected categories
            accounting_df = pd.DataFrame({
                'Date': df['Date'],
                'Description': df['Description'], 
                'Amount (Negated)': df['Amount (Negated)'],
                'Amount': df['Amount'],
                'Classification': df['predicted_category']  # Will be updated by corrections
            })
            
            # Save temporary accounting format file
            accounting_file = results_dir / f"accounting_{job_id}.csv"
            accounting_df.to_csv(accounting_file, index=False)
            
            filename = f"accounting_import_{job_id}.csv"
            return FileResponse(
                accounting_file,
                media_type='text/csv',
                filename=filename
            )
        except Exception as e:
            logger.error(f"Error creating accounting format: {e}")
            raise HTTPException(status_code=500, detail="Failed to create accounting format")
    
    else:
        # Default: full predictions format
        filename = f"expense_predictions_{job_id}.csv"
        return FileResponse(
            job.result_file,
            media_type='text/csv',
            filename=filename
        )

# Background task functions
async def process_file_background(job_id: str, file_path: str):
    """Background task for processing files"""
    try:
        job = processing_jobs[job_id]
        
        # Step 1: Validate and detect source
        job.status = "processing"
        job.progress = 0.1
        job.current_step = "Validating file"
        job.message = "Analyzing file structure and content"
        processing_jobs[job_id] = job
        
        validation_result = file_detector.detect_and_validate(file_path)
        
        if not validation_result.is_valid:
            job.status = "failed"
            job.error = f"File validation failed: {'; '.join(validation_result.issues)}"
            processing_jobs[job_id] = job
            return
        
        transaction_source = validation_result.transaction_source
        
        # Step 2: Load models and run predictions
        job.progress = 0.3
        job.current_step = "Loading ML models"
        job.message = "Initializing ensemble machine learning models"
        processing_jobs[job_id] = job
        
        predictor = BatchPredictor()
        
        job.progress = 0.5
        job.current_step = "Running predictions"
        job.message = f"Processing {validation_result.row_count} transactions"
        processing_jobs[job_id] = job
        
        # Load and process data
        df = pd.read_csv(file_path)
        
        # Add transaction source if missing
        if 'transaction_source' not in df.columns:
            df['transaction_source'] = transaction_source.value
        
        # Run predictions
        predictions, probabilities = predictor.predict_batch(df)
        
        # Add predictions to dataframe
        df['predicted_category'] = predictions
        df['confidence'] = probabilities
        
        # Step 3: Save results
        job.progress = 0.8
        job.current_step = "Saving results"
        job.message = "Finalizing predictions and saving results"
        processing_jobs[job_id] = job
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{job_id}_{timestamp}.csv"
        output_path = results_dir / output_filename
        
        df.to_csv(output_path, index=False)
        
        # Complete the job
        job.status = "completed"
        job.progress = 1.0
        job.current_step = "Complete"
        job.message = f"Successfully processed {len(df)} transactions"
        job.result_file = str(output_path)
        processing_jobs[job_id] = job
        
        logger.info(f"Processing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        job = processing_jobs.get(job_id)
        if job:
            job.status = "failed"
            job.error = str(e)
            processing_jobs[job_id] = job

async def retrain_models_background(job_id: str):
    """Background task for retraining models"""
    try:
        job = processing_jobs[job_id]
        corrected_file = job.result_file
        
        if not corrected_file:
            return
        
        # Determine transaction source from corrected data
        df = pd.read_csv(corrected_file)
        transaction_source = df['transaction_source'].iloc[0] if 'transaction_source' in df.columns else 'credit_card'
        
        # Update training data
        validator = CorrectionValidatorExtended()
        success = validator.process_corrections_file(
            corrected_file,
            transaction_source,
            auto_confirm=True
        )
        
        if not success:
            logger.error(f"Failed to integrate corrections for job {job_id}")
            return
        
        # Retrain models
        original_argv = sys.argv
        sys.argv = ['auto_model_ensemble.py', '--source', transaction_source]
        
        try:
            train_models()
            logger.info(f"Models retrained successfully for job {job_id}")
        except Exception as e:
            logger.error(f"Model retraining failed for job {job_id}: {e}")
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"Background retraining failed for job {job_id}: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in processing_jobs.values() if j.status == "processing"]),
        "total_jobs": len(processing_jobs)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create static directory for frontend assets
    static_dir = PROJECT_ROOT / "src" / "web" / "static"
    static_dir.mkdir(exist_ok=True)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    print("🚀 Starting Expense Categorization Web Server")
    print("📱 Open http://localhost:8000 in your browser")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)