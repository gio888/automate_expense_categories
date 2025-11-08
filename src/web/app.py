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

# Import filename utilities
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.filename_utils import (
    extract_source_and_period, create_predictions_filename, 
    create_accounting_filename, create_accounting_from_predictions_filename,
    create_corrected_filename, generate_filename
)

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
from src.utils.excel_processor import ExcelProcessor, is_excel_file
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

# Add CORS middleware - restricted to localhost for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# Global state for tracking processing jobs and uploaded files
processing_jobs = {}
uploaded_files = {}  # Track uploaded file metadata for naming
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

def get_column_value_flexible(row, primary_key, fallback_keys=None, default=''):
    """
    Safely get column value with flexible key matching for Excel/CSV compatibility.
    
    Args:
        row: DataFrame row or dictionary
        primary_key: Primary column name to look for
        fallback_keys: List of alternative column names to try
        default: Default value if no keys are found
        
    Returns:
        Column value or default if not found
    """
    if fallback_keys is None:
        fallback_keys = []
    
    # Try primary key first
    if hasattr(row, 'get'):
        value = row.get(primary_key)
        if value is not None and str(value).strip() != '' and str(value).lower() != 'nan':
            return value
    elif hasattr(row, primary_key) and getattr(row, primary_key) is not None:
        value = getattr(row, primary_key)
        if str(value).strip() != '' and str(value).lower() != 'nan':
            return value
    
    # Try fallback keys
    for fallback_key in fallback_keys:
        if hasattr(row, 'get'):
            value = row.get(fallback_key)
            if value is not None and str(value).strip() != '' and str(value).lower() != 'nan':
                return value
        elif hasattr(row, fallback_key) and getattr(row, fallback_key) is not None:
            value = getattr(row, fallback_key)
            if str(value).strip() != '' and str(value).lower() != 'nan':
                return value
    
    return default

# Mount static files - needs to be done early for start_web_server.py
from fastapi.staticfiles import StaticFiles
static_dir = PROJECT_ROOT / "src" / "web" / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
            /* Modern Responsive CSS */
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 1rem; background: #f5f5f5; 
                line-height: 1.5;
            }

            /* Dynamic container that expands for correction interface */
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 8px; 
                padding: 2rem; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transition: max-width 0.3s ease;
            }

            /* Expand container for correction interface */
            .container.expanded {
                max-width: min(95vw, 1400px);
            }

            .header { text-align: center; margin-bottom: 2rem; }

            /* Responsive upload zone */
            .upload-zone { 
                border: 2px dashed #ddd; 
                border-radius: 12px; 
                padding: 2.5rem 1rem; 
                text-align: center; 
                margin: 1.5rem 0; 
                transition: all 0.3s ease;
            }
            .upload-zone.dragover { border-color: #007bff; background: #f8f9ff; }
            .upload-zone input { display: none; }

            .upload-button { 
                background: #007bff; 
                color: white; 
                padding: 12px 24px; 
                border-radius: 8px; 
                border: none; 
                cursor: pointer; 
                font-size: 16px; 
                font-weight: 500;
                transition: all 0.2s ease;
            }
            .upload-button:hover { background: #0056b3; transform: translateY(-1px); }
            .upload-button:focus { 
                outline: 3px solid rgba(0, 123, 255, 0.3); 
                outline-offset: 2px; 
            }

            .file-info, .status { 
                padding: 1rem 1.25rem; 
                border-radius: 8px; 
                margin: 1rem 0; 
                border: 1px solid transparent;
            }
            .file-info { background: #f8f9fa; }
            .status.success { background: #d1ecf1; color: #0c5460; border-color: #bee5eb; }
            .status.error { background: #f8d7da; color: #721c24; border-color: #f5c6cb; }
            .status.warning { background: #fff3cd; color: #856404; border-color: #ffeaa7; }

            .progress-bar { 
                width: 100%; 
                height: 24px; 
                background: #e9ecef; 
                border-radius: 12px; 
                overflow: hidden; 
                margin: 0.5rem 0;
            }
            .progress-fill { 
                height: 100%; 
                background: linear-gradient(90deg, #28a745, #20c997); 
                transition: width 0.4s ease; 
                border-radius: 12px;
            }

            /* Responsive Table Design */
            .responsive-table-container {
                margin: 2rem 0;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                overflow: hidden;
                border: 1px solid #e2e8f0;
            }

            .responsive-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
            }

            .responsive-table th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                padding: 1rem 0.75rem;
                text-align: left;
                position: sticky;
                top: 0;
                z-index: 10;
                border-bottom: 2px solid #4a5568;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            }

            .responsive-table td {
                padding: 0.875rem 0.75rem;
                border-bottom: 1px solid #e2e8f0;
                vertical-align: top;
                font-size: 0.875rem;
            }

            .responsive-table tbody tr:hover {
                background-color: #f7fafc;
            }

            /* Column sizing for optimal responsive behavior */
            .col-date { width: 8%; min-width: 90px; }
            .col-description { width: 25%; min-width: 200px; white-space: normal; }
            .col-debit { width: 10%; min-width: 80px; text-align: right; }
            .col-credit { width: 10%; min-width: 80px; text-align: right; }
            .col-prediction { width: 22%; min-width: 180px; white-space: normal; }
            .col-confidence { width: 8%; min-width: 70px; text-align: center; }
            .col-correction { width: 17%; min-width: 200px; }

            /* Confidence indicators */
            .confidence-high { background-color: rgba(72, 187, 120, 0.1) !important; }
            .confidence-medium { background-color: rgba(251, 188, 5, 0.1) !important; }
            .confidence-low { background-color: rgba(245, 101, 101, 0.1) !important; }

            .confidence-badge {
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
                text-align: center;
                min-width: 45px;
            }

            .confidence-high .confidence-badge {
                background-color: #48bb78;
                color: white;
            }

            .confidence-medium .confidence-badge {
                background-color: #ed8936;
                color: white;
            }

            .confidence-low .confidence-badge {
                background-color: #f56565;
                color: white;
            }

            /* Enhanced dropdown styling with modern UX */
            .searchable-dropdown {
                position: relative;
                width: 100%;
            }

            .searchable-dropdown input {
                width: 100%;
                padding: 0.75rem 2.5rem 0.75rem 0.75rem;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 0.875rem;
                transition: all 0.2s ease;
                background: white url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill='none' stroke='%23343a40' stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='m2 5 6 6 6-6'/%3e%3c/svg%3e") no-repeat right 0.75rem center;
                background-size: 16px 12px;
                box-sizing: border-box;
                cursor: pointer;
            }

            .searchable-dropdown input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill='none' stroke='%23667eea' stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='m2 5 6 6 6-6'/%3e%3c/svg%3e");
            }

            /* Search icon when typing */
            .searchable-dropdown.searching input {
                background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill='%23667eea' d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z'/%3e%3c/svg%3e");
            }

            /* Selection confirmation visual feedback */
            .searchable-dropdown input.selected {
                border-color: #48bb78;
                background-color: rgba(72, 187, 120, 0.05);
            }

            .dropdown-list {
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                max-height: 200px;
                overflow-y: auto;
                z-index: 1000;
                display: none;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                margin-top: 4px;
            }

            .dropdown-item {
                padding: 0.875rem 0.75rem;
                cursor: pointer;
                border-bottom: 1px solid #f7fafc;
                font-size: 0.875rem;
                transition: all 0.15s ease;
                position: relative;
            }

            .dropdown-item:hover,
            .dropdown-item.highlighted {
                background-color: #667eea;
                color: white;
                transform: translateX(4px);
            }

            .dropdown-item.selected {
                background-color: #48bb78;
                color: white;
                font-weight: 600;
            }

            .dropdown-item:last-child {
                border-bottom: none;
            }

            /* Loading and empty states */
            .dropdown-loading,
            .dropdown-empty {
                padding: 1rem;
                text-align: center;
                color: #718096;
                font-style: italic;
            }

            .dropdown-loading::before {
                content: "⏳ ";
            }

            .dropdown-empty::before {
                content: "🔍 ";
            }

            /* Sortable columns styling */
            .responsive-table th.sortable {
                cursor: pointer;
                user-select: none;
                position: relative;
                padding-right: 2rem;
                transition: background-color 0.2s ease;
            }

            .responsive-table th.sortable:hover {
                background: linear-gradient(135deg, #5a6fd8 0%, #6b4d94 100%);
            }

            .responsive-table th.sortable::after {
                content: '';
                position: absolute;
                right: 0.5rem;
                top: 50%;
                transform: translateY(-50%);
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 5px solid rgba(255, 255, 255, 0.5);
                transition: all 0.2s ease;
            }

            .responsive-table th.sortable.sort-asc::after {
                border-bottom: 5px solid white;
                border-top: none;
            }

            .responsive-table th.sortable.sort-desc::after {
                border-top: 5px solid white;
                border-bottom: none;
            }

            /* Sort indicator for accessibility */
            .sort-indicator {
                position: absolute;
                left: -9999px;
            }

            /* Legacy table support for older tables */
            .predictions-table { 
                width: 100%; border-collapse: collapse; margin-top: 20px;
            }
            .predictions-table th, .predictions-table td { 
                padding: 8px 12px; border: 1px solid #ddd; text-align: left; 
            }
            .predictions-table th { 
                background: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 5;
            }
            .table-container { overflow: auto; border: 1px solid #ddd; border-radius: 4px; }

            /* Mobile Responsive Design */
            @media (max-width: 1200px) {
                .container.expanded {
                    max-width: 98vw;
                    padding: 1rem;
                }
                
                .responsive-table th,
                .responsive-table td {
                    padding: 0.5rem;
                    font-size: 0.8rem;
                }
            }

            @media (max-width: 768px) {
                body { padding: 0.5rem; }
                
                .container.expanded {
                    max-width: 100vw;
                    padding: 0.75rem;
                    border-radius: 0;
                }
                
                /* Stack table for mobile */
                .responsive-table-container {
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }
                
                .responsive-table {
                    min-width: 700px;
                }
                
                .responsive-table th,
                .responsive-table td {
                    padding: 0.5rem 0.25rem;
                    font-size: 0.75rem;
                }
                
                /* Mobile dropdown optimizations */
                .searchable-dropdown input {
                    padding: 1rem 3rem 1rem 1rem;
                    font-size: 1rem; /* Prevent zoom on iOS */
                    background-size: 20px 15px;
                    background-position: right 1rem center;
                }
                
                .dropdown-list {
                    max-height: 200px;
                    border-radius: 12px;
                }
                
                .dropdown-item {
                    padding: 1.125rem 1rem;
                    font-size: 1rem;
                    min-height: 48px; /* Touch target size */
                    display: flex;
                    align-items: center;
                }
                
                /* Mobile sort indicators */
                .responsive-table th.sortable::after {
                    right: 0.25rem;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-bottom: 4px solid rgba(255, 255, 255, 0.5);
                }
                
                .upload-zone {
                    padding: 1.5rem 1rem;
                }
            }

            /* Focus and keyboard navigation */
            *:focus {
                outline: 2px solid #667eea;
                outline-offset: 2px;
            }

            /* Source Detection and Selection Styles */
            .source-detection {
                margin: 1rem 0;
                padding: 1rem;
                background: linear-gradient(135deg, #f6f9fc 0%, #e9f2f9 100%);
                border-radius: 8px;
                border: 2px solid #d1e3f0;
            }

            .source-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
            }

            .source-high-confidence {
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                color: #155724;
                border: 2px solid #28a745;
            }

            .confidence-indicator {
                font-size: 0.875rem;
                font-weight: 500;
                opacity: 0.85;
            }

            .btn-link-subtle {
                background: none;
                border: none;
                color: #007bff;
                text-decoration: underline;
                cursor: pointer;
                font-size: 0.875rem;
                padding: 0.25rem 0.5rem;
                margin-left: 1rem;
                transition: color 0.2s ease;
            }

            .btn-link-subtle:hover {
                color: #0056b3;
            }

            .btn-link-subtle:focus {
                outline: 2px solid #667eea;
                outline-offset: 2px;
                border-radius: 4px;
            }

            /* Radio Card Design */
            .radio-group-modern {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                margin: 1rem 0;
            }

            .radio-card {
                position: relative;
                display: flex;
                align-items: center;
                padding: 1.25rem;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                background: white;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .radio-card:hover {
                border-color: #667eea;
                background: #f8f9ff;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
            }

            .radio-card:has(input:checked) {
                border-color: #667eea;
                background: linear-gradient(135deg, #f8f9ff 0%, #e9ecff 100%);
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
            }

            .radio-card input[type="radio"] {
                position: absolute;
                opacity: 0;
                width: 0;
                height: 0;
            }

            .radio-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            }

            .radio-title {
                font-size: 1rem;
                font-weight: 600;
                color: #2d3748;
            }

            .radio-description {
                font-size: 0.875rem;
                color: #718096;
                line-height: 1.4;
            }

            .radio-indicator {
                width: 24px;
                height: 24px;
                border: 2px solid #cbd5e0;
                border-radius: 50%;
                position: relative;
                transition: all 0.2s ease;
            }

            .radio-card:has(input:checked) .radio-indicator {
                border-color: #667eea;
                background: #667eea;
            }

            .radio-card:has(input:checked) .radio-indicator::after {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 8px;
                height: 8px;
                background: white;
                border-radius: 50%;
            }

            /* Help Accordion */
            .help-accordion {
                margin-top: 1rem;
                padding: 0.75rem;
                background: #f7fafc;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }

            .help-accordion summary {
                cursor: pointer;
                font-weight: 600;
                color: #4a5568;
                list-style: none;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .help-accordion summary::-webkit-details-marker {
                display: none;
            }

            .help-accordion summary:hover {
                color: #667eea;
            }

            .help-accordion[open] summary {
                margin-bottom: 0.75rem;
            }

            .help-content {
                padding-left: 1.75rem;
                color: #718096;
                font-size: 0.875rem;
                line-height: 1.6;
            }

            .help-content p {
                margin: 0.5rem 0;
            }

            /* Medium/Low Confidence States */
            .source-selection-medium,
            .source-selection-required {
                margin: 1rem 0;
                padding: 1.25rem;
                border-radius: 12px;
                border: 2px solid #ff9800;
            }

            .source-selection-medium {
                background: linear-gradient(135deg, #fff8f0 0%, #fff3e0 100%);
            }

            .source-selection-required {
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            }

            .source-heading {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.5rem;
                color: #2d3748;
            }

            .warning-icon {
                font-size: 1.25rem;
            }

            /* Mobile Responsive */
            @media (max-width: 768px) {
                .source-badge {
                    font-size: 0.9rem;
                    padding: 0.5rem 0.75rem;
                }

                .confidence-indicator {
                    font-size: 0.75rem;
                }

                .btn-link-subtle {
                    display: block;
                    margin-left: 0;
                    margin-top: 0.5rem;
                }

                .radio-card {
                    padding: 1rem;
                    min-height: 44px;
                }

                .radio-title {
                    font-size: 1rem;
                }

                .radio-indicator {
                    width: 28px;
                    height: 28px;
                }
            }

            /* Print styles */
            @media print {
                .upload-zone, .status { display: none; }
                .container { max-width: none; box-shadow: none; }
                .responsive-table { font-size: 10px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Expense Categorization Pipeline</h1>
                <p>Upload your transaction CSV or Excel files for automatic categorization using machine learning</p>
            </div>
            
            <div class="upload-zone" id="uploadZone">
                <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" />
                <p>📁 Drag and drop your CSV or Excel file here</p>
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
        
        <script src="/static/app.js?v=1754654547"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and validate a CSV or Excel file"""
    try:
        # Security: Define maximum file size (10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

        # Security: Sanitize filename - only keep the filename, remove any path components
        original_filename = Path(file.filename).name

        # Validate file type
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_extension = Path(original_filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

        # Generate unique file ID
        global job_counter
        job_counter += 1
        file_id = f"file_{job_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract source and period information for naming
        source, period = extract_source_and_period(original_filename)

        # Security: Use sanitized filename for file path
        file_path = upload_dir / f"{file_id}_{original_filename}"

        # Security: Validate that the resolved path is within the upload directory (prevent path traversal)
        # Python 3.8+ compatible - using relative_to() with exception handling
        try:
            file_path.resolve().relative_to(upload_dir.resolve())
        except ValueError:
            logger.warning(f"Path traversal attempt detected: {original_filename}")
            raise HTTPException(status_code=400, detail="Invalid file path")

        # Security: Check file size while saving to prevent DoS via disk exhaustion
        file_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # Read in chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    buffer.close()
                    file_path.unlink()  # Delete partial file
                    raise HTTPException(status_code=413, detail="File too large (maximum 10MB)")
                buffer.write(chunk)

        logger.info(f"File uploaded: {file_path} (source: {source}, period: {period}, size: {file_size} bytes)")

        # Store source and period for later use in filename generation
        uploaded_files[file_id] = {
            "original_filename": original_filename,
            "source": source,
            "period": period,
            "file_path": str(file_path)
        }

        # Validate file
        validation_result = file_detector.detect_and_validate(str(file_path))

        return FileUploadResponse(
            success=True,
            filename=original_filename,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")

@app.post("/process/{file_id}")
async def start_processing(file_id: str, background_tasks: BackgroundTasks, confirmed_source: Optional[str] = None):
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

        # Start background processing with user-confirmed source (if provided)
        background_tasks.add_task(process_file_background, job_id, str(file_path), confirmed_source)

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
        
        # Log column information for debugging
        logger.info(f"Results file columns for job {job_id}: {list(df.columns)}")
        
        # Convert to JSON-friendly format with correction support
        results = []
        for _, row in df.iterrows():
            result_row = {
                "id": len(results),
                "Date": get_column_value_flexible(row, 'Date', ['DATE', 'date'], ''),
                "Description": get_column_value_flexible(row, 'Description', ['DESCRIPTION', 'description'], ''),
                "Amount (Negated)": get_column_value_flexible(row, 'Amount (Negated)', ['DEBIT', 'debit'], 0),
                "Amount": get_column_value_flexible(row, 'Amount', ['CREDIT', 'credit'], 0),
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

        # Standardize column names to match training data format
        # This ensures compatibility with merge_training_data.py validation
        column_mapping = {
            'DATE': 'Date',
            'DESCRIPTION': 'Description',
            'DEBIT': 'Amount (Negated)',
            'CREDIT': 'Amount',
            'predicted_category': 'Category'
        }

        # Apply column renaming for columns that exist
        rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=rename_dict)

        logger.info(f"Standardized columns: {list(df.columns)}")

        # Generate standardized corrected filename from original predictions file
        original_predictions_filename = Path(job.result_file).name
        corrected_filename = create_corrected_filename(original_predictions_filename)
        corrected_path = results_dir / corrected_filename
        df.to_csv(corrected_path, index=False, encoding='utf-8')
        
        logger.info(f"Generated corrected file: {corrected_filename} (from: {original_predictions_filename})")
        
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
            
            # Log column information for debugging
            logger.info(f"Accounting download - available columns: {list(df.columns)}")
            
            # Create accounting format with corrected categories using flexible column mapping
            accounting_data = []
            for _, row in df.iterrows():
                accounting_data.append({
                    'Date': get_column_value_flexible(row, 'Date', ['DATE', 'date'], ''),
                    'Description': get_column_value_flexible(row, 'Description', ['DESCRIPTION', 'description'], ''),
                    'Amount (Negated)': get_column_value_flexible(row, 'Amount (Negated)', ['DEBIT', 'debit'], 0),
                    'Amount': get_column_value_flexible(row, 'Amount', ['CREDIT', 'credit'], 0),
                    'Classification': get_column_value_flexible(row, 'Category', ['predicted_category'], '')
                })
            accounting_df = pd.DataFrame(accounting_data)
            
            # Generate standardized accounting filename from predictions filename
            predictions_filename = Path(job.result_file).name
            accounting_filename = create_accounting_from_predictions_filename(predictions_filename)
            accounting_file = results_dir / accounting_filename
            
            # Save accounting file
            accounting_df.to_csv(accounting_file, index=False, encoding='utf-8')
            
            logger.info(f"Generated accounting file: {accounting_filename}")
            
            return FileResponse(
                accounting_file,
                media_type='text/csv',
                filename=accounting_filename,
                headers={"Content-Disposition": f"attachment; filename={accounting_filename}"}
            )
        except Exception as e:
            logger.error(f"Error creating accounting format: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create accounting format: {str(e)}")
    
    else:
        # Default: full predictions format with standardized name
        predictions_filename = Path(job.result_file).name
        logger.info(f"Downloading predictions file: {predictions_filename}")
        
        return FileResponse(
            job.result_file,
            media_type='text/csv',
            filename=predictions_filename,
            headers={"Content-Disposition": f"attachment; filename={predictions_filename}"}
        )

# Background task functions
async def process_file_background(job_id: str, file_path: str, confirmed_source: Optional[str] = None):
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

        # Use user-confirmed source if provided, otherwise use detected source
        if confirmed_source:
            transaction_source = TransactionSource(confirmed_source)
            logger.info(f"Using user-confirmed source: {confirmed_source} for job {job_id}")
        else:
            transaction_source = validation_result.transaction_source
            logger.info(f"Using auto-detected source: {transaction_source.value} ({validation_result.confidence:.0%} confidence) for job {job_id}")
        
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
        
        # Load and process data - handle both CSV and Excel files
        if is_excel_file(file_path):
            excel_processor = ExcelProcessor()
            df = excel_processor.process_excel_file(file_path)
            
            # For ML prediction, we need 'Description' (uppercase) - but save original data for display
            df_for_ml = df.copy()
            df_for_ml = df_for_ml.rename(columns={'description': 'Description'})
            
            # Standardize column names for web interface display
            df = df.rename(columns={
                'date': 'DATE',
                'description': 'DESCRIPTION'
            })
            
            # Split amount into DEBIT/CREDIT columns (accounting convention)
            df['DEBIT'] = df['amount'].where(df['amount'] > 0, 0)
            df['CREDIT'] = df['amount'].where(df['amount'] < 0, 0).abs()
            df = df.drop('amount', axis=1)
            
            # Use the ML-formatted dataframe for predictions
            df_for_predictions = df_for_ml
        else:
            df = pd.read_csv(file_path)
            df_for_predictions = df
        
        # Add transaction source if missing
        if 'transaction_source' not in df.columns:
            df['transaction_source'] = transaction_source.value
        if 'transaction_source' not in df_for_predictions.columns:
            df_for_predictions['transaction_source'] = transaction_source.value
        
        # Run predictions on ML-formatted data
        predictions, probabilities = predictor.predict_batch(df_for_predictions)
        
        # Add predictions to dataframe
        df['predicted_category'] = predictions
        df['confidence'] = probabilities
        
        # Step 3: Save results
        job.progress = 0.8
        job.current_step = "Saving results"
        job.message = "Finalizing predictions and saving results"
        processing_jobs[job_id] = job
        
        # Generate standardized predictions filename by extracting info from file path
        original_filename = Path(file_path).name
        # Remove the file_id prefix (e.g., "file_1_20250809_000237_For Automl..." -> "For Automl...")
        if '_' in original_filename:
            # Find the original filename after the file_id prefix
            parts = original_filename.split('_', 3)  # Split into at most 4 parts
            if len(parts) >= 4:
                actual_filename = parts[3]  # Everything after file_id_timestamp_
            else:
                actual_filename = original_filename
        else:
            actual_filename = original_filename
            
        source, period = extract_source_and_period(actual_filename)
        timestamp = datetime.now()
        
        output_filename = generate_filename(source, period, "predictions", timestamp)
        output_path = results_dir / output_filename
        
        logger.info(f"Generated predictions file: {output_filename} (extracted from: {actual_filename})")
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
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

    print("🚀 Starting Expense Categorization Web Server")
    print("📱 Open http://localhost:8000 in your browser")
    print("🔒 Server bound to localhost only for security")

    # Security: Bind to localhost only to prevent network exposure
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)