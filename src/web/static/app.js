/**
 * Frontend JavaScript for the Expense Categorization Pipeline
 */

class ExpensePipeline {
    constructor() {
        this.currentJobId = null;
        this.currentFileId = null;
        this.predictions = [];
        this.corrections = [];
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        
        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
    }
    
    async handleFileUpload(file) {
        try {
            this.showStatus('Uploading file...', 'info');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || 'Upload failed');
            }
            
            this.currentFileId = result.file_id;
            this.displayFileInfo(result);
            
            if (result.validation_result.is_valid) {
                this.showProcessButton();
            } else {
                this.showValidationIssues(result.validation_result);
            }
            
        } catch (error) {
            this.showStatus(`Upload failed: ${error.message}`, 'error');
        }
    }
    
    displayFileInfo(uploadResult) {
        const fileInfoDiv = document.getElementById('fileInfo');
        const validation = uploadResult.validation_result;
        
        const confidenceText = `${(validation.confidence * 100).toFixed(0)}%`;
        const sourceText = validation.transaction_source ? 
            validation.transaction_source.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Unknown';
        
        fileInfoDiv.innerHTML = `
            <div class="file-info">
                <h3>📄 File Information</h3>
                <p><strong>Filename:</strong> ${uploadResult.filename}</p>
                <p><strong>Rows:</strong> ${validation.row_count.toLocaleString()}</p>
                <p><strong>Size:</strong> ${this.formatFileSize(validation.file_size)}</p>
                <p><strong>Detected Source:</strong> ${sourceText} (${confidenceText} confidence)</p>
                <p><strong>Status:</strong> ${validation.is_valid ? 
                    '<span style="color: green;">✅ Valid</span>' : 
                    '<span style="color: orange;">⚠️ Has Issues</span>'}</p>
            </div>
        `;
        
        fileInfoDiv.style.display = 'block';
    }
    
    showValidationIssues(validation) {
        const fileInfoDiv = document.getElementById('fileInfo');
        
        let issuesHtml = '<div class="status warning"><h4>⚠️ Validation Issues</h4><ul>';
        validation.issues.forEach(issue => {
            issuesHtml += `<li>${issue}</li>`;
        });
        issuesHtml += '</ul>';
        
        if (validation.suggestions.length > 0) {
            issuesHtml += '<h4>💡 Suggestions</h4><ul>';
            validation.suggestions.forEach(suggestion => {
                issuesHtml += `<li>${suggestion}</li>`;
            });
            issuesHtml += '</ul>';
        }
        
        issuesHtml += `
            <button onclick="pipeline.processAnyway()" class="upload-button" style="background: orange; margin-top: 10px;">
                Process Anyway
            </button>
        </div>`;
        
        fileInfoDiv.innerHTML += issuesHtml;
    }
    
    showProcessButton() {
        const fileInfoDiv = document.getElementById('fileInfo');
        fileInfoDiv.innerHTML += `
            <button onclick="pipeline.startProcessing()" class="upload-button" style="margin-top: 15px;">
                🚀 Start Processing
            </button>
        `;
    }
    
    async processAnyway() {
        this.showStatus('Processing file despite validation issues...', 'warning');
        await this.startProcessing();
    }
    
    async startProcessing() {
        try {
            this.showStatus('Starting processing...', 'info');
            
            const response = await fetch(`/process/${this.currentFileId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || 'Processing failed');
            }
            
            this.currentJobId = result.job_id;
            this.startProgressPolling();
            
        } catch (error) {
            this.showStatus(`Processing failed: ${error.message}`, 'error');
        }
    }
    
    startProgressPolling() {
        const progressDiv = document.getElementById('progress');
        progressDiv.innerHTML = `
            <h3>📊 Processing Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
            </div>
            <p id="progressText">Initializing...</p>
        `;
        progressDiv.style.display = 'block';
        
        this.pollStatus();
    }
    
    async pollStatus() {
        try {
            const response = await fetch(`/status/${this.currentJobId}`);
            const status = await response.json();
            
            if (!response.ok) {
                throw new Error('Status check failed');
            }
            
            this.updateProgress(status);
            
            if (status.status === 'completed') {
                await this.loadResults();
            } else if (status.status === 'failed') {
                this.showStatus(`Processing failed: ${status.error}`, 'error');
            } else {
                // Continue polling
                setTimeout(() => this.pollStatus(), 2000);
            }
            
        } catch (error) {
            this.showStatus(`Status check failed: ${error.message}`, 'error');
        }
    }
    
    updateProgress(status) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        if (progressFill) {
            progressFill.style.width = `${status.progress * 100}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${status.current_step}: ${status.message}`;
        }
    }
    
    
    // FIXED: Safe data access method
    safeGetValue(obj, key, defaultValue = '') {
        return obj && obj[key] !== undefined && obj[key] !== null ? obj[key] : defaultValue;
    }
    
    async loadResults() {
        try {
            const response = await fetch(`/results/${this.currentJobId}`);
            const results = await response.json();
            
            if (!response.ok) {
                throw new Error('Results loading failed');
            }
            
            this.predictions = results.predictions;
            this.displayResults(results);
            
        } catch (error) {
            this.showStatus(`Results loading failed: ${error.message}`, 'error');
        }
    }
    
    displayResults(results) {
        const resultsDiv = document.getElementById('results');
        
        // Store results and categories for corrections
        this.predictions = results.predictions;
        this.validCategories = results.valid_categories || [];
        
        // Summary
        let html = `
            <div class="status success">
                <h3>🎉 Processing Complete!</h3>
                <p><strong>Total Transactions:</strong> ${results.total_rows}</p>
                <p><strong>High Confidence:</strong> ${results.high_confidence} (${((results.high_confidence/results.total_rows)*100).toFixed(1)}%)</p>
                <p><strong>Medium Confidence:</strong> ${results.medium_confidence} (${((results.medium_confidence/results.total_rows)*100).toFixed(1)}%)</p>
                <p><strong>Low Confidence:</strong> ${results.low_confidence} (${((results.low_confidence/results.total_rows)*100).toFixed(1)}%)</p>
                <p><strong>Available Categories:</strong> ${this.validCategories.length} loaded</p>
                <div style="margin-top: 15px;">
                    <button onclick="pipeline.showCorrectionInterface()" class="upload-button">
                        ✏️ Review & Correct Categories
                    </button>
                    <button onclick="pipeline.downloadResults('predictions')" class="upload-button" style="background: blue; margin-left: 10px;" title="All columns including confidence scores and metadata">
                        📥 Download Full Predictions
                    </button>
                </div>
            </div>
        `;
        
        // Predictions table (show first 20 rows)
        html += `
            <h3>📋 Predictions Preview</h3>
            <div style="max-height: 400px; overflow-y: auto;">
                <table class="predictions-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Description</th>
                            <th>Amount (Negated)</th>
                            <th>Amount</th>
                            <th>Predicted Category</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
                // FIXED: Add ID if missing for correction interface
        results.predictions.forEach((pred, index) => {
            if (!pred.id) {
                pred.id = index;
            }
        });
        
        results.predictions.slice(0, 20).forEach(pred => {
            const confidenceClass = pred.confidence > 0.7 ? 'confidence-high' : 
                                   pred.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
            
            html += `
                <tr class="${confidenceClass}">
                    <td>${this.safeGetValue(pred, "Date", "N/A")}</td>
                    <td>${this.safeGetValue(pred, "Description", "N/A")}</td>
                    <td>${parseFloat(this.safeGetValue(pred, "Amount (Negated)", 0)).toFixed(2)}</td>
                    <td>${parseFloat(this.safeGetValue(pred, "Amount", 0)).toFixed(2)}</td>
                    <td>${this.safeGetValue(pred, "predicted_category", "Unknown")}</td>
                    <td>${(parseFloat(this.safeGetValue(pred, "confidence", 0)) * 100).toFixed(1)}%</td>
                </tr>
            `;
        });
        
        html += '</tbody></table></div>';
        
        if (results.total_rows > 20) {
            html += `<p style="color: #666; font-style: italic;">Showing first 20 of ${results.total_rows} transactions</p>`;
        }
        
        resultsDiv.innerHTML = html;
        resultsDiv.style.display = 'block';
    }
    
    showCorrectionInterface() {
        const resultsDiv = document.getElementById('results');
        
        // Expand container for correction interface
        const container = document.querySelector('.container');
        container.classList.add('expanded');
        
        let html = `
            <div class="status warning">
                <h3>✏️ Review & Correct Categories</h3>
                <p>Review all ${this.predictions.length} predictions. The table below is fully responsive and adapts to your screen size.</p>
                <div style="margin-top: 1rem; display: flex; gap: 0.75rem; flex-wrap: wrap;">
                    <button onclick="pipeline.submitCorrections()" class="upload-button" aria-label="Save corrections and export data">
                        💾 Save Corrections & Export
                    </button>
                    <button onclick="pipeline.goBackToResults()" class="upload-button" style="background: #6c757d;" aria-label="Cancel corrections and go back">
                        ← Cancel
                    </button>
                </div>
            </div>
            
            <div class="responsive-table-container">
                <table class="responsive-table" id="correctionsTable" role="table" aria-label="Transaction corrections table">
                    <thead>
                        <tr role="row">
                            <th scope="col" class="col-date sortable" data-column="Date" aria-sort="none" role="button" tabindex="0">
                                Date
                                <span class="sort-indicator" aria-hidden="true"></span>
                            </th>
                            <th scope="col" class="col-description sortable" data-column="Description" aria-sort="none" role="button" tabindex="0">
                                Description
                                <span class="sort-indicator" aria-hidden="true"></span>
                            </th>
                            <th scope="col" class="col-debit sortable" data-column="Amount (Negated)" aria-sort="none" role="button" tabindex="0">
                                Debit
                                <span class="sort-indicator" aria-hidden="true"></span>
                            </th>
                            <th scope="col" class="col-credit sortable" data-column="Amount" aria-sort="none" role="button" tabindex="0">
                                Credit
                                <span class="sort-indicator" aria-hidden="true"></span>
                            </th>
                            <th scope="col" class="col-prediction sortable" data-column="predicted_category" aria-sort="none" role="button" tabindex="0">
                                ML Prediction
                                <span class="sort-indicator" aria-hidden="true"></span>
                            </th>
                            <th scope="col" class="col-confidence sortable" data-column="confidence" aria-sort="none" role="button" tabindex="0">
                                Confidence
                                <span class="sort-indicator" aria-hidden="true"></span>
                            </th>
                            <th scope="col" class="col-correction">Corrected Category</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        this.predictions.forEach((pred, index) => {
            const confidenceClass = pred.confidence > 0.7 ? 'confidence-high' : 
                                   pred.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
            
            const confidencePercent = Math.round(pred.confidence * 100);
            
            // Format amounts for debit/credit display
            const debitAmount = parseFloat(this.safeGetValue(pred, "Amount (Negated)", 0));
            const creditAmount = parseFloat(this.safeGetValue(pred, "Amount", 0));
            
            html += `
                <tr class="${confidenceClass}" role="row" tabindex="0" aria-rowindex="${index + 2}">
                    <td class="col-date" role="gridcell">${this.safeGetValue(pred, "Date", "N/A")}</td>
                    <td class="col-description" role="gridcell" title="${this.safeGetValue(pred, "Description", "N/A")}">${this.safeGetValue(pred, "Description", "N/A")}</td>
                    <td class="col-debit" role="gridcell">${debitAmount ? debitAmount.toFixed(2) : ''}</td>
                    <td class="col-credit" role="gridcell">${creditAmount ? creditAmount.toFixed(2) : ''}</td>
                    <td class="col-prediction" role="gridcell" title="${this.safeGetValue(pred, "predicted_category", "Unknown")}">${this.safeGetValue(pred, "predicted_category", "Unknown")}</td>
                    <td class="col-confidence" role="gridcell">
                        <span class="confidence-badge">${confidencePercent}%</span>
                    </td>
                    <td class="col-correction" role="gridcell">
                        <div class="searchable-dropdown" role="combobox" aria-expanded="false" aria-haspopup="listbox">
                            <input 
                                type="text" 
                                id="search_${pred.id}" 
                                value="${this.safeGetValue(pred, "predicted_category", "Unknown")}"
                                placeholder="Type to search categories..."
                                aria-label="Search and select category for ${this.safeGetValue(pred, "Description", "transaction")}"
                                autocomplete="off"
                                role="textbox"
                            />
                            <div id="dropdown_${pred.id}" class="dropdown-list" role="listbox" aria-label="Category options"></div>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        html += `
                </tbody>
            </table>
        </div>
        <div style="margin-top: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; text-align: center; color: #6c757d; font-size: 0.875rem;">
            <p style="margin: 0;"><strong>✨ Responsive Design:</strong> This table adapts to your screen size. All columns are visible without horizontal scrolling on desktop screens.</p>
            <p style="margin: 0.5rem 0 0 0;"><strong>Tip:</strong> Use Tab key to navigate between fields for keyboard accessibility.</p>
        </div>
        `;
        
        resultsDiv.innerHTML = html;
        
        // Initialize enhanced functionality after HTML is inserted
        setTimeout(() => {
            this.initializeCategoryDropdowns();
            this.initializeSortableColumns();
        }, 100);
        
        // Add keyboard navigation
        this.addKeyboardNavigation();
    }
    
    initializeCategoryDropdowns() {
        // Initialize enhanced category selectors for each prediction
        this.predictions.forEach(pred => {
            const inputElement = document.getElementById(`search_${pred.id}`);
            if (inputElement) {
                new CategorySelector(inputElement, this.validCategories, pred.id, this);
            }
        });
    }
    
    initializeSortableColumns() {
        const table = document.getElementById('correctionsTable');
        if (table) {
            this.tableSorter = new TableSorter(table, this);
        }
    }
    
    // Enhanced category selection handled by CategorySelector class
    selectCategory(predId, category) {
        // This method is now handled by the CategorySelector class
        // Keeping for backward compatibility
        const prediction = this.predictions.find(p => p.id === predId);
        if (prediction) {
            prediction.corrected_category = category;
        }
    }
    
    onCategoryChange(predId) {
        const dropdown = document.getElementById(`correction_${predId}`);
        if (dropdown) {
            const prediction = this.predictions.find(p => p.id === predId);
            if (prediction) {
                prediction.corrected_category = dropdown.value;
            }
        }
    }
    
    addKeyboardNavigation() {
        const table = document.getElementById('correctionsTable');
        if (!table) return;
        
        // Add keyboard navigation for table rows
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach((row, index) => {
            row.addEventListener('keydown', (e) => {
                switch(e.key) {
                    case 'ArrowDown':
                        e.preventDefault();
                        if (index < rows.length - 1) {
                            rows[index + 1].focus();
                        }
                        break;
                    case 'ArrowUp':
                        e.preventDefault();
                        if (index > 0) {
                            rows[index - 1].focus();
                        }
                        break;
                    case 'Enter':
                    case ' ':
                        e.preventDefault();
                        const input = row.querySelector('input');
                        if (input) input.focus();
                        break;
                }
            });
        });
    }

    goBackToResults() {
        const container = document.querySelector('.container');
        container.classList.remove('expanded');
        
        // Recreate results view
        const results = {
            total_rows: this.predictions.length,
            high_confidence: this.predictions.filter(p => p.confidence > 0.7).length,
            medium_confidence: this.predictions.filter(p => p.confidence >= 0.5 && p.confidence <= 0.7).length,
            low_confidence: this.predictions.filter(p => p.confidence < 0.5).length,
            predictions: this.predictions,
            valid_categories: this.validCategories
        };
        this.displayResults(results);
    }
    
    async submitCorrections() {
        try {
            // Collect corrections from searchable inputs
            const corrections = [];
            
            this.predictions.forEach(pred => {
                const searchInput = document.getElementById(`search_${pred.id}`);
                if (searchInput) {
                    const correctedCategory = searchInput.value.trim();
                    if (correctedCategory && correctedCategory !== pred.predicted_category) {
                        corrections.push({
                            id: pred.id,
                            predicted_category: correctedCategory
                        });
                    }
                    // Update prediction with corrected category (even if same as original)
                    pred.corrected_category = correctedCategory || pred.predicted_category;
                } else {
                    // Fallback: use original prediction if no correction made
                    pred.corrected_category = pred.predicted_category;
                }
            });
            
            this.showStatus(`Processing ${corrections.length} corrections and generating accounting file...`, 'info');
            
            // Submit corrections to backend
            if (corrections.length > 0) {
                const response = await fetch(`/correct/${this.currentJobId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        file_id: this.currentFileId,
                        corrections: corrections
                    })
                });
                
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.detail || 'Corrections failed');
                }
            }
            
            // Show completion interface
            this.showCompletionInterface(corrections.length);
            
        } catch (error) {
            this.showStatus(`Error processing corrections: ${error.message}`, 'error');
        }
    }
    
    showCompletionInterface(correctionCount) {
        const resultsDiv = document.getElementById('results');
        
        resultsDiv.innerHTML = `
            <div class="status success">
                <h3>✅ Corrections Complete!</h3>
                <p><strong>Total Transactions:</strong> ${this.predictions.length}</p>
                <p><strong>Manual Corrections:</strong> ${correctionCount}</p>
                <p>Your corrected data is ready for download in accounting system format.</p>
                
                <div style="margin-top: 20px;">
                    <button onclick="pipeline.downloadResults('accounting')" class="upload-button" title="5 columns: Date, Description, Amount (Negated), Amount, Classification - Ready for direct import">
                        📥 Download for Accounting System
                    </button>
                    <button onclick="pipeline.downloadResults('predictions')" class="upload-button" style="background: blue; margin-left: 10px;" title="All columns including confidence scores and metadata">
                        📥 Download Full Predictions
                    </button>
                    <button onclick="pipeline.retrainModels()" class="upload-button" style="background: green; margin-left: 10px;">
                        🎓 Retrain Models
                    </button>
                </div>
                
                <div style="margin-top: 15px; font-size: 0.9em; color: #666;">
                    <p><strong>📊 Accounting Format:</strong> Date, Description, Amount (Negated), Amount, Classification (5 columns)</p>
                    <p><strong>✅ Ready for import:</strong> Your accounting system can import this file directly</p>
                    <p><strong>📋 Full Predictions:</strong> Includes confidence scores, transaction source, and all metadata</p>
                    <p><strong>📁 File Naming:</strong> Files use standardized names: {source}_{period}_{stage}_{timestamp}.csv</p>
                </div>
            </div>
        `;
    }
    
    showRetrainButton() {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = `
            <div class="status success">
                <h3>✅ Corrections Applied!</h3>
                <p>Your corrections have been saved. You can now retrain the models to improve future predictions.</p>
                <button onclick="pipeline.retrainModels()" class="upload-button">🎓 Retrain Models</button>
                <button onclick="pipeline.downloadResults()" class="upload-button" style="background: green; margin-left: 10px;">
                    📥 Download Corrected Results
                </button>
            </div>
        `;
    }
    
    async retrainModels() {
        try {
            this.showStatus('Starting model retraining...', 'info');
            
            const response = await fetch(`/retrain/${this.currentJobId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || 'Retraining failed');
            }
            
            this.showStatus('🎓 Model retraining started in background. This may take several minutes...', 'success');
            
        } catch (error) {
            this.showStatus(`Retraining failed: ${error.message}`, 'error');
        }
    }
    
    async downloadResults(format = 'predictions') {
        try {
            const formatParam = format === 'accounting' ? '?format=accounting' : '';
            const response = await fetch(`/download/${this.currentJobId}${formatParam}`);
            
            if (!response.ok) {
                throw new Error('Download failed');
            }
            
            // Extract filename from Content-Disposition header or fallback to response filename
            let filename = this.extractFilenameFromResponse(response, format);
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            
            // Show appropriate success message
            if (format === 'accounting') {
                this.showStatus('📥 Accounting file downloaded! Ready for import into your accounting system.', 'success');
            } else {
                this.showStatus('📥 Predictions file downloaded!', 'success');
            }
            
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
        } catch (error) {
            this.showStatus(`Download failed: ${error.message}`, 'error');
        }
    }
    
    extractFilenameFromResponse(response, format) {
        // Try to extract filename from Content-Disposition header
        const contentDisposition = response.headers.get('Content-Disposition');
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
                let filename = filenameMatch[1].replace(/['"]/g, '');
                if (filename) {
                    return filename;
                }
            }
        }
        
        // Fallback: try to get filename from response URL or generate generic name
        const url = response.url;
        if (url && url.includes('/download/')) {
            // Extract job ID for fallback naming
            const jobIdMatch = url.match(/\/download\/([^?]+)/);
            if (jobIdMatch) {
                const jobId = jobIdMatch[1];
                if (format === 'accounting') {
                    return `accounting_${jobId}.csv`;
                } else {
                    return `predictions_${jobId}.csv`;
                }
            }
        }
        
        // Final fallback
        return format === 'accounting' ? 'accounting_export.csv' : 'predictions_export.csv';
    }
    
    showStatus(message, type) {
        // Remove existing status messages
        const existingStatus = document.querySelector('.status');
        if (existingStatus && !existingStatus.classList.contains('success')) {
            existingStatus.remove();
        }
        
        const statusDiv = document.createElement('div');
        statusDiv.className = `status ${type}`;
        statusDiv.textContent = message;
        
        // Insert after the upload zone
        const uploadZone = document.getElementById('uploadZone');
        uploadZone.parentNode.insertBefore(statusDiv, uploadZone.nextSibling);
        
        // Auto-remove info messages after 5 seconds
        if (type === 'info') {
            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.remove();
                }
            }, 5000);
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
}

/**
 * Enhanced Category Selector with modern UX patterns
 */
class CategorySelector {
    constructor(inputElement, categories, predictionId, pipeline) {
        this.input = inputElement;
        this.categories = categories;
        this.predictionId = predictionId;
        this.pipeline = pipeline;
        this.dropdown = document.getElementById(`dropdown_${predictionId}`);
        this.isOpen = false;
        this.selectedIndex = -1;
        this.filteredCategories = [];
        
        this.init();
    }
    
    init() {
        // Enhanced event listeners with modern UX patterns
        this.input.addEventListener('input', (e) => this.handleInput(e));
        this.input.addEventListener('focus', () => this.handleFocus());
        this.input.addEventListener('blur', (e) => this.handleBlur(e));
        this.input.addEventListener('keydown', (e) => this.handleKeydown(e));
        
        // Click to open/close dropdown
        this.input.addEventListener('click', () => {
            if (!this.isOpen) {
                this.showDropdown();
            }
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.input.closest('.searchable-dropdown').contains(e.target)) {
                this.hideDropdown();
            }
        });
    }
    
    handleInput(e) {
        const searchTerm = e.target.value;
        this.input.classList.add('searching');
        this.filterAndShowCategories(searchTerm);
        this.selectedIndex = -1; // Reset selection on new input
    }
    
    handleFocus() {
        this.showDropdown();
    }
    
    handleBlur(e) {
        // Delay hiding to allow for item clicks
        setTimeout(() => {
            if (!this.dropdown.contains(document.activeElement)) {
                this.hideDropdown();
            }
        }, 150);
    }
    
    handleKeydown(e) {
        switch(e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.navigateDown();
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.navigateUp();
                break;
            case 'Enter':
                e.preventDefault();
                if (this.selectedIndex >= 0 && this.filteredCategories[this.selectedIndex]) {
                    this.selectCategory(this.filteredCategories[this.selectedIndex]);
                }
                break;
            case 'Escape':
                this.hideDropdown();
                break;
            case 'Tab':
                this.hideDropdown();
                break;
        }
    }
    
    filterAndShowCategories(searchTerm) {
        this.filteredCategories = this.categories.filter(cat => 
            cat.toLowerCase().includes(searchTerm.toLowerCase())
        ).slice(0, 8); // Limit to 8 results for better UX
        
        this.renderDropdown();
        this.showDropdown();
    }
    
    renderDropdown() {
        if (this.filteredCategories.length === 0) {
            this.dropdown.innerHTML = '<div class="dropdown-empty">No matching categories</div>';
            return;
        }
        
        let html = '';
        this.filteredCategories.forEach((category, index) => {
            const isHighlighted = index === this.selectedIndex;
            const isSelected = this.input.value === category;
            
            html += `
                <div class="dropdown-item ${
                    isHighlighted ? 'highlighted' : ''
                } ${
                    isSelected ? 'selected' : ''
                }" 
                     role="option" 
                     aria-selected="${isSelected}"
                     data-index="${index}"
                     data-category="${category.replace(/'/g, '&apos;')}"
                >
                    ${category}
                </div>
            `;
        });
        
        this.dropdown.innerHTML = html;
        
        // Add click handlers to dropdown items
        this.dropdown.querySelectorAll('.dropdown-item').forEach((item, index) => {
            item.addEventListener('mousedown', (e) => {
                e.preventDefault(); // Prevent blur
                const category = item.getAttribute('data-category').replace(/&apos;/g, "'");
                this.selectCategory(category);
            });
            
            item.addEventListener('mouseenter', () => {
                this.selectedIndex = index;
                this.updateHighlight();
            });
        });
    }
    
    navigateDown() {
        if (this.selectedIndex < this.filteredCategories.length - 1) {
            this.selectedIndex++;
            this.updateHighlight();
            this.scrollToSelected();
        }
    }
    
    navigateUp() {
        if (this.selectedIndex > 0) {
            this.selectedIndex--;
            this.updateHighlight();
            this.scrollToSelected();
        }
    }
    
    updateHighlight() {
        this.dropdown.querySelectorAll('.dropdown-item').forEach((item, index) => {
            item.classList.toggle('highlighted', index === this.selectedIndex);
        });
    }
    
    scrollToSelected() {
        const selectedItem = this.dropdown.querySelector('.dropdown-item.highlighted');
        if (selectedItem) {
            selectedItem.scrollIntoView({ block: 'nearest' });
        }
    }
    
    selectCategory(category) {
        this.input.value = category;
        this.input.classList.remove('searching');
        this.input.classList.add('selected');
        
        // Update the prediction in memory
        const prediction = this.pipeline.predictions.find(p => p.id === this.predictionId);
        if (prediction) {
            prediction.corrected_category = category;
        }
        
        this.hideDropdown();
        
        // Visual feedback for successful selection
        setTimeout(() => {
            this.input.classList.remove('selected');
        }, 1000);
    }
    
    showDropdown() {
        this.dropdown.style.display = 'block';
        this.isOpen = true;
        this.input.closest('.searchable-dropdown').setAttribute('aria-expanded', 'true');
        
        // Show all categories if no search term
        if (!this.input.value.trim()) {
            this.filteredCategories = this.categories.slice(0, 8);
            this.renderDropdown();
        }
    }
    
    hideDropdown() {
        this.dropdown.style.display = 'none';
        this.isOpen = false;
        this.selectedIndex = -1;
        this.input.classList.remove('searching');
        this.input.closest('.searchable-dropdown').setAttribute('aria-expanded', 'false');
    }
}

/**
 * Sortable columns functionality
 */
class TableSorter {
    constructor(tableElement, pipeline) {
        this.table = tableElement;
        this.pipeline = pipeline;
        this.currentSort = { column: null, direction: null };
        this.originalData = [...pipeline.predictions];
        
        this.init();
    }
    
    init() {
        const headers = this.table.querySelectorAll('th.sortable');
        headers.forEach(header => {
            header.addEventListener('click', () => this.handleSort(header));
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.handleSort(header);
                }
            });
        });
    }
    
    handleSort(header) {
        const column = header.getAttribute('data-column');
        let direction = 'asc';
        
        // Toggle direction if same column
        if (this.currentSort.column === column) {
            direction = this.currentSort.direction === 'asc' ? 'desc' : 'asc';
        }
        
        this.sortTable(column, direction);
        this.updateSortIndicators(header, direction);
        
        this.currentSort = { column, direction };
    }
    
    sortTable(column, direction) {
        const sortedData = [...this.pipeline.predictions].sort((a, b) => {
            let aVal = this.pipeline.safeGetValue(a, column, '');
            let bVal = this.pipeline.safeGetValue(b, column, '');
            
            // Handle numeric columns
            if (['Amount (Negated)', 'Amount', 'confidence'].includes(column)) {
                aVal = parseFloat(aVal) || 0;
                bVal = parseFloat(bVal) || 0;
            }
            // Handle date columns
            else if (column === 'Date') {
                aVal = new Date(aVal);
                bVal = new Date(bVal);
            }
            // Handle string columns
            else {
                aVal = String(aVal).toLowerCase();
                bVal = String(bVal).toLowerCase();
            }
            
            if (aVal < bVal) return direction === 'asc' ? -1 : 1;
            if (aVal > bVal) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        
        // Update the pipeline data
        this.pipeline.predictions = sortedData;
        
        // Re-render the table body
        this.renderTableBody();
    }
    
    renderTableBody() {
        const tbody = this.table.querySelector('tbody');
        let html = '';
        
        this.pipeline.predictions.forEach((pred, index) => {
            const confidenceClass = pred.confidence > 0.7 ? 'confidence-high' : 
                                   pred.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
            
            const confidencePercent = Math.round(pred.confidence * 100);
            const debitAmount = parseFloat(this.pipeline.safeGetValue(pred, "Amount (Negated)", 0));
            const creditAmount = parseFloat(this.pipeline.safeGetValue(pred, "Amount", 0));
            
            html += `
                <tr class="${confidenceClass}" role="row" tabindex="0" aria-rowindex="${index + 2}">
                    <td class="col-date" role="gridcell">${this.pipeline.safeGetValue(pred, "Date", "N/A")}</td>
                    <td class="col-description" role="gridcell" title="${this.pipeline.safeGetValue(pred, "Description", "N/A")}">${this.pipeline.safeGetValue(pred, "Description", "N/A")}</td>
                    <td class="col-debit" role="gridcell">${debitAmount ? debitAmount.toFixed(2) : ''}</td>
                    <td class="col-credit" role="gridcell">${creditAmount ? creditAmount.toFixed(2) : ''}</td>
                    <td class="col-prediction" role="gridcell" title="${this.pipeline.safeGetValue(pred, "predicted_category", "Unknown")}">${this.pipeline.safeGetValue(pred, "predicted_category", "Unknown")}</td>
                    <td class="col-confidence" role="gridcell">
                        <span class="confidence-badge">${confidencePercent}%</span>
                    </td>
                    <td class="col-correction" role="gridcell">
                        <div class="searchable-dropdown" role="combobox" aria-expanded="false" aria-haspopup="listbox">
                            <input 
                                type="text" 
                                id="search_${pred.id}" 
                                value="${this.pipeline.safeGetValue(pred, "predicted_category", "Unknown")}"
                                placeholder="Type to search categories..."
                                aria-label="Search and select category for ${this.pipeline.safeGetValue(pred, "Description", "transaction")}"
                                autocomplete="off"
                                role="textbox"
                            />
                            <div id="dropdown_${pred.id}" class="dropdown-list" role="listbox" aria-label="Category options"></div>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        tbody.innerHTML = html;
        
        // Re-initialize category dropdowns after re-rendering
        setTimeout(() => this.pipeline.initializeCategoryDropdowns(), 50);
    }
    
    updateSortIndicators(activeHeader, direction) {
        // Clear all sort indicators
        this.table.querySelectorAll('th.sortable').forEach(header => {
            header.classList.remove('sort-asc', 'sort-desc');
            header.setAttribute('aria-sort', 'none');
        });
        
        // Set active sort indicator
        activeHeader.classList.add(`sort-${direction}`);
        activeHeader.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');
    }
}

// Initialize the pipeline when the page loads
const pipeline = new ExpensePipeline();