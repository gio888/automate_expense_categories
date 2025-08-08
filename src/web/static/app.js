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
                    <button onclick="pipeline.downloadResults('predictions')" class="upload-button" style="background: blue; margin-left: 10px;">
                        📥 Download Predictions
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
                            <th>Amount</th>
                            <th>Predicted Category</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        results.predictions.slice(0, 20).forEach(pred => {
            const confidenceClass = pred.confidence > 0.7 ? 'confidence-high' : 
                                   pred.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
            
            html += `
                <tr class="${confidenceClass}">
                    <td>${pred.date}</td>
                    <td>${pred.description}</td>
                    <td>$${parseFloat(pred.amount || 0).toFixed(2)}</td>
                    <td>${pred.predicted_category}</td>
                    <td>${(pred.confidence * 100).toFixed(1)}%</td>
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
        
        // Show all predictions, not just low confidence
        let html = `
            <div class="status warning">
                <h3>✏️ Review & Correct Categories</h3>
                <p>Review all ${this.predictions.length} predictions. Most should be correct already.</p>
                <div style="margin-top: 15px;">
                    <button onclick="pipeline.submitCorrections()" class="upload-button">💾 Save Corrections & Export</button>
                    <button onclick="pipeline.goBackToResults()" class="upload-button" style="background: gray; margin-left: 10px;">Cancel</button>
                </div>
            </div>
            
            <div style="max-height: 600px; overflow-y: auto; margin-top: 20px;">
                <table class="predictions-table" id="correctionsTable">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Description</th>
                            <th>Debit</th>
                            <th>Credit</th>
                            <th>ML Prediction</th>
                            <th>Confidence</th>
                            <th>Corrected Category</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        this.predictions.forEach(pred => {
            const confidenceClass = pred.confidence > 0.7 ? 'confidence-high' : 
                                   pred.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
            
            // Format amounts for debit/credit display
            const debitAmount = parseFloat(pred['Amount (Negated)'] || 0);
            const creditAmount = parseFloat(pred['Amount'] || 0);
            
            html += `
                <tr class="${confidenceClass}">
                    <td>${pred.Date}</td>
                    <td title="${pred.Description}" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${pred.Description}</td>
                    <td>${debitAmount ? debitAmount.toFixed(2) : ''}</td>
                    <td>${creditAmount ? creditAmount.toFixed(2) : ''}</td>
                    <td><span style="font-size: 0.9em; color: #666;">${pred.predicted_category}</span></td>
                    <td>${(pred.confidence * 100).toFixed(0)}%</td>
                    <td>
                        <select id="correction_${pred.id}" class="category-dropdown" style="width: 100%; padding: 4px;" onchange="pipeline.onCategoryChange(${pred.id})">
                            <option value="${pred.predicted_category}" selected>${pred.predicted_category}</option>
                        </select>
                    </td>
                </tr>
            `;
        });
        
        html += '</tbody></table></div>';
        
        resultsDiv.innerHTML = html;
        
        // Initialize searchable dropdowns after HTML is inserted
        setTimeout(() => this.initializeCategoryDropdowns(), 100);
    }
    
    initializeCategoryDropdowns() {
        // Add searchable functionality to each dropdown
        this.predictions.forEach(pred => {
            const dropdown = document.getElementById(`correction_${pred.id}`);
            if (dropdown) {
                this.makeDropdownSearchable(dropdown, pred.id);
            }
        });
    }
    
    makeDropdownSearchable(dropdown, predId) {
        // Replace select with searchable input + dropdown
        const wrapper = document.createElement('div');
        wrapper.className = 'searchable-dropdown';
        wrapper.style.position = 'relative';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.id = `search_${predId}`;
        input.value = dropdown.value;
        input.style = 'width: 100%; padding: 4px; border: 1px solid #ddd;';
        input.placeholder = 'Type to search categories...';
        
        const dropdownList = document.createElement('div');
        dropdownList.id = `dropdown_${predId}`;
        dropdownList.className = 'dropdown-list';
        dropdownList.style = `
            position: absolute; 
            top: 100%; 
            left: 0; 
            right: 0; 
            background: white; 
            border: 1px solid #ddd; 
            max-height: 200px; 
            overflow-y: auto; 
            z-index: 1000;
            display: none;
        `;
        
        wrapper.appendChild(input);
        wrapper.appendChild(dropdownList);
        
        // Replace the original dropdown
        dropdown.parentNode.replaceChild(wrapper, dropdown);
        
        // Add search functionality
        input.addEventListener('input', (e) => {
            this.filterCategories(predId, e.target.value);
        });
        
        input.addEventListener('focus', () => {
            this.showCategoryDropdown(predId, input.value);
        });
        
        // Hide dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!wrapper.contains(e.target)) {
                dropdownList.style.display = 'none';
            }
        });
    }
    
    filterCategories(predId, searchTerm) {
        const dropdownList = document.getElementById(`dropdown_${predId}`);
        const filteredCategories = this.validCategories.filter(cat => 
            cat.toLowerCase().includes(searchTerm.toLowerCase())
        );
        
        let html = '';
        filteredCategories.slice(0, 10).forEach(category => { // Limit to 10 results
            html += `
                <div class="dropdown-item" style="padding: 8px; cursor: pointer; hover: background-color: #f5f5f5;" 
                     onclick="pipeline.selectCategory(${predId}, '${category.replace(/'/g, "\\'")}')">
                    ${category}
                </div>
            `;
        });
        
        if (filteredCategories.length === 0) {
            html = '<div style="padding: 8px; color: #666;">No categories found</div>';
        }
        
        dropdownList.innerHTML = html;
        dropdownList.style.display = 'block';
    }
    
    showCategoryDropdown(predId, currentValue = '') {
        this.filterCategories(predId, currentValue);
    }
    
    selectCategory(predId, category) {
        const input = document.getElementById(`search_${predId}`);
        const dropdown = document.getElementById(`dropdown_${predId}`);
        
        if (input) {
            input.value = category;
        }
        
        if (dropdown) {
            dropdown.style.display = 'none';
        }
        
        // Update the prediction in memory
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
    
    goBackToResults() {
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
                    <button onclick="pipeline.downloadResults('accounting')" class="upload-button">
                        📥 Download for Accounting System
                    </button>
                    <button onclick="pipeline.downloadResults('predictions')" class="upload-button" style="background: blue; margin-left: 10px;">
                        📥 Download Full Predictions
                    </button>
                    <button onclick="pipeline.retrainModels()" class="upload-button" style="background: green; margin-left: 10px;">
                        🎓 Retrain Models
                    </button>
                </div>
                
                <div style="margin-top: 15px; font-size: 0.9em; color: #666;">
                    <p><strong>Accounting Format:</strong> Date, Description, Amount (Negated), Amount, Classification</p>
                    <p><strong>Ready for import:</strong> Your accounting system can import this file directly</p>
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
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            // Set appropriate filename based on format
            if (format === 'accounting') {
                a.download = `accounting_import_${this.currentJobId}.csv`;
                this.showStatus('📥 Accounting file downloaded! Ready for import into your accounting system.', 'success');
            } else {
                a.download = `expense_predictions_${this.currentJobId}.csv`;
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

// Initialize the pipeline when the page loads
const pipeline = new ExpensePipeline();