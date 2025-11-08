# Future Improvements & Tasks

## Recently Completed

### 2025-11-08: Column Mapping Fix
- [x] Fixed merge_training_data.py to handle web interface column format
- [x] Added mapping for DATE→Date, DESCRIPTION→Description, DEBIT/CREDIT→Amount columns
- [x] Cleaned up 10 legacy CSV upload files from data/uploads/

## High Priority - Project Organization & Documentation

### **File Structure Reorganization**
- [ ] Create `scripts/` directory for production entry points
- [ ] Create `tools/` directory for diagnostic/testing utilities
- [ ] Move production files to `scripts/`:
  - `start_web_server.py` (main web interface entry point)
  - `setup_validator.py` (setup validation tool)
  - `generate_categories.py` (category generation from personal config)
  - `process_transactions.py` (unified CLI workflow)
- [ ] Move diagnostic files to `tools/`:
  - `check_latest_data.py` (shows latest training data/models)
  - `check_registry.py` (model registry diagnostics)
  - `check_python.py` (Python environment info)
  - `test_file_detection.py` (file detection testing)
- [ ] Update import paths in moved files
- [ ] Test all functionality still works after reorganization

### **Documentation Enhancement Standards**
- [ ] Add file classification headers to all Python files:
  ```python
  # File Classification: PRODUCTION | DIAGNOSTIC | CORE | TEST
  # Purpose: [One-line description]
  # Usage: [When/how to use this file]
  ```
- [ ] Enhance module docstrings for production scripts with:
  - Clear purpose and usage examples
  - Command-line syntax and options
  - Dependencies and prerequisites
- [ ] Update README.md project structure section to reflect new organization
- [ ] Create `docs/development_guidelines.md` with:
  - File organization principles
  - Documentation requirements
  - Naming conventions
  - Contribution guidelines

## Medium Priority - Code Quality

### **Import Path Standardization**
- [ ] Standardize import path setup pattern across all files
- [ ] Ensure consistent PROJECT_ROOT handling

### **Documentation Completeness**
- [ ] Document all public APIs in core modules
- [ ] Add docstrings to complex functions (>20 lines or >5 parameters)
- [ ] Create inline documentation for complex ML pipeline logic

## Low Priority - Developer Experience

### **Testing & Validation**
- [ ] Add unit tests for column mapping logic in merge_training_data.py
- [ ] Test end-to-end web→corrections→merge→retrain workflow
- [ ] Create integration tests for file reorganization
- [ ] Add automated checks for documentation completeness
- [ ] Set up pre-commit hooks for documentation standards

### **Deployment & Packaging**
- [ ] Update any deployment scripts for new file locations
- [ ] Consider packaging scripts for easier distribution
- [ ] Create installation/setup automation

---

## Implementation Notes

**Estimated Time Investment:**
- File reorganization: 1-2 hours
- Documentation enhancement: 2-3 hours  
- Testing & validation: 30 minutes
- **Total: 4-6 hours**

**Benefits:**
- Clear separation between user-facing and developer tools
- Professional project structure following Python conventions
- Better onboarding experience for new users
- Easier maintenance and future development
- Improved code discoverability and documentation

**Prerequisites:**
- Backup current working state
- Test all workflows before reorganization
- Update any hardcoded paths in documentation

---

*Created: 2025-08-09 - Based on comprehensive code quality review*
*Last updated: 2025-11-08 - Added column mapping fix completion*
*Priority: High impact, medium effort - tackle when dedicated time available*