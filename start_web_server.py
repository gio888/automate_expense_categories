#!/usr/bin/env python3
"""
Startup script for the Expense Categorization Web Interface
"""

import os
import sys
from pathlib import Path

# Ensure we're in the right directory
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print("🎯 Starting Expense Categorization Web Interface")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected")
        print("💡 Consider activating: source venv/bin/activate")
        print()
    
    try:
        # Test imports
        print("🔍 Checking dependencies...")
        import fastapi
        import uvicorn
        import pandas
        print("✅ Dependencies OK")
        
        # Test configuration
        print("🔧 Checking configuration...")
        from src.config import get_config
        config = get_config()
        if config.validate_required():
            print("✅ Configuration OK")
        else:
            print("⚠️  Configuration has issues - some features may not work")
        
        print()
        print("🚀 Starting web server...")
        print("📱 Open http://localhost:8000 in your browser")
        print("🔒 Server bound to localhost only for security")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        print()

        # Start the server
        # Security: Bind to localhost only to prevent network exposure
        from src.web.app import app
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Try: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)