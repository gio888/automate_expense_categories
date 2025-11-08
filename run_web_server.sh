#!/bin/bash
# Helper script to run the web server with the correct virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found at .venv/"
    echo "💡 Create it with: python3 -m venv .venv"
    echo "💡 Then install dependencies: source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: Virtual environment activation may have failed"
fi

# Show Python version being used
echo "🐍 Using Python: $(which python3)"
echo "📦 NumPy version: $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""

# Run the web server
python3 start_web_server.py
