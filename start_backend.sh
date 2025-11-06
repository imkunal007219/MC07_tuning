#!/bin/bash
# Start script for backend API server
# Cleans up existing processes first, then starts the backend

set -e  # Exit on error

echo "🚀 Starting Drone Tuning Backend..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run cleanup first
echo "Step 1: Cleaning up existing processes..."
bash cleanup.sh

echo ""
echo "Step 2: Starting backend server..."
cd backend

# Check if virtual environment exists
if [ -d "../venv" ]; then
    echo "  → Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check Python dependencies
echo "  → Checking dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "⚠️  Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}

echo "  → Starting API server on http://localhost:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend is starting..."
echo "  Dashboard: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop (may need to run cleanup.sh after)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server
python3 api_server.py
