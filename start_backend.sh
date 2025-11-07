#!/bin/bash
# Start script for backend API server
# Cleans up existing processes first, then starts the backend

set -e  # Exit on error

echo "🚀 Starting Drone Tuning Backend..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Store PIDs for cleanup
PYTHON_PID=""

# Flag to prevent double cleanup
CLEANUP_DONE=false

# Signal handler for cleanup
cleanup_on_exit() {
    # Prevent double execution
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true

    echo ""
    echo ""
    echo "🛑 Shutting down backend..."
    echo ""

    # Kill the Python process if running
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo "  → Stopping API server (PID: $PYTHON_PID)..."
        kill -TERM "$PYTHON_PID" 2>/dev/null || true

        # Wait up to 3 seconds for graceful shutdown
        for i in {1..3}; do
            if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            echo "  → Force stopping API server..."
            kill -9 "$PYTHON_PID" 2>/dev/null || true
        fi
    fi

    # Run comprehensive cleanup
    echo ""
    bash "$SCRIPT_DIR/cleanup.sh"
    echo ""
    echo "✅ Backend stopped successfully!"

    # Remove trap to prevent double execution
    trap - SIGINT SIGTERM EXIT
}

# Trap signals for cleanup
trap cleanup_on_exit SIGINT SIGTERM EXIT

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
echo "  Backend is running..."
echo "  Dashboard: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop gracefully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server and store its PID
python3 api_server.py &
PYTHON_PID=$!

echo "Backend PID: $PYTHON_PID"
echo ""

# Disable exit-on-error temporarily for wait
set +e

# Wait for the Python process (will be interrupted by signals)
wait $PYTHON_PID
EXIT_CODE=$?

# Re-enable exit-on-error
set -e

# If wait was interrupted by signal, the trap will handle cleanup
# If process exited normally, exit with its code
if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 130 ] && [ $EXIT_CODE -ne 143 ]; then
    echo ""
    echo "⚠️  Backend exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
