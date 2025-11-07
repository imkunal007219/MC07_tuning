#!/bin/bash
# Cleanup script for drone tuning system
# Kills all running SITL instances, MAVProxy, and backend processes

echo "🧹 Cleaning up all processes..."

# Function to kill processes with retry
kill_process() {
    local process_name=$1
    local pattern=$2

    echo "  → Stopping $process_name..."

    # Try graceful kill first (SIGTERM)
    pkill -TERM -f "$pattern" 2>/dev/null
    sleep 1

    # Force kill if still running (SIGKILL)
    if pgrep -f "$pattern" > /dev/null 2>&1; then
        pkill -9 -f "$pattern" 2>/dev/null && echo "    ✓ $process_name stopped (force)" || echo "    - No $process_name running"
    else
        if [ $? -eq 0 ]; then
            echo "    ✓ $process_name stopped"
        else
            echo "    - No $process_name running"
        fi
    fi
}

# Kill processes in order (most dependent first)
kill_process "API server" "api_server.py"
kill_process "uvicorn processes" "uvicorn"
kill_process "MAVProxy instances" "mavproxy"
kill_process "sim_vehicle.py instances" "sim_vehicle.py"
kill_process "ArduCopter SITL instances" "arducopter"
kill_process "optimization processes" "optimizer.py"

# Also kill any Python processes in ardupilot directory (might be stuck)
echo "  → Stopping ardupilot Python processes..."
pkill -9 -f "ardupilot.*python" 2>/dev/null || true

# Kill any xterm windows spawned by SITL
echo "  → Closing SITL terminal windows..."
pkill -9 -f "xterm.*ArduCopter" 2>/dev/null || true
pkill -9 -f "xterm.*MAVProxy" 2>/dev/null || true

# Clean up any screen sessions from SITL
echo "  → Cleaning up screen sessions..."
screen -ls | grep -o "[0-9]*\." | xargs -I {} screen -S {} -X quit 2>/dev/null || true

# Wait for processes to fully terminate
sleep 2

# Check if any processes are still running
REMAINING=$(ps aux | grep -E "(api_server|arducopter|mavproxy|sim_vehicle)" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ All processes cleaned up successfully!"
else
    echo "⚠️  Warning: $REMAINING process(es) may still be running"
    echo ""
    echo "Still running:"
    ps aux | grep -E "(api_server|arducopter|mavproxy|sim_vehicle)" | grep -v grep
    echo ""
    echo "If processes persist, try: killall -9 python3 arducopter mavproxy.py"
fi

exit 0
