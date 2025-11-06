#!/bin/bash
# Cleanup script for drone tuning system
# Kills all running SITL instances, MAVProxy, and backend processes

echo "🧹 Cleaning up all processes..."

# Kill API server
echo "  → Stopping API server..."
pkill -9 -f "api_server.py" 2>/dev/null && echo "    ✓ API server stopped" || echo "    - No API server running"

# Kill all ArduCopter SITL instances
echo "  → Stopping ArduCopter SITL instances..."
pkill -9 -f "arducopter" 2>/dev/null && echo "    ✓ ArduCopter instances stopped" || echo "    - No ArduCopter running"

# Kill all MAVProxy instances
echo "  → Stopping MAVProxy instances..."
pkill -9 -f "mavproxy" 2>/dev/null && echo "    ✓ MAVProxy stopped" || echo "    - No MAVProxy running"

# Kill sim_vehicle.py processes
echo "  → Stopping sim_vehicle.py instances..."
pkill -9 -f "sim_vehicle.py" 2>/dev/null && echo "    ✓ sim_vehicle.py stopped" || echo "    - No sim_vehicle.py running"

# Kill any hanging Python processes related to optimization
echo "  → Stopping optimization processes..."
pkill -9 -f "optimizer.py" 2>/dev/null && echo "    ✓ Optimizer stopped" || echo "    - No optimizer running"

# Clean up any orphaned uvicorn processes
echo "  → Stopping uvicorn processes..."
pkill -9 -f "uvicorn" 2>/dev/null && echo "    ✓ Uvicorn stopped" || echo "    - No uvicorn running"

# Wait a moment for processes to terminate
sleep 1

# Check if any processes are still running
REMAINING=$(ps aux | grep -E "(api_server|arducopter|mavproxy|sim_vehicle)" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ All processes cleaned up successfully!"
else
    echo "⚠️  Warning: $REMAINING process(es) may still be running"
    echo "Run 'ps aux | grep -E \"(api_server|arducopter|mavproxy)\"' to check"
fi

exit 0
