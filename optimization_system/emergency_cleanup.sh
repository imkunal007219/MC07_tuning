#!/bin/bash
# Emergency cleanup script for stuck SITL/MAVProxy processes
# Run this if Ctrl+C doesn't work and terminal is hung

echo "Emergency SITL Cleanup - Killing all related processes..."

# Kill all ArduPilot SITL processes
pkill -9 arducopter 2>/dev/null
pkill -9 ArduCopter 2>/dev/null

# Kill all MAVProxy processes
pkill -9 mavproxy.py 2>/dev/null
pkill -9 -f mavproxy 2>/dev/null

# Kill sim_vehicle scripts
pkill -9 sim_vehicle 2>/dev/null
pkill -9 -f sim_vehicle.py 2>/dev/null

# Kill any xterm windows (MAVProxy console)
pkill -9 xterm 2>/dev/null

# Kill any Python processes related to SITL/MAVProxy
pkill -9 -f "run_in_terminal_window" 2>/dev/null

# Clean up temp directories
rm -rf /tmp/sitl_instance_* 2>/dev/null

echo "Cleanup complete! Checking for remaining processes..."

# Check if anything is still running
remaining=$(ps aux | grep -E '(arducopter|mavproxy|sim_vehicle)' | grep -v grep)
if [ -z "$remaining" ]; then
    echo "✓ All processes killed successfully"
else
    echo "⚠ Some processes may still be running:"
    echo "$remaining"
    echo ""
    echo "If processes are still stuck, try:"
    echo "  sudo killall -9 arducopter mavproxy.py sim_vehicle.py"
fi

echo ""
echo "You can now safely run tests again."
