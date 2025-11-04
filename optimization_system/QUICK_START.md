# Quick Start Guide - Updated System

## ✅ What's Already Working

The latest code includes:
1. **✅ MAVProxy Console** - Already enabled by default (shows in separate terminal)
2. **✅ EKF Initialization** - Proper 30s wait for all sensors
3. **✅ Port Fix** - Correct UDP connection to MAVProxy
4. **✅ Command Visibility** - Logs show all commands being sent

## 🚀 Run Your First Test

### Test Single SITL with Console

```bash
cd ~/Documents/MC07_tuning/optimization_system

# Source ArduPilot environment
. ~/.profile

# Run test script (opens console in separate window)
python3 test_sitl_startup.py
```

This will:
- Open **MAVProxy console** in separate xterm window
- Show all MAVLink commands in real-time
- Test connection and basic commands
- You can see exactly what's happening!

### Run Optimization with Console Visible

```bash
# Single instance with console (for debugging)
python3 main.py --phase phase1_rate --generations 1 --parallel 1 --speedup 1
```

**What you'll see:**
- Main terminal: Python optimization logs
- **Separate xterm window**: MAVProxy console showing:
  - Drone status
  - Commands being sent
  - Real-time telemetry
  - Pre-arm checks
  - Mode changes
  - Arming status

## 📊 Current System Architecture

```
┌─────────────────────────────────────────┐
│   Main Terminal (main.py)              │
│   - Optimization logs                   │
│   - Generation progress                 │
│   - Fitness scores                      │
└─────────────┬───────────────────────────┘
              │
              ├─> SITL Process (background)
              │
              ├─> MAVProxy Console (xterm window)  ← YOU CAN SEE THIS!
              │   - Shows all commands
              │   - Real-time telemetry
              │   - Drone status
              │
              └─> Python connects via UDP:14550
```

## 🔍 Troubleshooting

### No xterm Window Appearing?

Install xterm:
```bash
sudo apt-get install xterm
```

### Want to Disable Console?

Edit `sitl_manager.py` line ~191:
```python
cmd = [
    ...
    # "--console",  # Comment this out to disable xterm window
    ...
]
```

### Check What's Running

```bash
# See SITL processes
ps aux | grep arducopter

# See which ports are in use
netstat -tuln | grep -E "5760|14550"

# Kill all SITL
pkill -9 arducopter
pkill -9 sim_vehicle
```

## 📝 Key Configuration

### In main.py

```python
sitl_manager = SITLManager(
    num_instances=1,      # Start with 1 for debugging
    speedup=1,            # Real-time
)
```

### Connection Type (in sitl_manager.py)

```python
# Current setup (via MAVProxy):
connection_string = f"udp:127.0.0.1:{mavproxy_port}"  # Port 14550

# Alternative (direct to SITL - no console):
# connection_string = f"tcp:127.0.0.1:{sitl_port}"    # Port 5760
```

## 🎯 Next Steps

### 1. Test with Mission File (Coming Soon)

Instead of programmatic commands, load a mission file:
```python
# Future feature
mission_file = "missions/standard_test.waypoints"
upload_mission(connection, mission_file)
set_mode(connection, "AUTO")
```

### 2. Add Web Dashboard (Planned)

Browser-based monitoring:
- Real-time fitness plots
- SITL status
- Parameter tracking
- Control buttons (pause/stop)

### 3. Scale to Multiple Instances

Once single instance works:
```bash
# Run 4 parallel instances (only instance 0 shows console)
python3 main.py --phase phase1_rate --generations 10 --parallel 4 --speedup 1
```

## 📌 Important Files

- `test_sitl_startup.py` - Simple test script (good starting point)
- `main.py` - Full optimization system
- `sitl_manager.py` - SITL instance management
- `test_sequences.py` - Arm/takeoff/land sequences
- `CLAUDE.md` - Full project documentation

## 💡 Tips

1. **Start with 1 instance** - Get it working first
2. **Keep console open** - Watch what commands are sent
3. **Check logs** - Look in `/tmp/sitl_stdout_0.log`
4. **Be patient** - Initial EKF takes 30 seconds
5. **Monitor terminal** - Watch for "Waiting for EKF..." messages

## 🐛 Current Known Issues

None! Latest code should work. If you see issues:
1. Check ArduPilot path is correct
2. Ensure `.profile` is sourced
3. Verify xterm is installed
4. Check no other SITL running

## 📞 Need Help?

Check these in order:
1. This guide (QUICK_START.md)
2. Project docs (CLAUDE.md)
3. Test with simple script first (test_sitl_startup.py)
4. Check logs in /tmp/sitl_*.log
