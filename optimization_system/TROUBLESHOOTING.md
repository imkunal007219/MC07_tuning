# Troubleshooting Guide

## Issues Fixed

### Issue 1: test_mission_sequence.py Failed

**Error:**
```
TypeError: SITLManager.__init__() got an unexpected keyword argument 'show_console'
AttributeError: 'SITLManager' object has no attribute 'instances'
```

**Root Cause:**
1. Test script used `show_console` parameter that doesn't exist
2. Cleanup tried to access `self.instances` before __init__ completed

**Fix Applied:**
- Removed `show_console` parameter (console is always shown by default)
- Made `cleanup()` method safe to call even if __init__ failed
- Protected all attribute accesses in cleanup()

**Files Modified:**
- `test_mission_sequence.py` - Removed invalid parameter
- `sitl_manager.py` - Added safety checks in cleanup()

### Issue 2: Multiple Instances Created, Not Controlled

**Problem:**
- Running `python3 main.py --parallel 4` created more than 4 instances (5th, 6th, 7th)
- Only 4 connected, others showed "link down"
- None were controlled by automation - waited 1+ min with no commands sent

**Likely Root Causes:**

1. **Instance Pool Exhaustion:**
   - If genetic algorithm population size > num_instances, it will try to create more
   - Each generation might spawn too many workers

2. **Mission Execution Not Starting:**
   - The automation code might not be reaching the mission execution step
   - Could be a path issue with mission file
   - Could be a connection issue

**Debug Steps:** (See below)

## Testing Strategy

To isolate the problem, test in this order:

### Step 1: Test Mission Execution Alone

Test the mission execution without SITL management complexity:

```bash
cd ~/Documents/MC07_tuning/optimization_system

# Terminal 1 - Start SITL manually
cd ~/Documents/MC07_tuning/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -f drone-30kg --console

# Wait for "EKF2 IMU1 is using GPS" message

# Terminal 2 - Run mission test
cd ~/Documents/MC07_tuning/optimization_system
. ~/.profile
python3 test_simple_mission.py
```

**Expected Result:**
```
Setting AUTO_OPTIONS and ARMING_CHECK parameters...
✓ AUTO_OPTIONS set to 1.0
✓ ARMING_CHECK set to 0.0
Loading waypoints...
✓ Mission loaded successfully (4 waypoints)
Waiting 20 seconds for sensors...
✓ Sensors initialized
Setting AUTO mode...
✓ Mode set to AUTO
Arming vehicle...
✓ Vehicle armed
Setting throttle to mid-stick...
Mission running...
✓ Mission completed successfully
```

**If this fails:** The problem is in mission_executor.py sequence

**If this works:** The problem is in SITL management or optimization loop

### Step 2: Test SITL Manager with Single Instance

```bash
. ~/.profile
python3 test_mission_sequence.py
```

This tests:
- SITL manager can start instances
- Connection establishment
- Mission execution through SITL manager

**If this fails:** The problem is in sitl_manager.py

**If this works:** The problem is in parallel execution or optimization loop

### Step 3: Test with Small Population

```bash
. ~/.profile
python3 main.py --phase phase1_rate --generations 1 --parallel 2 --speedup 1
```

This tests the full optimization with minimal complexity (1 generation, 2 instances).

## Common Issues and Solutions

### Issue: "ArduPilot directory not found"

**Symptom:**
```
FileNotFoundError: ArduPilot directory not found. Searched in:
  - /home/user/MC07_tuning/ardupilot
  - ~/Documents/MC07_tuning/ardupilot
  ...
```

**Solution:**
Check where ArduPilot is installed:
```bash
ls ~/Documents/MC07_tuning/
```

If ardupilot is missing, the path detection is wrong. You can:

1. **Specify path explicitly in test:**
```python
sitl_manager = SITLManager(
    num_instances=1,
    speedup=1,
    ardupilot_path="/path/to/your/ardupilot"
)
```

2. **Or symlink it:**
```bash
ln -s /your/real/path/ardupilot ~/Documents/MC07_tuning/ardupilot
```

### Issue: "Mission file not found"

**Symptom:**
```
Mission file not found: /path/to/missions/simple_hover.waypoints
```

**Solution:**
Check if mission file exists:
```bash
ls ~/Documents/MC07_tuning/optimization_system/missions/
```

If missing, the missions directory wasn't created. It should contain:
- `simple_hover.waypoints`
- `README.md`

### Issue: Connection Timeout

**Symptom:**
```
Connection timeout on udp:127.0.0.1:14550
```

**Solution:**

1. **Check if SITL is running:**
```bash
ps aux | grep arducopter
ps aux | grep mavproxy
```

2. **Check if port is correct:**
```bash
netstat -tuln | grep 14550
```

3. **Check MAVProxy console** (xterm window) - should show:
```
MAV> link 1 OK
```

### Issue: "No altitude data received"

**Symptom:**
```
Waiting for EKF initialization...
No altitude data received
```

**Root Cause:**
- EKF not initialized yet
- Sensors not ready

**Solution:**
The new code already waits 20 seconds. If still seeing this:

1. **Increase wait time** in mission_executor.py:
```python
# Line 64, change timeout from 20 to 30:
if not self._wait_for_sensors_ready(timeout=30):
```

2. **Check SITL console** for EKF status:
```
MAV> EKF2 IMU1 is using GPS  # Should see this
```

### Issue: "Arming failed"

**Symptom:**
```
Arming vehicle in AUTO mode...
Failed to arm vehicle
```

**Common Causes:**

1. **AUTO_OPTIONS not set:**
   - Check logs: should see "AUTO_OPTIONS set to 1.0"
   - Watch MAVProxy console for parameter change

2. **Mode not AUTO before arming:**
   - Check logs: should see "Mode set to AUTO" BEFORE "Arming vehicle"

3. **Pre-arm checks failing:**
   - Should see "ARMING_CHECK set to 0.0" in logs
   - But can still fail if critical checks fail

**Debug in MAVProxy console:**
```
MAV> param show AUTO_OPTIONS
AUTO_OPTIONS     1.0

MAV> param show ARMING_CHECK
ARMING_CHECK     0.0

MAV> arm throttle
ARMED

# If arm fails, check:
MAV> arm list  # Shows why arming is blocked
```

### Issue: Vehicle Arms but Mission Doesn't Start

**Symptom:**
```
✓ Vehicle armed
Setting throttle to mid-stick...
Mission running...
(nothing happens for 60+ seconds)
```

**Possible Causes:**

1. **RC override not sent or not working:**
   - Should see: "RC override: Channel 3 = 1500" in logs

2. **Mission not uploaded:**
   - Check logs for: "Mission loaded successfully (4 waypoints)"

3. **Wrong mode:**
   - Should be in AUTO mode when armed
   - Check MAVProxy console: should show "Mode AUTO"

**Debug:**
In MAVProxy console, after arming:
```
MAV> mode
Mode: AUTO

MAV> wp list
# Should show 4 waypoints

MAV> rc 3 1500  # Manually set throttle if override didn't work
```

### Issue: Instances Not Being Controlled in Parallel Mode

**Symptom:**
```
Multiple SITL windows open
All showing "link 1 down" or no commands being sent
Waited 1+ minutes, nothing happens
```

**Likely Causes:**

1. **Population size > num_instances:**
   - Check config.py: `population_size`
   - Should be ≤ num_instances for first test
   - Example: If `--parallel 4`, population should be ≤ 4

2. **Thread deadlock:**
   - All workers waiting for instances
   - Instance queue empty but instances not released

3. **Exception in worker thread:**
   - Check logs for tracebacks
   - Workers might be silently failing

**Debug:**
Add more logging to see what's happening:

```bash
# Run with debug logging
python3 main.py --parallel 2 --generations 1 2>&1 | tee debug.log

# Then search the log for:
grep "Evaluating population" debug.log
grep "instance_id" debug.log
grep "Mission" debug.log
```

**Expected log flow for each instance:**
```
Evaluating population of N individuals...
Worker 0: Got instance 0
Instance 0: Setting AUTO_OPTIONS...
Instance 0: Loading waypoints...
Instance 0: Waiting for sensors...
Instance 0: Setting AUTO mode...
Instance 0: Arming...
Instance 0: Mission running...
Instance 0: Mission complete
Worker 0: Released instance 0
```

**If you don't see this pattern:** The workers aren't reaching the mission execution code.

## Quick Fixes

### Fix 1: Reduce Population Size

Edit `config.py`:
```python
OPTIMIZATION_CONFIG = {
    'population_size': 4,  # Change from 50 to match --parallel
    ...
}
```

### Fix 2: Increase Timeouts

If sensors/EKF timeout, edit `mission_executor.py`:
```python
# Line 64
if not self._wait_for_sensors_ready(timeout=30):  # Increase from 20 to 30
```

### Fix 3: Force Single-Threaded for Debugging

Test with 1 instance to eliminate threading issues:
```bash
python3 main.py --parallel 1 --generations 1 --speedup 1
```

## Getting Help

When reporting issues, please provide:

1. **Exact command run:**
   ```
   python3 main.py --parallel 4 --generations 5
   ```

2. **Complete error output** or logs

3. **What you see in MAVProxy console(s)**

4. **Output of:**
   ```bash
   ps aux | grep -E '(arducopter|mavproxy|sim_vehicle)'
   netstat -tuln | grep -E '(14550|14560|5760|5770)'
   ```

5. **Config settings:**
   ```bash
   grep population_size config.py
   ```

## Next Steps After Fixing

Once basic tests work:

1. ✅ `test_simple_mission.py` - Mission execution works
2. ✅ `test_mission_sequence.py` - SITL manager works
3. ✅ `main.py --parallel 1 --generations 1` - Optimization loop works
4. ✅ `main.py --parallel 2 --generations 2` - Parallel execution works
5. ✅ `main.py --parallel 4 --generations 5` - Full test

Then you can scale up to full optimization runs.
