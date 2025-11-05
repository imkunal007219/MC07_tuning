# Mission Execution Sequence Update

## Changes Made

Updated the mission execution sequence based on user's verified working method from manual testing.

## Previous (Incorrect) Sequence

```python
1. Upload mission
2. Verify mission
3. Wait for EKF
4. Arm vehicle
5. Set AUTO mode
```

**Problems:**
- Wrong order (arm before AUTO mode)
- Missing required parameters
- Missing RC 3 1500 command
- Insufficient sensor initialization time

## New (Correct) Sequence

```python
1. Set AUTO_OPTIONS = 1     # Allow arming in AUTO mode
2. Set ARMING_CHECK = 0     # Disable arming checks (SITL testing)
3. Load waypoints           # Using mission_loader
4. Wait 20 seconds          # Let ALL sensors stabilize (not just EKF)
5. Set mode AUTO            # BEFORE arming
6. Arm throttle             # In AUTO mode
7. Send RC 3 1500           # Throttle mid-stick (required)
```

## Code Changes

### File: `mission_executor.py`

#### Updated `run_mission()` method

**Line 50-54: Set Required Parameters**
```python
# Step 1: Set required parameters for AUTO mode arming
logger.info("Setting AUTO_OPTIONS and ARMING_CHECK parameters...")
self._set_parameter("AUTO_OPTIONS", 1)  # Allow arming in AUTO mode
self._set_parameter("ARMING_CHECK", 0)  # Disable arming checks for SITL testing
```

**Line 56-60: Load Waypoints**
```python
# Step 2: Load waypoints using correct MAVProxy method
logger.info(f"Loading waypoints: {mission_file}")
if not self._load_waypoints_mavproxy(mission_file):
    logger.error("Failed to load waypoints")
    return False, {}
```

**Line 62-65: Wait for Sensors (20 seconds)**
```python
# Step 3: Wait 20 seconds for all sensors to get ready
logger.info("Waiting 20 seconds for sensors to initialize...")
if not self._wait_for_sensors_ready(timeout=20):
    logger.warning("Sensor initialization timeout - proceeding anyway")
```

**Line 67-71: Set AUTO Mode FIRST**
```python
# Step 4: Set mode AUTO (BEFORE arming)
logger.info("Setting AUTO mode...")
if not self._set_mode("AUTO"):
    logger.error("Failed to set AUTO mode")
    return False, {}
```

**Line 73-77: Arm Vehicle**
```python
# Step 5: Arm throttle
logger.info("Arming vehicle in AUTO mode...")
if not self._arm_vehicle():
    logger.error("Failed to arm vehicle")
    return False, {}
```

**Line 79-82: Send RC Override**
```python
# Step 6: Send RC 3 1500 (throttle mid-stick) - required for mission start
logger.info("Setting throttle to mid-stick (RC 3 1500)...")
self._send_rc_override(channel=3, pwm=1500)
time.sleep(0.5)
```

#### New Helper Methods

**`_set_parameter()` - Line 187-221**
```python
def _set_parameter(self, param_name: str, param_value: float) -> bool:
    """Set a parameter value using PARAM_SET MAVLink message."""
    param_name_bytes = param_name.encode('utf-8')[:16]

    self.connection.mav.param_set_send(
        self.connection.target_system,
        self.connection.target_component,
        param_name_bytes,
        float(param_value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    # ... wait for confirmation ...
```

**`_load_waypoints_mavproxy()` - Line 223-247**
```python
def _load_waypoints_mavproxy(self, mission_file: str) -> bool:
    """
    Load waypoints using mission_loader.

    Equivalent to MAVProxy command: wp load <file>
    """
    if not self.mission_loader.load_and_upload(mission_file):
        return False

    count = self.mission_loader.verify_mission()
    if count <= 0:
        return False

    logger.info(f"✓ Mission loaded successfully ({count} waypoints)")
    return True
```

**`_wait_for_sensors_ready()` - Line 249-290**
```python
def _wait_for_sensors_ready(self, timeout: float = 20.0) -> bool:
    """
    Wait for sensors to be ready (EKF, GPS, etc.).

    Unlike _wait_for_ekf(), this waits the FULL timeout to let
    all sensors stabilize, not just until first position fix.
    """
    start_time = time.time()
    ekf_ready = False

    while time.time() - start_time < timeout:
        msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', ...)
        # Check EKF, then continue waiting full timeout
        if elapsed >= timeout:
            logger.info(f"✓ Sensors initialized ({timeout}s wait complete)")
            return True
```

**`_send_rc_override()` - Line 292-313**
```python
def _send_rc_override(self, channel: int, pwm: int):
    """Send RC channel override."""
    channels = [65535] * 8  # UINT16_MAX = no change
    channels[channel - 1] = pwm

    self.connection.mav.rc_channels_override_send(
        self.connection.target_system,
        self.connection.target_component,
        *channels
    )
```

## Parameters Explained

### AUTO_OPTIONS = 1
**Purpose:** Allows arming while in AUTO mode

**Default:** 0 (disallow)

**Why needed:** ArduPilot normally prevents arming in AUTO mode for safety. Setting this to 1 allows the mission to be armed and started automatically without manual intervention.

**MAVLink equivalent:**
```python
param_set AUTO_OPTIONS 1
```

### ARMING_CHECK = 0
**Purpose:** Disables pre-arm safety checks

**Default:** 1 (all checks enabled)

**Why needed:** In SITL testing, some checks may fail (e.g., GPS accuracy, compass calibration). Disabling allows testing without full sensor setup.

**MAVLink equivalent:**
```python
param_set ARMING_CHECK 0
```

**⚠️ WARNING:** Only disable in SITL/testing. NEVER disable on real hardware!

### RC 3 1500
**Purpose:** Sets throttle to mid-stick position

**Channel 3:** Throttle

**Value 1500:** Mid-stick (neutral for copters)

**Why needed:** ArduPilot requires RC input to be active before starting AUTO mission. This simulates pilot having throttle at mid-stick.

**MAVLink equivalent:**
```python
rc 3 1500
```

## Testing

### Test the Updated Sequence

```bash
cd ~/Documents/MC07_tuning/optimization_system
. ~/.profile
python3 test_mission_sequence.py
```

**Expected output:**
```
1. Setting AUTO_OPTIONS and ARMING_CHECK parameters...
   ✓ AUTO_OPTIONS set to 1.0
   ✓ ARMING_CHECK set to 0.0

2. Loading waypoints: missions/simple_hover.waypoints
   ✓ Mission loaded successfully (4 waypoints)

3. Waiting 20 seconds for sensors to initialize...
   ✓ EKF ready - Position: (-35.363261, 149.165237, 0.00m)
   ✓ Sensors initialized (20s wait complete)

4. Setting AUTO mode...
   ✓ Mode set to AUTO

5. Arming vehicle in AUTO mode...
   ✓ Vehicle armed

6. Setting throttle to mid-stick (RC 3 1500)...
   RC override: Channel 3 = 1500

7. Mission running - collecting telemetry...
   Reached waypoint 1
   Reached waypoint 2
   Reached waypoint 3
   Vehicle disarmed - mission complete

✓ Mission completed successfully
```

## Verification

To verify the sequence is correct, watch the MAVProxy console (separate xterm window):

```
MAV> AUTO_OPTIONS     1.0         # Parameter set
MAV> ARMING_CHECK     0.0         # Parameter set
MAV> Got MISSION_ITEM ...         # Waypoints loaded
MAV> EKF2 IMU1 is using GPS      # Sensors ready
MAV> Mode AUTO                    # Mode changed
MAV> ARMED                        # Armed successfully
MAV> Executing NAV_TAKEOFF        # Mission started
MAV> Reached waypoint #1
MAV> Executing NAV_LOITER_TIME
MAV> Reached waypoint #2
MAV> Executing NAV_LAND
MAV> DISARMED                     # Mission complete
```

## Comparison: Old vs New

| Step | Old Sequence | New Sequence | Why Changed |
|------|--------------|--------------|-------------|
| 1 | Upload mission | Set AUTO_OPTIONS=1 | Need parameter first |
| 2 | Verify mission | Set ARMING_CHECK=0 | Need parameter first |
| 3 | Wait for EKF only | Load waypoints | Proper order |
| 4 | Arm vehicle | Wait 20s for ALL sensors | More stable |
| 5 | Set AUTO mode | Set AUTO mode | Correct: mode before arm |
| 6 | Start mission | Arm throttle | Correct: arm after AUTO |
| 7 | - | Send RC 3 1500 | Missing in old version |
| 8 | Monitor | Monitor | Same |

## Benefits

✅ **Matches working manual method** - Uses exact sequence verified by user

✅ **Proper initialization** - 20 second wait for all sensors, not just EKF

✅ **Correct arming order** - AUTO mode set BEFORE arming

✅ **Required parameters** - AUTO_OPTIONS and ARMING_CHECK set

✅ **RC input** - Throttle mid-stick sent as required

✅ **More reliable** - Should eliminate "No altitude data" and arming failures

## Files Modified

- ✅ `mission_executor.py` - Updated mission execution sequence
- ✅ `test_mission_sequence.py` - New test for verification
- ✅ `MISSION_SEQUENCE_UPDATE.md` - This documentation

## Next Steps

1. **Test the new sequence:**
   ```bash
   python3 test_mission_sequence.py
   ```

2. **Run optimization** (uses new sequence automatically):
   ```bash
   python3 main.py --phase phase1_rate --generations 5 --parallel 1
   ```

3. **Verify logs** show correct sequence:
   - Parameters set first
   - 20-second sensor wait
   - AUTO mode before arming
   - RC override sent

## Troubleshooting

### If mission still fails to arm:

**Check 1:** Verify AUTO_OPTIONS parameter
```python
# In MAVProxy console, you should see:
AUTO_OPTIONS     1.0
```

**Check 2:** Verify mode is AUTO before arming
```python
# Logs should show:
Setting AUTO mode...
✓ Mode set to AUTO
Arming vehicle in AUTO mode...
```

**Check 3:** Verify RC override sent
```python
# Logs should show:
Setting throttle to mid-stick (RC 3 1500)...
RC override: Channel 3 = 1500
```

### If sensors timeout:

**Increase wait time** in mission_executor.py:
```python
# Line 63, change from 20 to 30 seconds:
if not self._wait_for_sensors_ready(timeout=30):
```

## References

**User's working manual sequence:**
```
1. wp load /path/to/waypoints.waypoints
2. param set AUTO_OPTIONS 1
3. param set ARMING_CHECK 0
4. (wait 20 seconds)
5. mode auto
6. arm throttle
7. rc 3 1500
```

**Code implementation:** `mission_executor.py:37-94`

**Test script:** `test_mission_sequence.py`
