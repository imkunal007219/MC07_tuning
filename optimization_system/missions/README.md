# Mission Files

This directory contains waypoint mission files used for automated drone testing.

## Mission Format

Files use QGroundControl WPL 110 format (MAVLink waypoint format).

## Available Missions

### simple_hover.waypoints ⭐ **RECOMMENDED FOR INITIAL TESTING**
**Duration:** ~40 seconds
**Pattern:** Simple takeoff, hover, land
**Sequence:**
1. Takeoff to 10m
2. Hover for 30 seconds at home position
3. Land

**Use Case:** Initial testing and basic parameter validation
- **Start here!** Simplest mission to verify drone can fly
- Tests basic stability
- Tests takeoff/land
- Minimal complexity - good for debugging
- Fast execution

### standard_test.waypoints
**Duration:** ~60 seconds
**Pattern:** Square pattern at 20m altitude
**Sequence:**
1. Takeoff to 10m
2. Climb to 20m at home position
3. Fly square pattern (50m x 50m)
4. Return to home
5. Land

**Use Case:** Full parameter optimization test
- Tests stability during hover
- Tests position control during waypoint navigation
- Tests altitude hold
- Tests landing accuracy
- More comprehensive than simple_hover

## Mission File Format

```
QGC WPL 110
SEQ	CURRENT	FRAME	COMMAND	P1	P2	P3	P4	X(LAT)	Y(LON)	Z(ALT)	AUTOCONTINUE
```

### Common Commands
- `16` = MAV_CMD_NAV_WAYPOINT (fly to waypoint, P1=hold time)
- `22` = MAV_CMD_NAV_TAKEOFF (takeoff to altitude)
- `21` = MAV_CMD_NAV_LAND (land at current position)

### Frames
- `0` = MAV_FRAME_GLOBAL (absolute altitude)
- `3` = MAV_FRAME_GLOBAL_RELATIVE_ALT (relative to home)

## Usage

```python
from mission_loader import upload_mission_file

# Upload mission
success = upload_mission_file(connection, "missions/simple_hover.waypoints")

# Start mission
connection.mav.command_long_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_CMD_MISSION_START,
    0, 0, 0, 0, 0, 0, 0, 0
)

# Set AUTO mode
connection.set_mode("AUTO")
```

## Adding New Missions

1. Create `.waypoints` file in this directory
2. Follow QGC WPL 110 format
3. Test with single SITL instance first
4. Update this README with mission details
